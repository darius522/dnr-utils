
from pathlib import Path
import shutil
from audio_utils import peak_norm
import os

from numpy.core.defchararray import partition
import numpy as np
import random
from itertools import repeat
from scipy import stats
import soundfile as sf
from glob import glob
from matplotlib import pyplot as plt
import pandas as pd
import collections
from tqdm import tqdm
from sacred import Experiment
from config import config_ingredient
import utils, audio_utils

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

@config_ingredient.capture
def aug_gain(config, audio, low=0.0, high=1.0):
    gain = np.random.uniform(low=low, high=high)
    return audio * gain

@config_ingredient.capture
def aug_reverb(config, audio, ir):
    pass

def gen_poisson(mu):
    """Adapted from https://github.com/rmalouf/learning/blob/master/zt.py"""
    r = np.random.uniform(low=stats.poisson.pmf(0, mu))
    return int(stats.poisson.ppf(r, mu))

def gen_norm(mu, sig):
    return np.random.normal(loc=mu, scale=sig)

def gen_skewnorm(skew, mu, sig):
    # negative a means skewed left while positive means skewed right; a=0 -> normal
    return stats.skewnorm(a=skew, loc=mu, scale=sig).rvs()

def get_some_noise(shape):
    return np.random.randn(shape).astype(np.float32)

class MixtureObj():
    def __init__(self, 
        seq_dur=60.0, 
        sr=44100, 
        files=None, 
        mix_lufs=None, 
        partition='train', 
        peak_norm_db=None,
        annot_path = None,
        class_dirs = None):

        self.seq_dur = seq_dur
        self.sr = sr
        self.input_dirs = class_dirs
        self.annot_path = annot_path
        self.mix = np.zeros(int(self.seq_dur*self.sr))
        self.mix_lufs = mix_lufs
        self.partition = partition
        self.allocated_windows = {'music'     :{'times':[],'samples':[],'levels':[],'files':[]},
                                  'sfx'       :{'times':[],'samples':[],'levels':[],'files':[]},
                                  'background':{'times':[],'samples':[],'levels':[],'files':[]},
                                  'speech'    :{'times':[],'samples':[],'levels':[],'files':[]}}

        # Set some submix-specific values
        self.submixes = {'music'     :[],
                         'sfx'       :[],
                         'background':[],
                         'speech'    :[]}

        self.mix_max_peak = 0.0
        self.peak_norm = audio_utils.db_to_gain(peak_norm_db)

        self.files = files

        self.num_seg_music      = gen_poisson(7.0)
        self.num_seg_sfx        = gen_poisson(12.0)
        self.num_seg_speech     = gen_poisson(8.0)
        self.num_seg_background = gen_poisson(6.0)
    
    def check_overlap_for_range(self, r1, c):
        ranges = self.allocated_windows[c]['times']
        for r2 in ranges:
            if not (np.sum(r1) < r2[0] or r1[0] > np.sum(r2)):
                return True
        return False

    def _check_for_files_exhaustion(self):
        if len(self.files['speech']) < self.num_seg_speech:
            return True
        return False
        
    def _load_wav(self, file):
        d, sr = sf.read(file)
        assert sr == self.sr, "Warning: Sampling rate of file {file} differs from expected sampling rate {self.sr}"
        return d

    def _register_event(self, mix_pos, length, cl, file, clip_start, clip_end, clip_gain):
        self.allocated_windows[cl]['files'].append(file)
        self.allocated_windows[cl]['times'].append([mix_pos,length])
        self.allocated_windows[cl]['samples'].append([clip_start,clip_end])
        self.allocated_windows[cl]['levels'].append(clip_gain)
    
    def _update_event_levels(self, gain):
        for c in self.allocated_windows.keys():
           self.allocated_windows[c]['levels'] = [gain * x for x in self.allocated_windows[c]['levels']]

    def _get_time_range_for_submix(self, data, c, idx, num_events):
        '''
        Compute time window for sound in submix

        Args:
            data: array sound [Nspl, Nch]
            c: the sound class [music, sfx, speech]
            idx: the sound index
        '''
        file_len = (len(data) / self.sr)
    
        # We don't want two mix of the same classto overlap
        min_start = 0.0 if not len(self.allocated_windows[c]['times'])\
                        else np.ceil(np.sum(self.allocated_windows[c]['times'][-1]))

        # Check if we have enough time remaining (at least 30% length of current file)
        if (self.seq_dur - min_start <= file_len * 0.3):
            return None, None

        # Pick start based on previous event pos.
        start   = min(gen_skewnorm(skew=5, mu=int(min_start + 1), sig=2.0), self.seq_dur - 2.0)
        max_len = file_len if self.seq_dur - start >= file_len else (self.seq_dur - start)

        # If speech, take the whole thing
        if c == 'speech':
            length = max_len * 0.99
        else:
            length = max(min(gen_norm(mu=max_len/2.0, sig=max_len/10.0), max_len), 0.1)

        return start, length

    def _create_from_annots(self, annot_path):
        '''
        Create the mixture strictly from a given annotation file
        '''
        columns = ['file',
                   'class',
                   'mix_start_time',
                   'mix_end_time',
                   'clip_start_time',
                   'clip_end_time',
                   'clip_gain',
                   'annotation']
        annots = pd.read_csv(annot_path,names=columns, skiprows=1)
        self.submixes = {k:np.zeros_like(self.mix) for k in self.submixes.keys()}

        for index, row in annots.iterrows():

            f_class = row['class']
            abs_path = os.path.join(self.input_dirs[f_class], '/'.join(row['file'].split('/')[1:]))
            data, _ = sf.read(abs_path)

            mix_start  = int(row['mix_start_time']*self.sr)
            mix_end    = int(row['mix_end_time']*self.sr)
            clip_start = int(row['clip_start_time']*self.sr)
            clip_end   = int(row['clip_end_time']*self.sr)

            clip_gain = float(row['clip_gain'])

            length = min(len(data), clip_end-clip_start, mix_end-mix_start)

            clip = data[clip_start:clip_start+length] * clip_gain
            self.submixes[f_class][mix_start:mix_start+length] += clip
        
        # Merge sfx and background into one submix
        self.submixes['sfx'] += self.submixes['background']
        self.submixes.pop('background', None)

        # Finally collapse submixes
        self.mix = np.sum(np.stack([self.submixes[c] for c in self.submixes.keys()],-1),-1)  

    def _set_submix(self, c, num_seg):

        submix = np.zeros(int(self.seq_dur*self.sr))
        class_lufs   = np.random.uniform(self.mix_lufs[c]-self.mix_lufs['ranges']['class'],
                                         self.mix_lufs[c]+self.mix_lufs['ranges']['class'])

        for i in range(num_seg):
            # If eval and speech, mix without replacement until exhaustion
            if self.partition == 'eval' and c == 'speech':
                f = self.files[c].pop(random.randrange(len(self.files[c])))
            else:
                f = np.random.choice(self.files[c])

            d = self._load_wav(f)
            s, l = self._get_time_range_for_submix(d, c, i, num_seg)
            clip_lufs = np.random.uniform(class_lufs-self.mix_lufs['ranges']['clip'],
                                          class_lufs+self.mix_lufs['ranges']['clip'])
            if s != None and l != None:
                try:
                    s_clip, l_clip = int(s * self.sr), int(l * self.sr)
                    r_spl = np.random.randint(0,len(d)-l_clip) if c == 'music' else 0
                    data = np.copy(d[r_spl:r_spl+l_clip])
                    data_norm, gain = audio_utils.lufs_norm(data=data,
                                                            sr=self.sr,
                                                            norm=clip_lufs)
                    submix[s_clip:s_clip+l_clip] = data_norm
                    self._register_event(mix_pos=s, 
                                         length=l, 
                                         cl=c, 
                                         file=f, 
                                         clip_start=r_spl/self.sr, 
                                         clip_end=(r_spl/self.sr)+l, 
                                         clip_gain=gain)
                except Exception as e: 
                    print(e)
                    print('could not register event')
                    print(f, c, s, l, s_clip, l_clip)
        self.mix_max_peak = max(self.mix_max_peak, np.max(np.abs(submix)))
        self.submixes[c] = submix

    def _create_final_mix(self):
        # Add some noise to sfx submix
        rand_range = self.mix_lufs['ranges']['class']

        # Set the peak norm gain
        peak_norm_gain = 1.0 if self.mix_max_peak <= self.peak_norm else self.peak_norm / self.mix_max_peak
        # Compute master gain
        master_lufs = np.random.uniform(self.mix_lufs['master']-rand_range,
                                        self.mix_lufs['master']+rand_range)

        # After adding to master, peak normalize the submixes
        for c in self.submixes.keys():
            # Peak norm
            self.submixes[c] *= peak_norm_gain
            # Add submix to main mix
            self.mix += self.submixes[c]

        self.mix, master_gain = audio_utils.lufs_norm(data=self.mix,
                                                      sr=self.sr,
                                                      norm=master_lufs)

        # Main LUFS norm
        for c in self.submixes.keys():
            self.submixes[c] *= master_gain

        # Optionally add some noise to the mix (i.e. avoid digital silences)
        if self.mix_lufs['noise'] != None:
            noise_lufs = np.random.uniform(self.mix_lufs['noise']-rand_range,
                                        self.mix_lufs['noise']+rand_range)
            noise    = get_some_noise(shape=int(self.seq_dur*self.sr))
            noise, _ = audio_utils.lufs_norm(data=noise,
                                            sr=self.sr,
                                            norm=noise_lufs)
            self.mix += noise

        # After peak norm / master norm, we need to update the registered events' props
        self._update_event_levels(master_gain*peak_norm_gain)

    def __call__(self):
        # Check if we build from the annotation or not
        if self.annot_path:
            self._create_from_annots(annot_path=self.annot_path)
            return self.submixes, self.mix, None, None
        # If not, Create submixes from scratch
        else:
            if not self._check_for_files_exhaustion():
                self._set_submix(c='music', num_seg=self.num_seg_music)
                self._set_submix(c='speech', num_seg=self.num_seg_speech)
                self._set_submix(c='sfx', num_seg=self.num_seg_sfx)
                self._set_submix(c='background', num_seg=self.num_seg_background)
                self._create_final_mix()

                # Merge sfx and background into one submix
                self.submixes['sfx'] += self.submixes['background']
                self.submixes.pop('background', None)

                return self.submixes, self.mix, self.allocated_windows, self.files
            else:
                return None, None, None, self.files

class DatasetCompiler():
    def __init__(
        self,
        seq_dur,
        num_files: int,
        input_dirs: dict = None,
        ground_truth_dirs: dict = None,
        output_dir: str = "../compiled_mixtures",
        alt_dir: str = None,
        sample_rate: float = 44100.0,
        partition: str = "tr",
        mix_lufs: dict = None,
        peak_norm_db: float = -0.5
    ) -> None:

        self.output_files     = []
        self.seq_dur   = seq_dur
        self.sr        = sample_rate
        self.partition = partition
        self.num_files = int(1e9) if partition == 'eval' else num_files
        self.alt_dir = alt_dir

        # If building from metadata files, we expect the directory containing the metadata files
        if self.alt_dir:
            self.alt_dir = os.path.join(alt_dir,self.partition)
            self.annot_files = self._get_annot_paths(self.alt_dir)

        self.input_dirs = input_dirs
        self.output_dir = output_dir
        self.wavfiles = {'music'     : self._get_filepaths(input_dirs['music']),
                         'speech'    : self._get_filepaths(input_dirs['speech'],c='speech'),
                         'sfx'       : self._get_filepaths(input_dirs['sfx']),
                         'background': self._get_filepaths(input_dirs['background'])}

        self.mix_lufs = mix_lufs
        self.peak_norm_db = peak_norm_db

        self.music_gt = pd.read_csv(ground_truth_dirs['music'])
        self.sfx_gt   = pd.read_csv(ground_truth_dirs['sfx'])
        self.df_gt    = pd.DataFrame(columns = ['fname','class','times','labels'])

        self.stats = {'overlap':{},'times':{}}

    def _get_mix_stats(self, data, res):
        '''
        Compute amount of overlap between given classes
        Args:
            data: times
            res: resolution of the timesteps
        '''
        # Average sample length
        for c in data.keys():
            for t in data[c]['times']:
                self.stats['times'].setdefault(c,[]).append(t[-1])

        # Merge background and sfx
        data['sfx']['times'] += data['background']['times']
        data.pop('background', None)

        overlap   = 0.0
        classes   = list(data.keys())
        timesteps = np.arange(0,self.seq_dur,res)
        labels    = []

        # Walk through steps
        for i, j in enumerate(timesteps):
            # Check if each class is present at this step
            detect = []
            for c in classes:
                for t in data[c]['times']:
                    s, e = t[0], sum(t)
                    if j >= s and j <= e:
                        detect.append(c)
                        break
            labels.append('-'.join(sorted(detect)))

        counter=collections.Counter(labels)

        values = list(counter.values())
        keys   = list(counter.keys())
        values = [x/sum(values) for x in values]
        for k,v in zip(keys,values):
            self.stats['overlap'].setdefault(k,[]).append(v)

        return overlap
    
    def _plot_stats(self):

        plt.rcParams.update({'font.size': 25})

        # Build the plot
        fig, ax = plt.subplots(2,1,figsize=(25,25))

        for i, stats in enumerate(self.stats.keys()):

            data = self.stats[stats]
            classes  = list(data.keys())

            x_pos = np.arange(len(classes))
            means = [np.mean(data[k]) for k in classes]
            error = [np.std(data[k]) for k in classes]
            try:
                classes[classes.index('')] = 'silence'
            except:
                pass

            ax[i].bar(x_pos, means, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
            ax[i].set_xticks(x_pos)
            ax[i].set_xticklabels(classes, rotation = 45, va="center", position=(0,-0.28))
            ax[i].yaxis.grid(True)

        ax[0].set_ylabel('Presence Amount')
        ax[0].set_title(self.partition+' - Classes Overlap ('+str(len(self.output_files))+' Mixtures)')
        ax[1].set_ylabel('Length (s)')
        ax[1].set_title(self.partition+' - Average Sample Length per Classes ('+str(len(self.output_files))+' Mixtures)')

        # Save the figure and show
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir,self.partition+'_set_stats.png'))

    def _get_filepaths(self, dir, c=''):
        files = glob(dir+'/**/*.wav', recursive=True)
        # If eval and speech, repeat list twice
        if self.partition == 'eval' and c == 'speech':
            return [x for item in files for x in repeat(item, 2)]
        return files

    def _get_annot_paths(self, dir):
        return glob(dir+'/**/annots.csv', recursive=True)

    def _write_wav(self, f_dir, sr, data, source):
        audio_utils.validate_audio(data)
        sf.write(os.path.join(f_dir,source+'.wav'), data, sr, subtype='FLOAT')

    def _write_ground_truth(self, f_dir, data):

        out_df = pd.DataFrame(columns = ['file','class',
                                        'mix start time','mix end time',
                                        'clip start sample','clip end sample',
                                        'clip gain',
                                        'annotation'])

        sources =  list(data.keys())
        # Sort events in ascending order, time-wise
        all_data = [[t,spl,f,s,l] for s in sources 
                            for t, spl, f, l in zip(data[s]['times'],
                                                    data[s]['samples'],
                                                    data[s]['files'],
                                                    data[s]['levels'])]
        sorted_data = sorted(all_data,
                            key=lambda x: x[0][0])

        # Retrieve annotations and write to pd frame
        for e in sorted_data:
            time, spl, file, source, level = e
            f_id  = os.path.basename(file).split('.')[0]
            annot = None
            if source == 'background' or source == 'sfx':
                annot = self.sfx_gt[self.sfx_gt['fname']==int(f_id)]['labels'].values[0]
            elif source == 'music':
                annot = self.music_gt[self.music_gt['track_id']==int(f_id)]['genres'].values[0]
            elif source == 'speech':
                readfile = open('-'.join(file.split('-')[:-1]) + '.trans.txt', "r").read()
                line  = readfile.find(f_id) + len(f_id) + 1
                annot = readfile[line:].split('\n')[0]
            
            row = [file, source, time[0], sum(time), spl[0], spl[1], level, annot]
            out_df.loc[len(out_df.index)] = row 

        out_df.to_csv(os.path.join(f_dir,'annots.csv'))

    def __call__(self):
        
        # If no alt-dir is specified, build dnr from scratch
        if not self.alt_dir:
            print('Building DNR from scratch...')
            files = self.wavfiles
            for i in tqdm(range(self.num_files)):
                sources, mix, annots, files = MixtureObj(seq_dur=self.seq_dur, 
                                                        partition=self.partition,
                                                        sr=self.sr, 
                                                        files=self.wavfiles,
                                                        mix_lufs=self.mix_lufs,
                                                        peak_norm_db=self.peak_norm_db)()

                # For eval set, we drain the speech dataset until empty
                if sources:
                    f_name, f_dir = utils.make_unique_filename(self.output_files, self.output_dir)
                    self._write_ground_truth(f_dir, annots)
                    sources['mix'] = mix
                    for source in list(sources.keys()):
                        self._write_wav(f_dir, int(self.sr), sources[source], source)
                    self.output_files.append(f_name)
                    self._get_mix_stats(annots, 0.1)
                else:
                    print('Reached the end of one dataset with remaining files:')
                    for k in files:
                        print(k,len(files[k]))
                    break
            
            self._plot_stats()
        else:
            print('Building DNR from annotation files...')
            for path in tqdm(self.annot_files):
                mix_id = path.split('/')[-2]
                f_dir = Path(os.path.join(self.output_dir, mix_id))
                f_dir.mkdir(parents=True, exist_ok= True)
                shutil.copy(path,f_dir)
                sources, mix, _, _ = MixtureObj(annot_path=path, 
                                                class_dirs=self.input_dirs,
                                                sr=self.sr,
                                                mix_lufs=self.mix_lufs,
                                                peak_norm_db=self.peak_norm_db)()
                sources['mix'] = mix
                for source in list(sources.keys()):
                    self._write_wav(f_dir, int(self.sr), sources[source], source)

ex = Experiment('Dataset Compilation', ingredients=[config_ingredient])

@ex.automain
def main(cfg):

    config = cfg['dst_config']

    set_seed(config["seed"])

    alt_dir = None if config['alt_dir'] == None or not os.path.exists(config['alt_dir'])\
                    else config['alt_dir']
    output_dir = utils.create_dir(os.path.join(config['output_dir'], config['partition']))
    utils.json_from_dict(config,output_dir)
    
    files_dir_dict = {'music'     : os.path.join(config['source_root'],'FMA',config['partition']),
                      'speech'    : os.path.join(config['source_root'],'LibriSpeech',config['partition']),
                      'sfx'       : os.path.join(config['source_root'],'FSD50K','foreground',config['partition']),
                      'background': os.path.join(config['source_root'],'FSD50K','background',config['partition'])}

    gt_dir_dict = {'music'     : os.path.join(config['source_root'],'FMA','fma_genres_ground_truth.csv'),
                   'speech'    : None,
                   'sfx'       : os.path.join(config['source_root'],'FSD50K','all_combined.csv'),
                   'background': os.path.join(config['source_root'],'FSD50K','all_combined.csv')}

    mix_lufs = {'music'     :config['lufs_music'],
                'sfx'       :config['lufs_sfx'],
                'speech'    :config['lufs_speech'],
                'background':config['lufs_background'],
                'master'    :config['lufs_master'],
                'noise'     :config['lufs_noise'],
                'ranges'    :{'class':config['lufs_range_class'],
                              'clip':config['lufs_range_clip']}}

    DatasetCompiler(seq_dur=config['seq_dur'],
                    num_files=config['num_files'],
                    input_dirs=files_dir_dict,
                    ground_truth_dirs=gt_dir_dict,
                    output_dir=output_dir,
                    alt_dir=alt_dir,
                    partition=config['partition'],
                    mix_lufs=mix_lufs,
                    sample_rate=config['sr']
                    )()