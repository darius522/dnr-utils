from matplotlib import pyplot as plt
from glob import glob
import os
import argparse
import shutil
import tqdm
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
import librosa
import pandas as pd
import shutil
from distutils.dir_util import copy_tree

from resampy import resample
from scipy.io import wavfile
from pydub import AudioSegment
import soundfile as sf

from audio_utils import (trim_relative_silence_from_audio,
                        lufs_norm)

# Define whether each fsd50k classes fall into background (2), foreground (1), or not usable (0)
fsd50k_classes = [1,0,0,2,2,1,1,1,0,0,1,1,1,1,2,1,2,1,1,0,0,
                  0,1,2,1,1,2,2,1,2,2,1,1,0,0,1,1,0,0,1,1,1,
                  1,0,0,0,1,1,0,1,1,2,1,1,0,1,1,0,1,1,2,2,1,
                  1,1,1,1,0,0,0,2,1,1,1,0,0,1,1,2,1,2,2,1,2,
                  0,0,1,0,0,1,0,2,1,1,1,1,0,0,0,2,0,0,2,2,0,
                  1,1,0,1,2,0,0,0,0,2,2,1,2,2,2,0,0,2,0,1,0,
                  0,0,1,1,1,1,2,2,2,2,1,1,0,0,0,1,1,1,1,0,0,
                  1,0,0,0,2,2,1,1,1,0,0,0,0,1,1,2,0,2,0,0,1,
                  1,1,1,2,2,1,1,1,1,2,2,2,2,0,1,1,2,1,2,1,2,
                  2,0,1,2,2,0,0,1,1,0,1]

def prepare_fsd50k(src_dir, dst_dir):

    '''
    Write the FSD50K dataset from src to dst directory
    Format the output as instructed, which, in this case, solely involves silence trimming

    Args:
        src_dir: root path of FSD50K 
        dst_dir: destination of formatted FSD50K
    '''
    out_sr = 44100

    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok= True)

    # First, let's create the ground truth-combined .csv
    df_dev = pd.read_csv(os.path.join(src_dir,'FSD50K.ground_truth/dev.csv'))
    df_eval = pd.read_csv(os.path.join(src_dir,'FSD50K.ground_truth/eval.csv'))
    df_eval['split'] = 'test'
    pd.concat([df_dev,df_eval]).to_csv(os.path.join(dst_dir,'all_combined.csv'))

    paths = glob(src_dir+'/**/*.wav', recursive=True)
    random.shuffle(paths)

    for path in tqdm(paths):
        data, _ = librosa.load(path, sr=out_sr)
        trim_s, trim_e = trim_relative_silence_from_audio(data, out_sr)
        out_name = os.path.basename(path)
        out_dir = Path(os.path.join(dst_dir,path.split('/')[-2]))
        out_dir.mkdir(parents=True, exist_ok= True)
        if not os.path.isfile(out_name):
            try:     
                sf.write(os.path.join(out_dir,out_name), data[trim_s:trim_e], out_sr, 'PCM_16')
            except Exception as e: 
                print(e)

def refactor_fsd50k(ground_truth_path, root_path):
    '''
    Given a FSD50K vocabulary list (by default found in FSD50K.ground_truth)
    Restructure the dataset folders according to the classes defined by "fsd50k_classes"

    Args:
        ground_truth_path: Dataset ground truth folder directory (by default found in FSD50K.ground_truth)
        root_path: root_path of the location where FSD50K will be formatted
    '''

    vocab_csv  = pd.read_csv(os.path.join(ground_truth_path,'vocabulary.csv'),names=["label", "group"])
    vocab_csv['class'] = fsd50k_classes
    for s in ['dev','eval']:
        class_paths = {1:Path(os.path.join(root_path,'foreground')),
                       2:Path(os.path.join(root_path,'background'))}
        labels_csv = pd.read_csv(os.path.join(ground_truth_path,'{}.csv'.format(s)))
        paths = glob(os.path.join(root_path,'FSD50K.{}_audio').format(s)+'/*.wav')

        for p in tqdm(paths):
            fname = os.path.basename(p).split('.')[0]
            ls = labels_csv.loc[labels_csv['fname'] == int(fname)]
            ls = ls['labels'].values[0].split(',')
            cs = []
            for l in ls:
                # check file class value
                cs.append(vocab_csv.loc[vocab_csv['label']==l]['class'].values[0])

            # move the file to the corresponding dir based on class recurrency
            rec_c = np.argmax(np.bincount(cs))
            if rec_c==0:
                os.remove(p)
                continue
            dst_dir = class_paths[rec_c]
 
            split = labels_csv.loc[labels_csv['fname']==int(fname)]['split'].values[0] if s=='dev' else s
            split_path = Path(os.path.join(dst_dir, split))
            split_path.mkdir(parents=True, exist_ok= True)
            shutil.move(p, split_path)

        os.rmdir(os.path.join(root_path,'FSD50K.{}_audio'.format(s)))
        copy_tree(ground_truth_path, root_path)

def prepare_fma(src_dir, dst_dir):

    '''
    Write the FMA dataset from src to dst directory
    Format the output as instructed. The original files are .mp3, and stereo. 
    Conversion to .wav-mono is thus needed

    Args:
        src_dir: root path of original FMA 
        dst_dir: destination of formatted FMA
    '''

    def get_split_from_fname(fname, df):
        '''Retrieve split from a given filename and metadata dataframe'''
        # Keep spit name consistent across datasets
        split_format = {'training':'train','validation':'val','test':'eval'}
        f_idx = int(fname.split('.')[0].lstrip("0"))
        return split_format[df.loc[f_idx]['set']]

    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok= True)

    # First we create the annotation csv for FMA
    create_gt_fma(os.path.join(src_dir,'fma_metadata'), dst_dir)

    out_sr = 44100
    paths = glob(src_dir+'/**/*.mp3')

    # Load metadata csv and filter out columns
    df_tracks = pd.read_csv(os.path.join(src_dir,'fma_metadata','tracks.csv'))
    df_tracks = df_tracks.loc[df_tracks['set.1'].isin(['small', 'medium'])]
    df_ids    = df_tracks.loc[2:][['Unnamed: 0','set']]
    df_ids['Unnamed: 0'] = pd.to_numeric(df_ids['Unnamed: 0'])
    df_ids    = df_ids.set_index('Unnamed: 0')

    for path in tqdm(paths):
        fname = os.path.basename(path).split('.')[0]+'.wav'
        split = get_split_from_fname(fname, df_ids)

        sub = path.split('/')[-2]
        full_dst = Path(os.path.join(dst_dir,split,sub))
        full_dst.mkdir(parents=True, exist_ok= True)

        out_name = os.path.join(full_dst, fname)
        if not os.path.isfile(out_name):
            try:                                                     
                sound = AudioSegment.from_mp3(path)
                # Mono, 44100, WAV-PCM
                sound = sound.set_frame_rate(out_sr)
                sound = sound.set_channels(1)
                data  = np.array(sound.get_array_of_samples())
                sf.write(out_name, data, out_sr, 'PCM_16')
            except Exception as e: 
                print(e)

def create_gt_fma(fma_meta_folder, dst_dir):
    '''
    The FMA metadata is huge, for our experiment we only care about the "all_genres" tag.
    This method aims at creating a new - stripped down - CSV for FMA dataset
    Args:
        fma_meta_folder: the fma metadata folder path. Expected to find "genres.csv" and "tracks.csv"
        dst_dir: directory where the new dataset resides (where the new .csv will be dumped)
    '''
    from ast import literal_eval

    output = pd.DataFrame(columns = ['track_id','genres'])

    labels = pd.read_csv(os.path.join(fma_meta_folder,'tracks.csv'))
    labels = labels.loc[labels['set.1'].isin(['small', 'medium'])]
    label_genres = labels[['Unnamed: 0','track.9']]

    vocab = pd.read_csv(os.path.join(fma_meta_folder,'genres.csv'))

    for i, track in tqdm(label_genres.iterrows()):
        song = track['Unnamed: 0']
        try:
            song_genres = list(literal_eval(track['track.9']))
            literals = []
            if isinstance(song_genres, list):
                for genre_id in song_genres:
                    literals.append(vocab.loc[vocab['genre_id'] == genre_id]['title'].values[0])
                output.loc[len(output.index)] = [song,','.join(literals)]
        except Exception as e:
            print(e)
    
    output.to_csv(os.path.join(dst_dir,'fma_genres_ground_truth.csv'))

def build_librispeech_hq(librivox_dir, librispeech_dir, dst_dir, validate=True):
    '''
    Build librispeech segment files from a given LibriSpeech dataset directory.
    LibriSpeech is by default available in 16kHz. In order to use it in 44.1kHz, we need
    to build it from "source" (LibriVox). LibriSpeech is still needed as it contains the source
    metadata needed in order to reconstruct it from LibriVox
    Args:
        - librivox_dir: path of librivox to build from
        - librispeech_dir: path of librispeech to build from
        - dst_dir: dest. path of where librispeech-hq will be built
    '''
    def segment_to_dict(path):
        '''
        take a .seg text file and turn it into a dict of filename:timestamps pairs
        '''
        d = {}
        with open(path) as f:
            for line in f:
                key = line.split()[0]
                val = line.split()[1:]
                d[key] = val
        
        return d
    
    def utterance_to_dict(path):
        '''
        take a .map text file and turn it into a dict of filename1:filename2 pairs
        '''
        d = {}
        with open(path) as f:
            lines = f.readlines()
            # first two lines are special cases
            d['sentences only'] = False if 'False' in lines[0] else True
            for line in lines[2:]:
                key = line.split()[0]
                val = line.split()[1:]
                d[key] = val
            
        return d

    sr = 44100
    # Keep split names consistent across datasets
    split_format = {'train-clean-100':'train','dev-clean':'val','test-clean':'eval'}
    paths = glob(librivox_dir+'/**/*.mp3', recursive=True)
    speaker_per_set = {'test-clean':  [x.split('/')[-2] for x in glob(os.path.join(librispeech_dir,'test-clean','*/'))],
                       'dev-clean':   [x.split('/')[-2] for x in glob(os.path.join(librispeech_dir,'dev-clean','*/'))],
                       'train-clean-100': [x.split('/')[-2] for x in glob(os.path.join(librispeech_dir,'train-clean-100','*/'))]}
    for path in tqdm(paths):
        # First open text of all segments from orig path
        parent = path.split('/')[:-1]
        chapter_id = parent[-1]
        speaker_id = parent[-2]
        audio_filename = path.split('/')[-1].split('.')[0]
        current_split = [key for key, item in speaker_per_set.items() if speaker_id in item]

        if len(current_split) == 0:
            continue

        chapter_dir = Path(os.path.join(dst_dir,split_format[current_split[0]],speaker_id, chapter_id))
        chapter_dir.mkdir(parents=True, exist_ok= True)

        trans_filename = "{}-{}.trans.txt".format(speaker_id,chapter_id)
        # We need to copy the .trans file from src to dst
        shutil.copy(os.path.join(librispeech_dir, current_split[0], speaker_id, chapter_id, trans_filename),
                    chapter_dir)

        # Get the utterance dict
        utt_filename = "utterance_map.txt"
        utt_path = os.path.join('/'.join(parent[:-1]),utt_filename)
        utt_dict = utterance_to_dict(utt_path)

        # Get the segments dictionary corresponding to the "sentences only" value
        seg_ext = '.sents.seg.txt' if utt_dict["sentences only"] else '.seg.txt'
        utt_dict.pop('sentences only', None)
        seg_filename = "{}-{}{}".format(speaker_id, chapter_id, seg_ext)
        seg_path = os.path.join('/'.join(parent), seg_filename)
        seg_dict = segment_to_dict(seg_path)

        try:  
            # Load sound                                                   
            sound = AudioSegment.from_mp3(path)
            sound = sound.set_channels(1)
            data  = np.array(sound.get_array_of_samples())

            if sound.frame_rate != sr:
                data = resample(data, sound.frame_rate, sr)

            # 'key' is the new filename, 'item' is the segment to get the timestamp from (in 'seg_dict')
            for key, items in utt_dict.items():
                item = items[0]
                #check if we're still checking on the right audio file
                if '-{}-'.format(audio_filename) in key:
                    new_audio_path = os.path.join(chapter_dir,key+'.wav')
                    start_spl, end_spl = int(float(seg_dict[item][0])*sr), int(float(seg_dict[item][1])*sr)
                    new_data = data[start_spl:end_spl]
                    sf.write(new_audio_path, new_data, sr, 'PCM_16', format='WAV')
        except Exception as e: 
            print(e)
    
    if validate:
        validate_librispeech_hq(librispeech_dir, dst_dir)

def validate_librispeech_hq(librispeech_dir, librispeech_dir_new, t_thres = 0.05):

    all_files = ['/'.join(x.split('/')[3:]) for x in glob(librispeech_dir+'/**/*.wav', recursive=True)]

    for file in tqdm(all_files):
        
        low_a, l_sr = sf.read(os.path.join(librispeech_dir,file))
        high_a, h_sr = sf.read(os.path.join(librispeech_dir_new,file))

        if np.abs((len(low_a)/l_sr)-(len(high_a)/h_sr)) > t_thres:
            print('Warning, found file',file,'with time discrepancy larger than threshold: ',len(low_a)/l_sr,len(high_a)/h_sr)

def validate_audio(dirs):
    '''
    Given a directory of files, validate audio and check for errors.
    '''
    broken = []
    for path in dirs:
        paths   = glob(path+'/**/*.wav', recursive=True)
        for path in tqdm(paths):
            try:
                _, d = wavfile.read(path)
                if (np.isnan(d).any() or np.isposinf(d).any() or np.isneginf(d).any() or np.isinf(d).any()):
                    print(np.isnan(d).any(),
                        np.isposinf(d).any(),
                        np.isneginf(d).any(),
                        np.isinf(d).any())
                    broken.append(path)
            except Exception as e: 
                broken.append(path)
                print(e, path)

    np.savez('./dnr_broken_files.npz',broken)

def get_filelen_stats(path_dic, title=''):
    '''
    Given a directory of files, compute and plots length distribution in seconds
    Not used explicitly in the script, but good to have just in case.
    '''
    from collections import OrderedDict
    import matplotlib

    font = {'family' : 'normal','size': 22}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    plt.cla()
    plt.clf()

    lengths = OrderedDict({'Librispeech':[],'FSD50k-Foreground':[],'FSD50k-Background':[],'FSD50k-Background':[],'FMA':[]})
    max_files = 1000
    for k, path in path_dic.items():
        print(k,path)
        paths = glob(path+'/**/*.wav', recursive=True)
        for path in tqdm(paths[:max_files]):
            try:
                data, sr = sf.read(path)
                lengths[k].append(data.shape[0] / sr)
            except Exception as e: 
                print(e)

    plt.figure(figsize=(12,6))
    plt.hist([x for x in lengths.values()], bins=50, label=list(lengths.keys()), alpha=0.5)
    plt.title('DNR per-class file length distribution')
    plt.legend(loc='best',bbox_to_anchor=(0.5, 0.5))
    plt.xlabel('File length (Sec)')
    plt.ylabel('Num. files')
    plt.xlim([0,32])
    plt.ylim([0,200])
    plt.tight_layout()
    plt.savefig('./img_'+title+'.pdf')

    return lengths

def main(args):

    # FSD50K Datasets: dev and eval set
    prepare_fsd50k(src_dir=args.fsd50k_path,
                   dst_dir=os.path.join(args.dest_dir,'FSD50K'))

    # Refactor the FSD50K dataset
    refactor_fsd50k(ground_truth_path=os.path.join(args.fsd50k_path,'FSD50K.ground_truth'),
                    root_path=os.path.join(args.dest_dir,'FSD50K'))
    
    # FMA Datasets: medium
    prepare_fma(src_dir=args.fma_path,
                dst_dir=os.path.join(args.dest_dir,'FMA'))

    # LibriSpeech: Build it off LibriVox
    build_librispeech_hq(librivox_dir=args.librivox_path, 
                         librispeech_dir=args.librispeech_path, 
                         dst_dir=os.path.join(args.dest_dir,'LibriSpeech'), 
                         validate=args.validate_audio)

    # Optionally, validate everything
    if args.validate_audio:
        validate_audio([os.path.join(args.dest_dir,'FMA'),
                        os.path.join(args.dest_dir,'FSD50K'),
                        os.path.join(args.dest_dir,'LibriSpeech')])

if __name__ == '__main__':
    '''
    Run this script in order to standardize all or parts of the datasets involved in the Divid and Remix (DNR) dataset.
    The script expects the three datasets to be found at a given directory (dest-dir). For more info on how to build and
    dataset-related specificities, please visit: <DNR website>

    Standardization involves:
        - Resampling based on the target sampling-rate
        - Summing all stereo files to mono (DNR is purely mono)
        - Rewriting files to a common format - WAV-PCM (some files are natively in .mp3 format (i.e. FMA))
        - Trimming the leading silences where necessary
        - Creating sets directory (val, eval, train)
        - FSD50K needs to be refactored in two classes: "background" and "foreground" prior to DNR building
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--fsd50k-path', 
                        type=str, 
                        required=True, 
                        help='The path of the original FSD50k dataset')
    parser.add_argument('--fma-path', 
                        type=str, 
                        required=True, 
                        help='The path of the original FMA dataset')
    parser.add_argument('--librispeech-path', 
                        type=str, 
                        required=True, 
                        help='The path of the original LibriSpeech dataset')
    parser.add_argument('--librivox-path', 
                        type=str, 
                        required=True, 
                        help='The path of the original LibriVox dataset')
    parser.add_argument('--dest-dir', 
                        type=str, 
                        required=True, 
                        help='Destination directory where the datasets reside')
    parser.add_argument('--validate-audio', 
                        type=bool, 
                        required=False,
                        default=True,
                        help='Whether audio validation should be done at the end of the process or not')

    main(parser.parse_args())