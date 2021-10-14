import soundfile as sf
from scipy.signal import resample_poly
from glob import glob
import os
from tqdm import tqdm
from pathlib import Path
from shutil import copyfile
import argparse

def downsample_dnr(src_path, dst_path):
    '''
    Given root path of the DNR-HQ dataset (44100 Hz), 
    downsample the dataset to 16kHz and write it to a given destination root path
    Args:
        - src_path: path of the original DNR-HQ dataset (in 44100Hz).
        - dst_path: path of of the root destination where the new downsampled dataset should be written
    '''

    wav_paths = glob(src_path+'/**/*.wav',recursive=True)
    target_sr = 16000
    orig_sr = 44100

    for p in tqdm(wav_paths, total=len(wav_paths)):
        a, a_sr = sf.read(p)
        assert a_sr == orig_sr, 'Incompatible SR found, DNR-HQ is expected to be at 44100Hz'

        a_r = resample_poly(a, target_sr//100, orig_sr//100)

        parent = os.path.dirname(p).split('/')
        fname = os.path.basename(p)

        track_id = parent[-1]
        split = parent[-2]
        new_p = Path(os.path.join(dst_path, split, track_id))
        new_p.mkdir(parents=True,exist_ok=True)

        # Write the new soundfile
        sf.write(os.path.join(new_p,fname), a_r, target_sr)

        # Also write the annotations csv file
        src_csv_path = os.path.join(os.path.dirname(p), 'annots.csv')
        dst_csv_path = os.path.join(new_p, 'annots.csv')
        if not os.path.isfile(dst_csv_path):
            copyfile(src_csv_path, dst_csv_path)

def main(args):
    downsample_dnr(src_path=args.input_dir, dst_path=args.output_dir)

if __name__ == '__main__':
    '''
    Run this script in order to resample the DNR-HQ dataset to 16kHz. 
    Prior to running the script you should have acquired DNR either by downloading it from Zenodo or building
    it with `compile_dataset.py`.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', 
                        type=str, 
                        required=True, 
                        help='The path of the original DNR-HQ dataset (in 44100Hz)')
    parser.add_argument('--output-dir', 
                        type=str, 
                        required=True, 
                        help='The path where the new downsampled DNR dataset is expected to be built')

    main(parser.parse_args())