
import soundfile as sf
import numpy as np
import resampy
from math import ceil
import librosa

from scipy.signal import stft
import pyloudnorm as pyln
import math
import warnings
# We silence this warning as we peak normalize the samples before bouncing them
warnings.filterwarnings("ignore", message="Possible clipped samples in output.")


def trim_relative_silence_from_audio(audio, sr, frame_duration=0.04,
                                     hop_duration=0.01):

    assert 0 < hop_duration <= frame_duration
    frame_length = int(frame_duration * sr)
    hop_length = int(hop_duration * sr)

    _, _, S = stft(audio, nfft=frame_length, noverlap=frame_length-hop_length, 
              padded=True, nperseg=frame_length, boundary='constant')
    rms = librosa.feature.rms(S=S, frame_length=frame_length,
                              hop_length=hop_length, pad_mode='constant').flatten()
    threshold = 0.01 * rms.max()
    active_flag = rms >= threshold
    active_idxs = active_flag.nonzero()[0]
    start_idx = max(int(max(active_idxs[0] - 1, 0) * hop_duration * sr), 0)
    end_idx = min(int(ceil(min(active_idxs[-1] + 1,
                               rms.shape[0]) * hop_duration * sr)),
                  audio.shape[0])

    
    return start_idx, end_idx

def lufs_norm(data, sr, norm=-6):
    block_size = 0.4 if len(data) / sr >= 0.4 else len(data) / sr
    # measure the loudness first 
    meter = pyln.Meter(rate=sr, block_size=block_size)
    loudness = meter.integrated_loudness(data)
    
    assert not math.isinf(loudness)

    norm_data = pyln.normalize.loudness(data, loudness, norm)
    n, d = np.sum(np.array(norm_data)), np.sum(np.array(data))
    gain = n/d if d else 0.0

    return norm_data, gain

def get_lufs(data, sr):
    block_size = 0.4 if len(data) / sr >= 0.4 else len(data) / sr
    # measure the loudness first 
    meter = pyln.Meter(rate=sr, block_size=block_size) # create BS.1770 meter
    loudness = meter.integrated_loudness(data)

    return loudness

def peak_norm(data, mx):
    eps = 1e-10
    max_sample = np.max(np.abs(data))
    scale_factor = mx / (max_sample + eps)

    return data * scale_factor

def gain_to_db(g):
    return 20 * np.log10(g)

def db_to_gain(db):
    return 10 ** (db / 20.)
    
def gain_from_combined_db_levels(dbs):
    return np.prod([10 ** (db / 20.) for db in dbs])

def validate_audio(d):
    assert np.isnan(d).any() == False, "Nan value found in mixture"
    assert np.isneginf(d).any() == False, "Neg. Inf value found in mixture"
    assert np.isposinf(d).any() == False, "Pos. Inf value found in mixture"
    assert np.isinf(d).any() == False, "Inf value found in mixture"