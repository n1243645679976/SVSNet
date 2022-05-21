import os
import torch
import librosa
import threading
import numpy as np
#import pyworld as pw
#import pysptk
import scipy
#import python_speech_features
import sys
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm
from os.path import join
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--database-dir')
parser.add_argument('--output-dir')
args = parser.parse_args()
nj = 20


base_dir = args.database_dir
output_dir = args.output_dir
filename_extension = 'wav'
feature_to_extract = 'wav'
list_file = join(output_dir, 'list.txt')
SAMPLING_RATE = 16000

# for vad
vad_threshold = 30

# for mcc
MCC_DIM = 34

# for mels
N_MELS = 80

# for stft
HOP_LENGTH=64
WIN_LENGTH=127
#stft_model = STFT(256, 256, 256)

# for en
EPSILON = 1e-10

# for wld
F0_CEIL = 500
F0_FLOOR = 50
FFT_SIZE = 127 # for wld and stft

# rolling window
SEG_LENGTH = 1600  # in ms
SEG_STEP_LENGTH = 120  # in ms

def get_mcc_alpha(fs):
    """
    The parameter 'alpha' used for extracting mcc changes with the audio sampling rate
    https://bitbucket.org/happyalu/mcep_alpha_calc/
    For simply:
    FS      :   ALPHA
    8000    :   0.312
    11025   :   0.357
    16000   :   0.410
    22050   :   0.455
    32000   :   0.504
    44100   :   0.544
    48000   :   0.554
    """
    if fs == 8000:
        return 0.312
    elif fs == 11025:
        return 0.357
    elif fs == 16000:
        return 0.410
    elif fs == 22050:
        return 0.455
    elif fs == 32000:
        return 0.504
    elif fs == 44100:
        return 0.544
    elif fs == 48000:
        return 0.554
    else:
        raise ValueError("input fs value is illegal,\nshould be one of 8k, 16k, 22.05k, 32k, 44.1k, 48k")

def rolling_window(a, window, frame_step, window_step):
    seg_num = (a.shape[0] - 1 - frame_step * (window - 1)) // window_step + 1
    shape = (window, seg_num, a.shape[1])
    strides = (a.strides[1] * a.shape[1], a.strides[1] * window_step * a.shape[1], a.strides[1])
    return as_strided(a, shape=shape, strides=strides)
    

def wav2pw(x, fs=22050, fft_size=FFT_SIZE):
    linear = librosa.stft(y=x, n_fft=fft_size,
                          hop_length=HOP_LENGTH,
                          win_length=WIN_LENGTH,
                          window=scipy.signal.hamming)
    linear = linear.T
    mag = np.abs(linear)
    #FT
    print(mag.shape)
    
    return {    
        'mag': mag,
    }

def extract_one_file(directory, file, need_return=False):
    x, fs = librosa.load(join(directory, file), sr=SAMPLING_RATE,
                        mono=True, dtype=np.float64)
#    _f0, t = pw.dio(x, fs)    # raw pitch extractor
#    f0 = pw.stonemask(x, _f0, t, fs)
#    sp = pw.cheaptrick(x, f0, t, fs)
#    ap = pw.d4c(x, f0, t, fs)
    diction = {}
    if feature_to_extract == 'wav.vad':
        inds = librosa.effects.split(x, vad_threshold)
        x_ = np.array([])
        for ind in inds:
            x_ = np.hstack([x_, x[ind[0]:ind[1]]])
        diction['wav.vad'] = torch.tensor(x_, dtype=torch.float)
    elif feature_to_extract == 'wav':
        diction['wav'] = torch.tensor(x, dtype=torch.float)
    elif feature_to_extract == 'mag':
        features = wav2pw(x, fft_size=FFT_SIZE)
        diction['mag'] = torch.tensor(features['mag'], dtype=torch.float)
    elif feature_to_extract == 'lfbank':
        features = librosa.feature.melspectrogram(x, 16000, n_fft=1024, hop_length=256, n_mels=80, fmin=80, fmax=7600).transpose()
        features = np.maximum(features, 1e-10)
        features = np.log10(features)
        diction['lfbank'] = features
    else:
        print('{} not implemented.\n'.format(feature_to_extract))

    if not need_return:
        torch.save(diction, join(FEAT_DIR, name + '.pt'))
        del diction
    else:
        return diction
"""    
def extract_under_dir(directory):
    wavfiles = [i for i in os.listdir(directory) if i.endswith('.wav')]
    for i in tqdm(range(len(wavfiles))):
        extract_one_file(directory, wavfiles[i])


def extract_and_save_to_pt():
    if not os.path.isdir(join(FEAT_DIR, 'train')):
        os.makedirs(join(FEAT_DIR, 'train'))
    if not os.path.isdir(join(FEAT_DIR, 'test')):
        os.makedirs(join(FEAT_DIR, 'test'))
        
    for files in os.listdir(join(FEAT_DIR, 'train')):
        if not os.path.isdir(join(FEAT_DIR, 'train', files)):
            os.remove(join(FEAT_DIR, 'train', files))
    for files in os.listdir(join(FEAT_DIR, 'test')):
        if not os.path.isdir(join(FEAT_DIR, 'test', files)):
            os.remove(join(FEAT_DIR, 'test', files))
    extract_under_dir(WAV_DIR)
"""

def do_things(things):
    for thing in things:
        if thing.split('.')[-1] == filename_extension:
            name = thing.split('.')[0].split('/')[-1] + '.' + feature_to_extract + '.pt'
            a = extract_one_file(base_dir, thing, need_return=True)
            torch.save(a, join(output_dir, name))
            print(join(output_dir, name))
    
if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(list_file, 'w+') as f:
        os.makedirs(output_dir, exist_ok=True)
        things = [thing for thing in librosa.util.find_files(base_dir)]
        threads = []
        for i in range(nj):
            threads.append(threading.Thread(target=do_things, args=(things[i::nj],)))
            threads[i].start()
        for i in range(nj):
            threads[i].join()
