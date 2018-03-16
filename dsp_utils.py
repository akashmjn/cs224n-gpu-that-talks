#!/usr/local/bin/python3

"""
DSP functions for generating / inverting spectrograms

Code modifed from: https://www.github.com/kyubyong/dc_tts
"""

from __future__ import print_function, division

import numpy as np
import librosa
import os, copy
from scipy import signal
from scipy.io import wavfile

import tensorflow as tf
import pdb


######## Audio to spectrogram, and inversion functions #######

def load_spectrograms(fpath,params,mode):
    """
    Read the wave file in `fpath` and extracts downsampled mel-scale, and 
    linear scale log magnitude spectrograms that are normalized. 
    Mel is downsampled in time to T/params.reduction_factor=4, where T is
    the length of the full spectrogram. 
    Based on mode - for 'train_ssrn' - it loads upto a fixed-size random sample from the file 
    """

    fname = os.path.basename(fpath)
    mel, mag = get_spectrograms(fpath,params,mode)
    t = mel.shape[0]

    # Marginal padding for reduction shape sync.
    num_paddings = params.reduction_factor - (t % params.reduction_factor) if t % params.reduction_factor != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")

    # Reduction
    mel = mel[::params.reduction_factor, :]
    return fname, mel, mag

def get_spectrograms(fpath,params,mode):
    '''Parse the wave file in `fpath` and
    Returns normalized log magnitude melspectrogram and linear scale spectrogram.

    Args:
      fpath: A string. The full path of a sound file.
      mode: if 'train_ssrn' does not process full audio file, only a random patch to speec up computation

    Returns:
      mel: A 2d array of shape (T, F) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=params.sampling_rate)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # If training SSRN, only use a smaller random patch of the data 
    # that produces params.ssrn_T number of frames
    if mode=='train_ssrn':
        n, sample_len = len(y), params.ssrn_T*(params.hop_length-1)
        if sample_len >= n: # in case of clips smaller than threshold
            y = np.concatenate([y,np.zeros(sample_len-n)]) # pad with zeros
        else:
            idx = np.random.choice(n-sample_len)
            y = y[idx:idx+sample_len] # random chunk of input signal

    # Preemphasis
    y = np.append(y[0], y[1:] - params.pre_emphasis * y[:-1]) 

    # stft
    linear = librosa.stft(y=y,
                          n_fft=params.n_fft,
                          hop_length=params.hop_length,
                          window='hann')

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(params.sampling_rate, params.n_fft, params.F)  # (F, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (F, t)

    # amplitude to decibel
    mel = _amp_to_db(mel,params)
    mag = _amp_to_db(mag,params)

    # normalize
    mel = _normalize(mel,params)
    mag = _normalize(mag,params)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, F)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag

def spectrogram2wav(mag,params):
    """ Generate wav file from scaled log-magnitude spectrogram (0-1)

    Args:
      mag (np.ndarray): A numpy array of (T, 1+n_fft//2) with values from 0-1
      mode (str): Indicates whether linear magnitude or in log magnitude

    Returns:
      wav: A 1-D numpy array.
    """

    if np.max(mag)>1.0 or np.min(mag)<0.0:
        raise Warning('Input to spectrogram2wav is not normalized in (0,1]')

    # transpose and ensuring mag
    mag = np.clip(mag.T,0,1)

    # de-normalize, convert back to linear magnitudes
    mag = _db_to_amp(_denormalize(mag,params)) 

    # apply sharpening factor
    mag = mag ** params.sharpening_factor

    # wav reconstruction
    wav = griffin_lim(mag,params)

    # undo-preemphasis
    wav = signal.lfilter([1], [1, -params.pre_emphasis], wav)

    # trim out silent portions of signal
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

def griffin_lim(spectrogram,params):
    """
    Applies Griffin-Lim's spectrogram inversion 
    """
    X_best = copy.deepcopy(spectrogram)
    for i in range(params.n_iter):
        X_t = invert_spectrogram(X_best,params)
        est = librosa.stft(X_t,n_fft=params.n_fft,hop_length=params.hop_length,window='hann')
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best,params)
    y = np.real(X_t)

    return y

def invert_spectrogram(spectrogram,params):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return librosa.istft(spectrogram,hop_length=params.hop_length, window="hann")


#### Misc helper functions ####

"""
Referenced from: https://github.com/r9y9/deepvoice3_pytorch/blob/master/audio.py
"""
def _amp_to_db(x,params):
    min_level = np.exp(params.min_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def _normalize(S,params):
    S = S - params.min_db  # S in [min_db, peak_value], converted to [0, peak_value-min_db]
    S = S/np.max(S)        # scaled [0,1]
    return np.clip(S, 0, 1)

def _denormalize(S,params):
    S = S*(params.ref_db-params.min_db) # full dynamic range [0, peak_value-min_db]
    S = S + params.min_db               # back to range [min_db, peak_value]
    return S

def save_wav(wav, path, sr):
    """Writes out an arbitrary float array wav to file 
    """
    # wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # wavfile.write(path, sr, wav.astype(np.int16)) # 16-bit PCM
    wavfile.write(path, sr, wav) 

if __name__ == '__main__':

    from utils import Params
    params = Params('./runs/default/params.json')
    fpath = '../data/lj-speech/LJSpeech-1.0/wavs/LJ001-0006.wav/'
    dsp_utils.get_spectrograms(fpath,params)
