#!/usr/local/bin/python3

"""
Functions for data I/O 

Code referenced from: https://www.github.com/kyubyong/dc_tts
Author: kyubyong park. kbpark.linguist@gmail.com. 
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf
import codecs, re, os, unicodedata
from .dsp_utils import *

import pdb

def load_vocab(params):
    """
    Returns two dicts for lookup from char2idx and idx2char using params.vocab

    Args:
        params (utils.Params): Object containing various hyperparams
    Returns:
        char2idx (dict): From char to int indexes in the vocab
        idx2char (dict): From indexes in the vocab to char
    """
    char2idx = {char: idx for idx, char in enumerate(params.vocab)}
    idx2char = {idx: char for idx, char in enumerate(params.vocab)}
    return char2idx, idx2char

def text_normalize(text,params):
    """
    Normalizes an input string based on params.vocab 
    
    Args:
        text (str): Input text 
        params (utils.Params): Object containing various hyperparams
    Returns:
        text (str): Normalized text based on params.vocab
    """
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn') # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(params.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def load_data(params,mode="train",lines=None):
    '''Loads data
      Args:
          mode: "train" or "synthesize".
    '''
    # Load vocabulary
    char2idx, idx2char = load_vocab(params)

    if 'train' in mode or 'val' in mode:
        # toggle train/val datasets
        transcript_csv_path = params.transcript_csv_path_train if 'train' in mode else params.transcript_csv_path_val
        # Parse
        fpaths, text_lengths, texts = [], [], []
        lines = codecs.open(transcript_csv_path, 'r', 'utf-8').readlines()
        for line in lines:
            fname, _, text = line.strip().split(params.transcript_csv_sep)[:3]
            fpath = os.path.join(params.wavs_dir_path,fname + ".wav")
            fpaths.append(fpath)
            text = text_normalize(text,params) + params.end_token  # E: EOS
            text = [char2idx[char] for char in text]
            text_lengths.append(len(text))
            texts.append(np.array(text, np.int32).tostring())
        return fpaths, text_lengths, texts

    elif mode=='synthesize': # synthesize on unseen test text.
        # Parse from a file
        lines = codecs.open(params.test_data, 'r', 'utf-8').readlines()[1:]
        sents = [text_normalize(line.split(" ", 1)[-1],params).strip() + params.end_token for line in lines] # text normalization, E: EOS

    elif lines is not None:
        sents = [text_normalize(line,params) + params.end_token for line in lines]       

    print("Loading test sentences: {}".format(sents))
    max_len = max([len(sent) for sent in sents])
    texts = np.zeros((len(sents), max_len), np.int32)
    for i, sent in enumerate(sents):
        texts[i, :len(sent)] = [char2idx[char] for char in sent]
    return texts

def get_batch(params,mode,logger):
    """Loads training data and put them in queues"""
    
    with tf.device('/cpu:0'):
        # Load data
        logger.info('Loading in filenames from load_data with mode: {}'.format(mode))
        fpaths, text_lengths, texts = load_data(params,mode) # list
        maxlen, minlen = max(text_lengths), min(text_lengths)

        # Calc total batch count
        num_batch = len(fpaths) // params.batch_size

        # Create Queues
        shuffle_batch = False if 'val' in mode else True
        fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=shuffle_batch)
        logger.info('Created input queues for data, total num_batch: {}'.format(num_batch))

        # Parse
        text = tf.decode_raw(text, tf.int32)  # (None,)

        if params.prepro:
            def _load_spectrograms(fpath):
                fname = os.path.basename(fpath)
                mel = "mels/{}".format(fname.replace("wav", "npy"))
                mag = "mags/{}".format(fname.replace("wav", "npy"))
                return fname, np.load(mel), np.load(mag)

            fname, mel, mag = tf.py_func(_load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])
        else:
            parse_func = lambda path: load_spectrograms(path,params,mode)
            fname, mel, mag = tf.py_func(parse_func, [fpath], [tf.string, tf.float32, tf.float32])  # (None, F)
            logger.info('Defined load_spectrograms op with mode: {}'.format(mode))

        # Add shape information
        fname.set_shape(())
        text.set_shape((None,))
        mel.set_shape((None, params.F))
        mag.set_shape((None, params.n_fft//2+1))

        # Batching
        bucket_sizes = [i for i in range(minlen + 1, maxlen - 1, (maxlen-minlen)//params.num_buckets)]
        _, (texts, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[text, mel, mag, fname],
                                            batch_size=params.batch_size,
                                            bucket_boundaries=bucket_sizes,
                                            num_threads=params.num_threads,
                                            capacity=params.batch_size*params.Qbatch,
                                            dynamic_pad=True)
        logger.info('Created {} bucketed queues from min/max len {}/{}, batch_size: {}, capacity: {}'.format(params.num_buckets,minlen,maxlen,
            params.batch_size,params.batch_size*params.Qbatch))

    return texts, mels, mags, fnames, num_batch

