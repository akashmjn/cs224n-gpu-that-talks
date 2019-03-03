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

def text_normalize(text,params,remove_accents=True,ensure_fullstop=True):
    """
    Normalizes an input string based on params.vocab 
    
    Args:
        text (str): Input text 
        params (utils.Params): Object containing various hyperparams
    Returns:
        text (str): Normalized text based on params.vocab
    """
    if remove_accents:
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                               if unicodedata.category(char) != 'Mn') # Strip accents
    else:
        text = unicodedata.normalize('NFD',text)

    text = text.lower()
    text = re.sub("[^{}]".format(params.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def process_csv_file(csv_path,params,mode='IndicTTSHindi'):
    # Process text file containing file,labels
    # Returns file_paths, text_lengths, indexes (np.array of ints)

    # Load vocabulary
    char2idx, idx2char = load_vocab(params)   
    fpaths, text_lengths, indexes = [], [], []
    lines = codecs.open(csv_path, 'r', 'utf-8').readlines()

    print('Processing csv file with mode: {}'.format(mode))
    for line in lines:
        if mode=='LJSpeech':
            assert('lj' in csv_path.lower())
            fname, _, text = line.strip().split(params.transcript_csv_sep)[:3]
            text = text_normalize(text,params) + params.end_token  # E: EOS
        elif mode=='IndicTTSHindi':
            assert('indic' in csv_path.lower())
            fname, text = line.strip().split(params.transcript_csv_sep)[:2]
            text = text_normalize(text,params,False) + params.end_token  # E: EOS           
        fpath = os.path.join(params.data_dir,'wavs',fname + ".wav")
        fpaths.append(fpath)
        text = [char2idx[char] for char in text]
        text_lengths.append(len(text))
        indexes.append(np.array(text, np.int32).tostring())
    return fpaths, text_lengths, indexes   


def load_data(params,mode="train",lines=None):
    '''Loads data
      Args:
          mode: "train" or "synthesize".
    '''

    if 'train' in mode or 'val' in mode:
        # toggle train/val datasets
        transcript_csv_path = params.transcript_csv_path_train if 'train' in mode else params.transcript_csv_path_val
        return process_csv_file(transcript_csv_path,params,'IndicTTSHindi')

    elif mode=='synthesize' or lines is not None: # inference mode on unseen test text
        # Load vocabulary
        char2idx, idx2char = load_vocab(params)   
        if lines is None: # Reading from text file
            lines = codecs.open(params.test_data, 'r', 'utf-8').readlines()[1:]
            if "hindi" in params.data_dir.lower():
                sents = [text_normalize(line.split(" ",1)[-1],params,False).strip() + params.end_token for line in lines]
            else:
                sents = [text_normalize(line.split(" ", 1)[-1],params).strip() + params.end_token for line in lines] # text normalization, E: EOS
        else:             # Demo mode - direct list of sentences
            sents = [text_normalize(line,params) + params.end_token for line in lines]       

        print("Loading test sentences: {}".format(sents))
        max_len = max([len(sent) for sent in sents])
        indexes = np.zeros((len(sents), max_len), np.int32)
        for i, sent in enumerate(sents):
            indexes[i, :len(sent)] = [char2idx[char] for char in sent]
        return indexes

def parse_tfrecord(serialized_inp):
    # TODO: add support for randomly sampling mag patches for SSRN

    reader = tf.TFRecordReader()

    feature_struct = {
        'fname': tf.FixedLenFeature([],tf.string),
        'indexes': tf.FixedLenFeature([],tf.string),
        'mel': tf.FixedLenFeature([],tf.string),
        'mag': tf.FixedLenFeature([],tf.string),
        'input-len': tf.FixedLenFeature([],tf.int64),
        'mel-shape': tf.FixedLenFeature([2],tf.int64),
        'mag-shape': tf.FixedLenFeature([2],tf.int64)
    }    

    features = tf.parse_single_example(serialized_inp,features=feature_struct) 

    indexes = tf.decode_raw(features['indexes'],tf.int32)
    mel = tf.reshape(tf.decode_raw(features['mel'],tf.float32),features['mel-shape'])
    mag = tf.reshape(tf.decode_raw(features['mag'],tf.float32),features['mag-shape'])
    # pad some end silence to get model to learn to stop    
    mel = tf.pad(mel,[[0,3],[0,0]]) 
    mel_mask_shape = tf.cast(features['mel-shape'],tf.int32)
    mel_mask_shape = mel_mask_shape + tf.constant([3,0],tf.int32) 
    mel_mask = tf.ones(mel_mask_shape,tf.float32)

    return (indexes,mel,mag,mel_mask) 

def get_batch(params,mode,logger):
    """Loads training data and put them in queues"""
    
    with tf.device('/cpu:0'):
        # Load data
        logger.info('Loading in filenames from load_data with mode: {}'.format(mode))
        fpaths, text_lengths, indexes = load_data(params,mode) # list
        maxlen, minlen = max(text_lengths), min(text_lengths)

        # Calc total batch count
        num_batch = len(fpaths) // params.batch_size

        # Create Queues
        shuffle_batch = False if 'val' in mode else True
        fpath, text_length, index = tf.train.slice_input_producer([fpaths, text_lengths, indexes], shuffle=shuffle_batch)
        logger.info('Created input queues for data, total num_batch: {}'.format(num_batch))

        # Parse
        index = tf.decode_raw(index, tf.int32)  # (None,)

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
        index.set_shape((None,))
        mel.set_shape((None, params.F))
        mag.set_shape((None, params.n_fft//2+1))

        # Batching
        bucket_sizes = [i for i in range(minlen + 1, maxlen - 1, (maxlen-minlen)//params.num_buckets)]
        _, (indexes, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[index, mel, mag, fname],
                                            batch_size=params.batch_size,
                                            bucket_boundaries=bucket_sizes,
                                            num_threads=params.num_threads,
                                            capacity=params.batch_size*params.Qbatch,
                                            dynamic_pad=True)
        logger.info('Created {} bucketed queues from min/max len {}/{}, batch_size: {}, capacity: {}'.format(params.num_buckets,minlen,maxlen,
            params.batch_size,params.batch_size*params.Qbatch))

    return indexes, mels, mags, fnames, num_batch

def get_batch_prepro(tfrecord_path,params,logger):

    # find total number of batches
    num_batch_train = sum(1 for line in open(params.transcript_csv_path_train))//params.batch_size
    num_batch_val = sum(1 for line in open(params.transcript_csv_path_val))//params.batch_size

    padded_shapes = (
            tf.TensorShape([None]),
            tf.TensorShape([None,params.F]),
            tf.TensorShape([None,params.Fo]),
            tf.TensorShape([None,params.F])
        )

    dataset = tf.data.TFRecordDataset([tfrecord_path])\
                .map(parse_tfrecord,params.num_threads)\
                .padded_batch(params.batch_size,padded_shapes)\
                .prefetch(1) # pads with 0s: works for mels, mags, and indexes since vocab[0] is P

    iterator = dataset.make_initializable_iterator()
    indexes, mels, mags, mel_mask = iterator.get_next()

    # indexes.set_shape((None,None))
    mels.set_shape((None,None,params.F))
    mags.set_shape((None,None,params.Fo))
    mel_mask.set_shape((None,None,params.F))
    logger.info('Created iterators over tensors of shape: {} {} {}, mask:{}'.format(
            indexes.shape, mels.shape, mags.shape, mel_mask.shape
        ))
    # TODO: add a mask
    batch = {'indexes':indexes,'mels':mels,'mags':mags,'mels_mask':mel_mask}
    iterator_init_op = iterator.initializer

    return batch, iterator_init_op, num_batch_train, num_batch_val

