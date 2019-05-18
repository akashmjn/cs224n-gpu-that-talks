#!/usr/local/bin/python3

"""
Functions for data I/O 

Some functions referenced from: https://www.github.com/kyubyong/dc_tts 
Author: Akash Mahajan: akashmjn@alumni.stanford.edu 
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf
import sentencepiece as spm
import tf_sentencepiece as tfs
import codecs, re, os, unicodedata
from .dsp_utils import *


def tokenize_tf(transcription,model_proto,add_bos_eos=True,nbest_size=64,alpha=0.1):
    # By default adds <sos> and <eos> tokens
    tokens, seq_len = tfs.encode(transcription,model_proto=model_proto,
                            nbest_size=nbest_size,alpha=alpha,
                            add_eos=add_bos_eos,add_bos=add_bos_eos)
    return tokens, seq_len

def tokenize():
    #TODO: Implement with tf_sentencepiece https://colab.research.google.com/drive/1rQ0tgXmHv02sMO6VdTO0yYaTvc1Yv1yP
    pass

def process_csv_file(csv_path,params,text_col_num=1): #TODO:SP
    # Process text file containing file,labels
    # Returns file_paths, text_lengths, indexes (np.array of ints)

    fpaths, text_lengths, transcriptions = [], [], []
    lines = codecs.open(csv_path, 'r', 'utf-8').readlines()

    print('Processing csv file..')
    for line in lines:
        columns = line.strip().split(params.transcript_csv_sep)
        fname, text = columns[0], columns[text_col_num]
        fpath = os.path.join(params.data_dir,'wavs',fname + ".wav")
        fpaths.append(fpath)
        text_lengths.append(len(text))
        transcriptions.append(text)
    return fpaths, text_lengths, transcriptions   

def load_tokens_from_text(params,lines=None):
    sp = spm.SentencePieceProcessor()
    sp.load(params.spm_model)
    if lines is None:
        lines = codecs.open(params.test_data, 'r', 'utf-8').read().splitlines()[1:]
        lines = list(map(lambda x: x.split('. ',1)[-1],lines))
    
    print("Loading test sentences: {}".format(lines))
    # Encode each line, add control tokens
    tokens_list = []
    for line in lines:
        tokens_list.append([sp.bos_id(),*sp.encode_as_ids(line),sp.eos_id()])
    # Pack into numpy array
    lengths = [len(t) for t in tokens_list]
    tokens = np.zeros((len(tokens_list),max(lengths)),np.int32)
    for i,t in enumerate(tokens_list): tokens[i,:lengths[i]] = t
    return tokens, lengths


class TFRecordDataloader(object):
    def __init__(self,params,tfrecord_path,logger):
        self.params = params
        self.tfrecord_path = tfrecord_path
        self.logger = logger
        # load sentencepiece model
        self.spm_proto = tf.gfile.GFile(params.spm_model, 'rb').read()
        self.spm_model = spm.SentencePieceProcessor()
        self.spm_model.Load(params.spm_model)
        self.vocab_size = self.spm_model.GetPieceSize()
        # find total number of batches
        self.num_batch_train = sum(1 for line in open(params.transcript_csv_path_train))//params.batch_size
        self.num_batch_val = sum(1 for line in open(params.transcript_csv_path_val))//params.batch_size

        #TODO: Figure out curriculum learning implementation
        # one way: dataset = dataset.filter(lambda elem: tf.shape(elem)[0] < max_length)
        self.dataset = tf.data.TFRecordDataset([self.tfrecord_path])\
                    .map(self.parse_tfrecord,self.params.num_threads)\
                    .shuffle(10000)\
                    .apply(self._get_batching_func(self.params.dynamic_batch))\
                    .prefetch(1) # pads with 0s: works for mels, mags, and indexes since vocab[0] is P

        self.iterator = self.dataset.make_initializable_iterator()
        self.iterator_init_op = self.iterator.initializer
        self.logger.info('Created dataset and iterators..')
    
    def _get_batching_func(self,dynamic=False):
        """
        Makes params.num_buckets uniformly upto params.max_T length
        dynamic: batch_sizes adjusted with ref to middle bucket to keep no. of frames same 
                else, same batch size for all buckets
        """
        padded_shapes = (
                tf.TensorShape([None]),
                tf.TensorShape([None,self.params.F]),
                tf.TensorShape([None,self.params.Fo]),
                tf.TensorShape([None,self.params.F])
            )
        bin_width = round(self.params.max_T//self.params.num_buckets)
        bucket_sizes = [ (i+1)*bin_width for i in range(self.params.num_buckets) ] 
        mid_bucket_size = bucket_sizes[len(bucket_sizes)//2]

        if dynamic:
            batch_sizes = [ round(mid_bucket_size/s*self.params.batch_size) for s in [*bucket_sizes,bucket_sizes[-1]] ]
        else:
            batch_sizes = [self.params.batch_size]*(len(bucket_sizes)+1)

        bucketing_func = tf.data.experimental.bucket_by_sequence_length( # Use no. of mel frames to bucket
            lambda t,ml,mg,mlm: tf.shape(ml)[0],bucket_sizes,batch_sizes,padded_shapes=padded_shapes
        )
        self.logger.info("Created bucketed batches. Bucket lengths: {}, Batch sizes: {}".format(bucket_sizes,batch_sizes))
        return bucketing_func

    def parse_tfrecord(self,serialized_inp):
        # TODO: add support for randomly sampling mag patches for SSRN

        feature_struct = {
            'fname': tf.FixedLenFeature([],tf.string),
            'transcription': tf.FixedLenFeature([],tf.string),
            'mel': tf.FixedLenFeature([],tf.string),
            'mag': tf.FixedLenFeature([],tf.string),
            'input-len': tf.FixedLenFeature([],tf.int64),
            'mel-shape': tf.FixedLenFeature([2],tf.int64),
            'mag-shape': tf.FixedLenFeature([2],tf.int64)
        }    

        features = tf.parse_single_example(serialized_inp,features=feature_struct) 

        transcription = tf.reshape(features['transcription'],[1,])  # Shape: (1,) - rank 1 required by sentencepiece
        tokens, seq_len = tokenize_tf(transcription,self.spm_proto,add_bos_eos=True,
                                        nbest_size=self.params.spm_nbest,alpha=self.params.spm_alpha) # Shape: (1,N)
        tokens = tf.squeeze(tokens)                                 #  Shape: (N,)
        mel = tf.reshape(tf.decode_raw(features['mel'],tf.float32),features['mel-shape'])
        mag = tf.reshape(tf.decode_raw(features['mag'],tf.float32),features['mag-shape'])
        mel_mask_shape = tf.cast(features['mel-shape'],tf.int32)
        # pad some end silence to get model to learn to stop    
        # mel = tf.pad(mel,[[0,3],[0,0]]) # TODO: Take this from params
        # mel_mask_shape = mel_mask_shape + tf.constant([3,0],tf.int32) 
        mel_mask = tf.ones(mel_mask_shape,tf.float32)

        return (tokens,mel,mag,mel_mask) 

    def get_batch(self):

        tokens, mels, mags, mel_mask = self.iterator.get_next()
        tokens.set_shape((None,None))
        mels.set_shape((None,None,self.params.F))
        mags.set_shape((None,None,self.params.Fo))
        mel_mask.set_shape((None,None,self.params.F))
        batch = {'tokens':tokens,'mels':mels,'mags':mags,'mels_mask':mel_mask}
        self.logger.info('Getting batch of tensors of shape: {} {} {}, mask:{}'.format(
                tokens.shape, mels.shape, mags.shape, mel_mask.shape
            ))

        return batch 


## Older functions referenced from Kyubyong Park repo (deprecated) ##

def load_vocab(params):
    """
    Returns two dicts for lookup from char2idx and idx2char using params.vocab
    ref: https://www.github.com/kyubyong/dc_tts 

    Args:
        params (utils.Params): Object containing various hyperparams
    Returns:
        char2idx (dict): From char to int indexes in the vocab
        idx2char (dict): From indexes in the vocab to char
    """
    char2idx = {char: idx for idx, char in enumerate(params.vocab)}
    idx2char = {idx: char for idx, char in enumerate(params.vocab)}
    return char2idx, idx2char

def text_normalize(text,params,remove_accents=False,ensure_fullstop=True):
    """
    Normalizes an input string based on params.vocab 
    ref: https://www.github.com/kyubyong/dc_tts 
    
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

def load_data(params,mode="train",lines=None): #TODO:SP
    '''Loads data
    ref: https://www.github.com/kyubyong/dc_tts 
      Args:
          mode: "train" or "synthesize".
    '''

    if 'train' in mode or 'val' in mode:
        # toggle train/val datasets
        transcript_csv_path = params.transcript_csv_path_train if 'train' in mode else params.transcript_csv_path_val
        return process_csv_file(transcript_csv_path,params)

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

def get_batch(params,mode,logger):
    """
    Loads training data and put them in queues
    ref: https://www.github.com/kyubyong/dc_tts 
    """
    
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


