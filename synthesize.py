#!/usr/local/bin/python3

import argparse
import logging
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import pdb

import tensorflow as tf
from tqdm import tqdm
from graph import ModelGraph
from utils import Params
from data_load import load_vocab, text_normalize, load_data
from dsp_utils import spectrogram2wav, save_wav
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('-m1', help="Checkpoint directory for Text2Mel")
parser.add_argument('-m2', help="Checkpoint directory for SSRN")
parser.add_argument('--test_data', help="Data file containing sentences to synthesize")
parser.add_argument('--n_iter',type=int,default=50, help="Number of Griffin lim iterations to run for inversion")
parser.add_argument('--sample_dir', default="../samples",
                    help="Directory to save generated samples.")
args = parser.parse_args()

# Initialize graph, path to model checkpoints
params1 = Params(os.path.join(args.m1,'params.json'))
params2 = Params(os.path.join(args.m2,'params.json'))
params = params1
params.dict['test_data'] = args.test_data # setting this based on what passed in
params.dict['n_iter'] = args.n_iter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use single GPUs available
g = ModelGraph(params,'synthesize')

# text inputs 
input_arr = load_data(params,'synthesize')

# shape (1, len)
output_mel = np.zeros((input_arr.shape[0],params.max_T,params.F)) 
output_mag = np.zeros((input_arr.shape[0],params.max_T,params.Fo))

def invert_mag(output_mag):
    print('Generating full audio for batch of samples')
    wav = spectrogram2wav(output_mag,params) # adding some silence before
    fdir = os.path.join(args.sample_dir,params.model_name)
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    fname = os.path.join(fdir,'sample_{}'.format(i))
    save_wav(wav,fname+'.wav',params.sampling_rate)

# TODO: wrap this up in a function 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Load saved models
    text2mel_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'TextEnc|AudioEnc|AudioDec|InputEmbed')
    ssrn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN')

    saver1 = tf.train.Saver(var_list=text2mel_vars)
    saver1.restore(sess, tf.train.latest_checkpoint(args.m1))
    print("Text2Mel Restored!")
    saver2 = tf.train.Saver(var_list=ssrn_vars)
    saver2.restore(sess, tf.train.latest_checkpoint(args.m2))
    print("SSRN Restored!")   

    n_samples = output_mag.shape[0]
    # Generate all the mel frames
    # TODO: Implement forced monotonic attention 
    for i in range(1,params.max_T):
        print('Mel frame {}/{}'.format(i+1,params.max_T),end='\r')
        prev_slice = output_mel[:,:i,:]
        model_out = sess.run(g.Yhat,{g.S:prev_slice,g.transcripts:input_arr})
        output_mel[:,i,:] = model_out[:,-1,:]

    # Convert to magnitude spectrograms
    output_mag, attn_out = sess.run([g.Zhat,g.A],{g.S:output_mel,g.transcripts:input_arr})
    mags_list = [output_mag[i] for i in range(n_samples)]
    with Pool(12) as p:
        p.map(invert_mag,mags_list)
    for i in range(n_samples):
        print('Saving plots for sample: {}/{}'.format(i+1,n_samples))
        fdir = os.path.join(args.sample_dir,params.model_name)
        fname = os.path.join(fdir,'sample_{}'.format(i))
        plt.imsave(fname+'_mel.png',output_mel[i].T,cmap='gray')
        plt.imsave(fname+'_mag.png',output_mag[i].T,cmap='gray')
        plt.imsave(fname+'_attn.png',attn_out[i].T,cmap='gray')

