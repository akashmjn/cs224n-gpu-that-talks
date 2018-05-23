#!/usr/local/bin/python3

import argparse
import logging
import os,sys
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pdb

import tensorflow as tf
from src.graph import ModelGraph
from src.utils import Params
from src.data_load import load_vocab, text_normalize, load_data
from src.dsp_utils import spectrogram2wav, save_wav

def invert_mag(inp_triple):
    """
    Processes a magnitude spectrogram and saves to corresponding audio file. 
    Used by multiprocessing pool function uses an input tuple as a hack for 
    additional arguments needed for printing/saving. 

    Args:
        inp_triple (tuple): (mags_np.ndarray,i,pool_args) pool_args (dict) - sample_dir (output_dir), params
    """
    
    output_mag, i, pool_args = inp_triple
    sample_dir = pool_args['sample_dir']
    params = pool_args['params']

    print('Generating full audio for sample {}'.format(i))
    wav = spectrogram2wav(output_mag,params) # adding some silence before
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    fname = os.path.join(sample_dir,'sample_{}'.format(i))
    save_wav(wav,fname+'.wav',params.sampling_rate)   

def synthesize(m1_dir,m2_dir,sample_dir,n_iter=150,test_data=None,lines=None,ref_db=30):
    # NOTE: currently passes all input sentences as one batch

    # Initialize graph, path to model checkpoints
    params1 = Params(os.path.join(m1_dir,'params.json'))
    params2 = Params(os.path.join(m2_dir,'params.json'))
    params = params1
    if test_data is not None:
        params.dict['test_data'] = test_data # setting this based on what passed in
    params.dict['n_iter'] = n_iter
    params.dict['ref_db'] = ref_db # output volume

    if lines is None: # Toggle whether read in a file or text passed via function call
        # text inputs 
        input_arr = load_data(params,'synthesize')
    else:
        input_arr = load_data(params,'demo',lines)

    # shape (1, len)
    n_batch, max_text_len = input_arr.shape[0], input_arr.shape[1]
    text_lengths = np.zeros((n_batch,)) 
    for i in range(n_batch):
        for j in range(max_text_len-1,-1,-1):
            if input_arr[i,j] != 0:
                text_lengths[i] = j
                break
    params.dict['batch_size'] = n_batch
    output_mel = np.zeros((n_batch,params.max_T,params.F)) 
    output_mag = np.zeros((n_batch,params.max_T,params.Fo))
    last_attended = np.zeros((n_batch,))

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use single GPUs available
    g = ModelGraph(params,'synthesize')      

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        # Load saved models
        text2mel_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'TextEnc|AudioEnc|AudioDec|InputEmbed')
        ssrn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN')
    
        saver1 = tf.train.Saver(var_list=text2mel_vars)
        saver1.restore(sess, tf.train.latest_checkpoint(m1_dir))
        print("Text2Mel Restored!")
        saver2 = tf.train.Saver(var_list=ssrn_vars)
        saver2.restore(sess, tf.train.latest_checkpoint(m2_dir))
        print("SSRN Restored!")   
    
        # create flags indicating if and where each input in batch has stopped
        stop_flags = np.array([False]*n_batch)
        stop_idxs = np.zeros((n_batch,))

        # Generate all the mel frames
        # TODO: Fix constrained monotonic attention 
        for i in range(1,params.max_T):
            if all(stop_flags): break # end of audio for all inputs in batch

            print(last_attended)
            print('Mel frame {}/{}'.format(i+1,params.max_T),end='\r')
            prev_slice = output_mel[:,:i,:]

            model_preds, stop_preds = sess.run([g.Yhat,g.YStoplogit]) 
            # threshold 0.5 for stop sigmoid output
            stop_preds = stop_preds > 0.0
            for j,stop_pred in enumerate(stop_preds):
                if stop_pred and not stop_flags[j]: # encountering for first time
                    stop_idxs[j] = i
                    stop_flags[j] = stop_pred

            # monotonic contrained attention softmax
            # model_preds, attn_out = sess.run([g.Yhat,g.A],
            #     {g.S:prev_slice,g.transcripts:input_arr,g.last_attended:last_attended})
            # last_attended += np.argmax(attn_out[:,-1,:],axis=1) # slicing out the last time frame, and moving attention window forward
            # last_attended = np.clip(last_attended,a_min=0,a_max=text_lengths-params.attn_window_size)

            output_mel[:,i,:] = model_preds[:,-1,:]
    
        # truncate mel predictions using stop_idxs 
        for i,stop_idx in enumerate(stop_idxs): output_mel[i,stop_idx:,:] = 0
        # Convert to magnitude spectrograms
        output_mag, attn_out = sess.run([g.Zhat,g.A],{g.S:output_mel,g.transcripts:input_arr})       
        # output_mag, attn_out = sess.run([g.Zhat,g.A],
        #         {g.S:output_mel,g.transcripts:input_arr,g.last_attended:last_attended})
        print("Magnitude spectrograms generated, inverting ..")
        pool_args = {}
        pool_args['sample_dir'] = sample_dir
        pool_args['params'] = params

        mags_list = [ (output_mag[i],i,pool_args) for i in range(n_batch)]

        # Griffin-lim inversion seems to be relatively time-taking hence parallelizing
        with Pool(cpu_count()) as p:
            p.map(invert_mag,mags_list)
        for i in range(n_batch):
            fname = os.path.join(sample_dir,'sample_{}'.format(i))
            print('Saving plots for sample: {}/{}'.format(i+1,n_batch))
            plt.imsave(fname+'_mel.png',output_mel[i].T,cmap='gray')
            plt.imsave(fname+'_mag.png',output_mag[i].T,cmap='gray')
            plt.imsave(fname+'_attn.png',attn_out[i].T,cmap='gray')

    tf.reset_default_graph()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('m1', help="Checkpoint directory for Text2Mel")
    parser.add_argument('m2', help="Checkpoint directory for SSRN")
    parser.add_argument('test_data', help="Data file containing sentences to synthesize")
    parser.add_argument('--n_iter',type=int,default=150, help="Number of Griffin lim iterations to run for inversion (default: 150)")
    parser.add_argument('--sample_dir', default="../samples",
                        help="Directory to save generated samples.")
    parser.add_argument('--ref_db',type=int,default=30,help="Output audio volume (default: 30)")
    args = parser.parse_args()

    synthesize(args.m1,args.m2,args.sample_dir,args.n_iter,args.test_data,ref_db=args.ref_db)
