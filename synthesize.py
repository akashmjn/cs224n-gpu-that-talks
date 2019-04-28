#!/usr/local/bin/python3

import argparse
import logging
import os,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pdb

import tensorflow as tf
from src.graph import SynthesizeGraph, OldModelGraph
from src.utils import Params
from src.data_load import load_tokens_from_text 
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

def restore_checkpoints(sess,m1_dir,m2_dir):
    # Load saved models
    text2mel_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'TextEnc|AudioEnc|AudioDec|InputEmbed')
    ssrn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN')

    # text2mel_vars = [v for v in text2mel_vars if 'Stop_FC' not in v.name] # ignore Stop_FC
    saver1 = tf.train.Saver(var_list=text2mel_vars)
    saver1.restore(sess, tf.train.latest_checkpoint(m1_dir))
    print("Text2Mel Restored!")
    saver2 = tf.train.Saver(var_list=ssrn_vars)
    saver2.restore(sess, tf.train.latest_checkpoint(m2_dir))
    print("SSRN Restored!")   

def get_text_lengths(input_arr):
    n_batch, max_text_len= input_arr.shape[0], input_arr.shape[1]
    text_lengths = np.zeros((n_batch,)) 
    for i in range(n_batch):
        for j in range(max_text_len-1,-1,-1):
            if input_arr[i,j] != 0:
                text_lengths[i] = j
                break
    return text_lengths

def track_stop_preds(stop_preds,stop_idxs,stop_flags,t):
    # threshold 0.5 for stop sigmoid output
    if len(stop_preds.shape)>1: stop_preds = stop_preds[:,-1] # stop_preds is dim: n_batch, T
    stop_preds = stop_preds > 0.0 
    for j,stop_pred in enumerate(stop_preds):
        if stop_pred and not stop_flags[j]: # encountering for first time
            stop_idxs[j] = t-2              # compensate for silence padded to training batches
            stop_flags[j] = stop_pred

def synthesize(m1_dir,m2_dir,sample_dir,n_iter=150,test_data=None,lines=None,ref_db=30):
    # NOTE: currently passes all input sentences as one batch

    # Initialize params
    params1 = Params(os.path.join(m1_dir,'params.json'))
    params2 = Params(os.path.join(m2_dir,'params.json'))
    params = params1
    if test_data is not None:
        params.dict['test_data'] = test_data # setting this based on what passed in
    params.dict['n_iter'] = n_iter
    params.dict['ref_db'] = ref_db # output volume
    # Load text as int arrays
    if lines is None: # Toggle whether read in a file or text passed via function call
        # text inputs 
        input_arr, text_lengths = load_tokens_from_text(params)
    else:
        input_arr, text_lengths = load_tokens_from_text(params,lines)
    n_batch = input_arr.shape[0]
    params.dict['batch_size'] = n_batch
    # Create empty arrays 
    output_mel = np.zeros((n_batch,params.max_T,params.F)) 
    output_mag = np.zeros((n_batch,params.max_T,params.Fo))
    # create flags indicating if and where each input in batch has stopped
    stop_flags = np.array([False]*n_batch)
    stop_idxs = np.zeros((n_batch,),dtype=int)
    #last_attended = np.zeros((n_batch,))

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use single GPUs available
    g = SynthesizeGraph(params)      

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restore_checkpoints(sess,m1_dir,m2_dir)
    
        ## Step1: Pre-compute text encoding (K, V) = TextEncBlock(character sequence) 
        K, V = sess.run([g.K_pre,g.V_pre],{g.transcripts:input_arr})

        ## Step2: Iterate over t, qt = AudioEncBlock(S:t), rt = Attention(qt,KV), St = AudioDecBlock(qt,rt)
        # TODO: Fix constrained monotonic attention
        for t in range(1,params.max_T):
            if all(stop_flags): break # end of audio for all inputs in batch

            print('Mel frame {}/{}'.format(t+1,params.max_T),end='\r')
            # optimization: fixed-width window to encode previously generated frames 
            slice_window = max(0,t-100) # TODO: fix hardcoded value
            prev_slice = output_mel[:,slice_window:t,:]

            model_preds, stop_preds = sess.run([g.Y,g.YStoplogit],
                                                {g.K_inp:K,g.V_inp:V,g.S:prev_slice}) 
            output_mel[:,t,:] = model_preds[:,-1,:] 
            track_stop_preds(stop_preds,stop_idxs,stop_flags,t)

            # monotonic contrained attention softmax
            # model_preds, attn_out = sess.run([g.Yhat,g.A],
            #     {g.S:prev_slice,g.transcripts:input_arr,g.last_attended:last_attended})
            # last_attended += np.argmax(attn_out[:,-1,:],axis=1) # slicing out the last time frame, and moving attention window forward
            # last_attended = np.clip(last_attended,a_min=0,a_max=text_lengths-params.attn_window_size)
        # output_mag, attn_out = sess.run([g.Zhat,g.A],
        #         {g.S:output_mel,g.transcripts:input_arr,g.last_attended:last_attended})
   
        ## Step3: Process complete utterance and invert Z = SSRN(Y:T)
        # print("Truncating. Stop idxs: {}".format(stop_idxs)) # truncate mels evenly
        output_mel = output_mel[:,:max(stop_idxs),:]
        # Convert to magnitude spectrograms
        output_mag, attn_out = sess.run([g.Zhat,g.A],
                                            {g.K_inp:K,g.V_inp:V,g.S:output_mel})       
        output_mag_list = [ output_mag[i,:stop_idxs[i]*params.reduction_factor,:]
                            for i in range(n_batch) ] # truncate mags individually
        # Griffin-lim inversion relatively time-taking hence parallelizing
        print("Magnitude spectrograms generated, inverting ..")
        pool_args = {}
        pool_args['sample_dir'] = sample_dir
        pool_args['params'] = params
        pool_input_list = [(output_mag_list[i],i,pool_args) for i in range(n_batch)] 
        with Pool(cpu_count()) as p:
            p.map(invert_mag,pool_input_list)

        for i in range(n_batch):
            fname = os.path.join(sample_dir,'sample_{}'.format(i))
            print('Saving plots for sample: {}/{}'.format(i+1,n_batch))
            plt.imsave(fname+'_mel.png',output_mel[i].T,cmap='gray')
            plt.imsave(fname+'_mag.png',output_mag_list[i].T,cmap='gray')
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

    synthesize(m1_dir=args.m1,m2_dir=args.m2,sample_dir=args.sample_dir,n_iter=args.n_iter,test_data=args.test_data,ref_db=args.ref_db)
