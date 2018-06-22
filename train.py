#!/usr/local/bin/python3

"""Train the model"""

import argparse
import logging
import os,sys
import pdb

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tqdm import tqdm
from src.graph import ModelGraph, UnsupervisedGraph
from src.utils import Params



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('params', help="Path to params.json file containing different hyperparameters")
    parser.add_argument('mode', help="Indicate which model to train. Options: train_text2mel, train_ssrn")
    parser.add_argument('--gpu', type=int, default=0,help="GPU to train on if multiple available")
    parser.add_argument('--chkp',help="(For direct transfer learning) path to checkpoint dir to be restored")
    args = parser.parse_args()

    params = Params(args.params)
    print('Running a training run with params from: {}'.format(args.params))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # default use single GPU
    # train_tfrecord_path = str(os.path.join(params.data_dir,'train.tfrecord'))

    gs = tf.train.get_or_create_global_step() 
    # g = ModelGraph(params,args.mode)
    g = UnsupervisedGraph(params,args.mode)
    logger = g.logger

    ### Hack-y approach to partial loading/transfer learning with MonitoredTrainingSession
    if hasattr(args,'chkp') and args.chkp:
        # restore everything except for input embeddings (which will vary based on vocab)
        # NOTE: init_fn of scaffold is only called if params.log_dir does not contain any checkpoints
        with tf.variable_scope('TransferLearnOps'):
            text2mel_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'TextEnc|AudioEnc|AudioDec')
            saver1 = tf.train.Saver(var_list=text2mel_vars)       
        def restore_text2mel_vars(scaffold,sess):
            saver1.restore(sess, tf.train.latest_checkpoint(args.chkp))
            print("Text2Mel pretrained variables restored!")       
        scaffold = tf.train.Scaffold(local_init_op=g.iterator_init_op,init_fn = restore_text2mel_vars)
    else:
        scaffold = tf.train.Scaffold(local_init_op=g.iterator_init_op)   

    hooks = [tf.train.StopAtStepHook(last_step=params.num_steps)]
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    with tf.train.MonitoredTrainingSession(
        scaffold=scaffold,checkpoint_dir=params.log_dir,hooks=hooks) as sess:

        while not sess.should_stop():
            sess.run(g.iterator_init_op)
            g.logger.info('Initialized iterator')                          
            for _ in tqdm(range(g.num_train_batch), total=g.num_train_batch, ncols=70, leave=False, unit='b'):
                _, global_step, loss_out, L1_out, CE_out = sess.run([g.train_op, gs,
                                                                            g.loss, g.L1_loss, g.CE_loss])
    
                if global_step % 50==0:
                    logger.info('Training loss at step {}: {:.4f}, L1: {:.4f}, CE: {:.4f}'.format(
                        global_step,loss_out, L1_out, CE_out))
                if sess.should_stop(): break # end condition 
    
            print(global_step)

            # training steps 
            # sess.run(g.iterator_init_op,
            #              feed_dict={g.tfrecord_path:train_tfrecord_path}
            #          )           
            # # validation steps
            # sess.run(g.iterator_init_op,
            #         {g.tfrecord_path:os.path.join(params.data_dir,'val.tfrecord')}
            #     )           
            # for _ in tqdm(range(g.num_val_batch), total=g.num_val_batch, ncols=70, leave=False, unit='b'):
            #     # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            #     loss_out, L1_out, CE_out, attn_loss = sess.run([g.loss, g.L1_loss, g.CE_loss, g.attn_loss])

    logger.info('Completed {} steps!'.format(global_step))
