#!/usr/local/bin/python3

"""Train the model"""

import argparse
import logging
import os,sys
import pdb

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tqdm import tqdm
from src.graph import ModelGraph, Text2MelTrainGraph, SSRNTrainGraph, UnsupervisedTrainGraph
from src.utils import Params


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('params', help="Path to params.json file containing different hyperparameters")
    parser.add_argument('mode', help="Indicate which model to train. Options: train_text2mel, train_ssrn")
    parser.add_argument('--gpu', type=int, default=0,help="GPU to train on if multiple available")
    parser.add_argument('--chkp',help="(For direct transfer learning) path to checkpoint dir to be restored")
    parser.add_argument('--restore-vars',help="tf.GraphKey used to restore variables from CHKP",default='TextEnc|AudioEnc|AudioDec')
    parser.add_argument('--train-vars',help="tf.GraphKey used to update variables in training",
                                        default='InputEmbed|TextEnc|AudioEnc|AudioDec')
    args = parser.parse_args()

    params = Params(args.params)
    print('Running a training run with params from: {}'.format(args.params))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # default use single GPU

    # Add trainable variables to params
    params.dict['trainable_vars'] = args.train_vars

    # Parse mode and setup graph 
    gs = tf.train.get_or_create_global_step() 
    if args.mode in 'train_text2mel':
        g = Text2MelTrainGraph(params)
    elif args.mode in 'train_ssrn':
        g = SSRNTrainGraph(params)
    elif args.mode in 'train_unsupervised':
        g = UnsupervisedTrainGraph(params)
    else:
        raise Exception('Unsupported mode')
    logger = g.logger

    ### partial loading/transfer learning hack with MonitoredTrainingSession 
    if args.chkp:
        # restore everything except for input embeddings (which will vary based on vocab)
        with tf.variable_scope('TransferLearnOps'):
            restored_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, args.restore_vars)
            saver = tf.train.Saver(var_list=restored_vars)       
        def restore_pretrained_vars(scaffold,sess):
            logger.info("Restoring pretrained variables {} from {}".format(args.restore_vars,args.chkp))
            saver.restore(sess, tf.train.latest_checkpoint(args.chkp))
            print("Text2Mel pretrained variables restored!")       
        # NOTE: init_fn of scaffold is only called if params.log_dir does not contain any checkpoints
        scaffold = tf.train.Scaffold(local_init_op=g.iterator_init_op,init_fn = restore_pretrained_vars)
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
                if sess.should_stop(): 
                    sess.close()
                    break # end condition 
    
            print(global_step)

    logger.info('Completed {} steps!'.format(global_step))

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
