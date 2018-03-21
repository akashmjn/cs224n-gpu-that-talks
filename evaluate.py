#!/usr/local/bin/python3

"""Scripts and utilities for evaluating models"""

import argparse
import logging
import os,sys

import tensorflow as tf
from tqdm import tqdm
from graph import ModelGraph
from utils import Params

def evaluate_model_preds(mode,params_path):
    """
    Given a log directory, generates predictions and returns loss metrics for full dataset. 
    mode - indicates 'val/train_text2mel/ssrn' and toggles accordingly. 
    """

    params = Params(params_path)
    print('Running predictions with model from: {}'.format(params_path))
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use all GPUs available
    params.dict['Qbatch'] = 2      # hacky - reusing batching from Supervisor
    params.dict['num_threads'] = 12 
    params.dict['num_buckets'] = 2 # simplifiying overkill queue params
    params.dict['batch_size'] = 64
    params.dict['attention_mode'] = 'guided' # gives as estimate of attention monotonocity
    g = ModelGraph(params,mode)
    logger = g.logger
    total_loss_avg, L1_loss_avg, CE_loss_avg, att_loss_avg = 0.0, 0.0, 0.0, 0.0

    sv = tf.train.Supervisor(logdir=params.log_dir,summary_op=None)
    with sv.managed_session() as sess:

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
            loss_out, L1_out, CE_out, att_out = sess.run([g.loss, g.L1_loss, g.CE_loss, g.att_loss])

            if _ % 20 == 0:
                logger.info('Prediction loss: {:.4f}, L1: {:.4f}, CE: {:.4f}, Att: {:.4f}'.format(
                    loss_out, L1_out, CE_out, att_out))
            total_loss_avg += loss_out/g.num_batch
            L1_loss_avg += L1_out/g.num_batch
            CE_loss_avg += CE_out/g.num_batch
            att_loss_avg += att_out/g.num_batch


    logger.info('Completed predictions: Avg loss: {:.4f}, L1: {:.4f}, CE: {:.4f}, Att: {:.4f}'.format(
                total_loss_avg, L1_loss_avg, CE_loss_avg, att_loss_avg))
    tf.reset_default_graph() # clean up in case of multiple function calls

    return total_loss_avg, L1_loss_avg, CE_loss_avg, att_loss_avg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--params', help="Path to params.json in a checkpoint directory to be evaluated")
    parser.add_argument('--mode', help="One of train/val_text2mel/ssrn")
    args = parser.parse_args()

    evaluate_model_preds(args.mode,args.params)

