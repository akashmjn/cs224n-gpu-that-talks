#!/usr/local/bin/python3

"""Scripts and utilities for evaluating models"""

import argparse
import logging
import os,sys

import tensorflow as tf
from tqdm import tqdm
from graph import ModelGraph
from utils import Params


if __name__ == '__main__':

    params_path = sys.argv[1]
    params = Params(params_path)
    print('Running predictions with model from: {}'.format(params_path))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # default use single GPU
    g = ModelGraph(params,'val_text2mel')
    logger = g.logger
    total_loss_avg, L1_loss_avg, CE_loss_avg = 0.0, 0.0, 0.0 

    sv = tf.train.Supervisor(logdir=params.log_dir,summary_op=None)
    with sv.managed_session() as sess:

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
            loss_out, L1_out, CE_out = sess.run([g.loss, g.L1_loss, g.CE_loss])
            logger.info('Prediction loss: {:.2f}, L1: {:.2f}, CE: {:.2f}'.format(
                loss_out, L1_out, CE_out))
            total_loss_avg += loss_out/g.num_batch
            L1_loss_avg += L1_out/g.num_batch
            CE_loss_avg += CE_loss_avg/g.num_batch


    logger.info('Completed predictions: Avg loss: {:.2f}, L1: {:.2f}, CE: {:.2f}'.format(
                total_loss_avg, L1_loss_avg, CE_loss_avg))
