#!/usr/local/bin/python3

import logging
import os,sys
import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug

from model import TextEncBlock, AudioEncBlock, AudioDecBlock, AttentionBlock, SSRNBlock
from utils import set_logger, Params, learning_rate_decay
from data_load import load_data, get_batch


class ModelGraph(object):
    """
    Encapsulates all the graph nodes used by the model for training / inference

    # initialize character embeddings
    # build graph 
    """
    def __init__(self,params,mode):
        self.params = params
        self.logger = set_logger(os.path.join(self.params.log_dir,
                                    self.params.model_name+'_'+mode+'.log') ) # sets path for logging
        with tf.variable_scope("gs"): # global step variable to track batch updates
            self.global_step = tf.Variable(0, name='global_step', trainable=False)                  

        # gets labels, mel spectrograms, full magnitude spectrograms, fnames, and total no of batches
        if 'train' in mode: 
            self.transcripts, self.Y, self.Z, self.fnames, self.num_batch = get_batch(params,'train_ssrn')
        if mode=='train_text2mel':
            self.build_text2mel(mode=mode,reuse=None) # TODO: Toggle for inference, maybe look at combined training?
        elif mode=='train_ssrn':
            self.build_ssrn(mode,reuse=None)
        tf.summary.merge_all()

    def build_ssrn(self,mode,reuse=None):
        """
        Creates graph for the SSRN model for either training or inference
        During training, for now takes true mels Y as input with target as the true mag Z
        """
        self.logger.info('Building training graph for SSRN')
        self.Zlogit, self.Zhat = SSRNBlock(self.Y,self.params.c,self.params.Fo,reuse=reuse)

        tf.summary.image('train/mag_target', tf.expand_dims(tf.transpose(self.Z[:1], [0, 2, 1]), -1))
        tf.summary.image('train/mag_hat', tf.expand_dims(tf.transpose(self.Zhat[:1], [0, 2, 1]), -1))
        tf.summary.image('train/mel_inp', tf.expand_dims(tf.transpose(self.Y[:1], [0, 2, 1]), -1))
        tf.summary.histogram('train/Zhat',self.Zhat)

        self.add_loss_op(mode)
        self.add_train_op()

    def build_text2mel(self,mode,reuse=None):
        """
        Creates graph for either training or inference
        """
        if mode=='train_text2mel':
            self.S = tf.pad(self.Y[:,:-1,:],[[0,0],[1,0],[0,0]]) # feedback input is one-prev-shifted target input) 
            self.logger.info('Initialized input character embeddings with dim: {}'.format(self.S.shape))
            self.add_input_embeddings(reuse)
            self.add_predict_op(reuse)
            self.add_loss_op(mode)
            self.add_train_op()

    def add_input_embeddings(self,reuse=None):
        """
        Args:
            reuse (bool): indicates whether to use variables already defined (from checkpoints)   
        Returns:
            L (tf.tensor): tf.float32 tensor obtained after looking up embeddings (shape: batch_size, N, e) 
        """
        with tf.variable_scope('InputEmbeddings',reuse=reuse):
            vocab_size, e = len(self.params.vocab), self.params.e
            # TODO: Add ability to make padding embedding zero
            embedding_mat = tf.get_variable(name='char_embeddings',shape=[vocab_size,e],
                dtype=tf.float32,trainable=True)
            self.L = tf.nn.embedding_lookup(embedding_mat,self.transcripts)

        return self.L
    
    def add_predict_op(self,reuse=None):
        
        # building training graph for Text2Mel
        self.logger.info('Building training graph for Text2Mel')
        self.KV = TextEncBlock(self.L,self.params.d)
        self.logger.info('Encoded KV with dim: {}'.format(self.KV.shape))
        self.Q = AudioEncBlock(self.S,self.params.d)
        self.logger.info('Encoded Q with dim: {}'.format(self.Q.shape))
        self.A , self.R = AttentionBlock(self.KV, self.Q)
        self.logger.info('Encoded R with dim: {}'.format(self.R.shape))
        self.RQ = tf.concat([self.R, self.Q],axis=2)
        self.logger.info('Concatenated RQ with dim: {}'.format(self.RQ.shape))
        self.Ylogit, self.Yhat = AudioDecBlock(self.RQ,self.params.F)
        self.logger.info('Decoded Yhat with dim: {}'.format(self.Yhat.shape))

        tf.summary.image('train/mel_target', tf.expand_dims(tf.transpose(self.Y[:1], [0, 2, 1]), -1))
        tf.summary.image('train/mel_hat', tf.expand_dims(tf.transpose(self.Yhat[:1], [0, 2, 1]), -1))
        tf.summary.image('train/A', tf.expand_dims(tf.transpose(self.A[:1], [0, 2, 1]), -1))
        tf.summary.histogram('train/Ylogit',self.Ylogit)
        tf.summary.histogram('train/Yhat',self.Yhat)
    
        return self.Ylogit, self.Yhat
    
    def add_loss_op(self,mode):
    
        if mode=='train_ssrn':
            target, pred, logit, sumlabel = self.Z, self.Zhat, self.Zlogit, 'train/Z'
        elif mode=='train_text2mel':
            target, pred, logit, sumlabel = self.Y, self.Yhat, self.Ylogit, 'train/Y'

        # compute loss (without guided attention loss for now)
        self.L1_loss = tf.reduce_mean(tf.abs(target-pred))
        assert len(self.L1_loss.shape.as_list())==0,'Loss not scalar, shape: {}'.format(self.L1_loss.shape)
        self.CE_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target,logits=logit))
        assert len(self.CE_loss.shape.as_list())==0,'Loss not scalar, shape: {}'.format(self.CE_loss.shape)
        self.loss = self.params.l1_loss_weight*self.L1_loss + self.CE_loss

        tf.summary.scalar('train/L1_loss',self.L1_loss)
        tf.summary.scalar('train/CE_loss',self.CE_loss)
        tf.summary.scalar('train/total_loss',self.loss)
        tf.summary.histogram(sumlabel,target)
    
        return self.loss, self.L1_loss, self.CE_loss

    def add_train_op(self):

        with tf.variable_scope('optimizer'):
            self.lr = learning_rate_decay(self.params,self.global_step) 
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=self.params.beta1,beta2=self.params.beta2)
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate,
            #                                         beta1=self.params.beta1,beta2=self.params.beta2)
    
            ## gradient clipping
            self.gvs = self.optimizer.compute_gradients(self.loss)
            self.clipped = []
            for grad, var in self.gvs:
                grad = tf.clip_by_value(grad, -self.params.grad_clip_value, self.params.grad_clip_value)
                self.clipped.append((grad, var))
            self.grad_norm = tf.global_norm([g for g,v in self.clipped])

            tf.summary.scalar('train/lr',self.lr)
            tf.summary.scalar('train/grad_global_norm',self.grad_norm)

        self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step) # increments gs      
        # self.train_op = self.optimizer.minimize(self.loss)


####### test functions ########

def test_graph_setup(mode='placeholder'):

    assert mode in ['placeholder']

    if mode=='placeholder':
        logger = set_logger('debug_tests.log') # logger output 
        # test tensor shape to be fed in
        e, d, F = 26, 5, 10
        L_len, Y_len = 4, 6
        Y =  tf.placeholder(dtype=tf.float32,shape=(None,Y_len,F)) # target mel frames 
        L =  tf.placeholder(dtype=tf.float32,shape=(None,L_len,e)) # input character embeddings
        S =  tf.pad(Y[:,:-1,:],[[0,0],[1,0],[0,0]]) # feedback input (one-previous target input) 
        # test tensor values used for test
        L_feed = np.zeros([1,L_len,e])
        L_feed[:,1,:] = 10 
        Y_feed = np.ones([1,Y_len,F])
        Y_feed[:,5,:] = 10    
    
        Ylogit, Yhat = add_predict_op(L,S,d,F)
        loss, L1_loss, CE_loss = add_loss_op(Y,Ylogit,Yhat,'text2mel')
    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            loss_out, L1_out, CE_out, Y_out = sess.run([loss, L1_loss, CE_loss, Yhat],{L:L_feed,Y:Y_feed})

    logger.info('Final loss: {:.2f}, L1: {:.2f}, CE: {:.2f}, \
        with output shape: {}'.format(loss_out, L1_out, CE_out, Y_out.shape))

    np.set_printoptions(precision=3)
    logger.info(Y_out)   


if __name__ == '__main__':

    # test_graph_setup()
    # params = Params('./runs/default/params.json')
    # fpaths, text_lengths, texts = load_data(params)
    # texts, mels, mags, fnames, num_batch = get_batch(params)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    test_graph_setup('placeholder')
