#!/usr/local/bin/python3

import logging
import os,sys
import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug

from .model import TextEncBlock, AudioEncBlock, AudioDecBlock, AttentionBlock, SSRNBlock
from .utils import set_logger, Params, learning_rate_decay, get_timing_signal_1d
from .data_load import load_data, get_batch, get_batch_prepro


class ModelGraph(object):
    """
    Encapsulates all the graph nodes used by the model for training / inference

    # initialize character embeddings
    # build graph 
    """
    def __init__(self,params,mode):
        """
        Builds out the model graph in different modes depending on inference, or training 
        different parts of the model. 

        Args:
            params (utils.Params): Object containing various hyperparams for building model graph
            mode (str): Either of 'train_text2mel', 'train_ssrn', or 'synthesize'
        """
        self.params = params
        self.mode = mode
        self.logger = set_logger(os.path.join(self.params.log_dir,
                                    self.params.model_name+'_'+mode+'.log') ) # sets path for logging
        # with tf.variable_scope("gs"): # global step variable to track batch updates
        #     self.global_step = tf.Variable(0, name='global_step', trainable=False)                  
        self.global_step = tf.train.get_global_step()

        # gets labels, mel spectrograms, full magnitude spectrograms, fnames, and total no of batches
        if 'synthesize' not in mode: 
            if hasattr(params,'prepro') and params.prepro:
                # self.tfrecord_path = tf.placeholder(tf.string,name='tfrecord_path')
                self.tfrecord_path = tf.constant(os.path.join(params.data_dir,'train.tfrecord'))
                batch, self.iterator_init_op,\
                self.num_train_batch, self.num_val_batch = get_batch_prepro(
                        self.tfrecord_path,params,self.logger
                    ) 
                self.transcripts, self.Y, self.Z, self.Y_mask = batch['indexes'], batch['mels'], batch['mags'], batch['mels_mask']
                self.Y_stop_labels = 1 - self.Y_mask[:,:,0] # 0 when data exists, 1 when padded
            else:
                self.transcripts, self.Y, self.Z, self.fnames, self.num_train_batch = get_batch(params,mode,self.logger)
        if mode in ['train_text2mel','val_text2mel','synthesize']:
            self.build_text2mel(reuse=None) # TODO: maybe look at combined training?
        if mode in ['train_ssrn','val_ssrn','synthesize']:
            self.build_ssrn(reuse=None)
        tf.summary.merge_all()

    def build_ssrn(self,reuse=None):
        """
        Creates graph for the SSRN model for either training or inference
        During training, takes true mels Y as input with target as the true mag Z
        """
        assert self.mode in ['train_ssrn','val_ssrn','synthesize']
        self.logger.info('Building training graph for SSRN ...')

        if self.mode in ['train_ssrn','val_ssrn']:
            self.Zlogit, self.Zhat = SSRNBlock(self.Y,self.params.c,self.params.Fo,reuse=reuse) # input, labels: true mels, mags
            self.add_loss_op()
            self.add_train_op()
            tf.summary.image('train/mag_target', tf.expand_dims(tf.transpose(self.Z[:1], [0, 2, 1]), -1))
            tf.summary.image('train/mag_hat', tf.expand_dims(tf.transpose(self.Zhat[:1], [0, 2, 1]), -1))
            tf.summary.image('train/mel_inp', tf.expand_dims(tf.transpose(self.Y[:1], [0, 2, 1]), -1))
            tf.summary.histogram('train/Zhat',self.Zhat)           
        elif self.mode=='synthesize':
            self.Zlogit, self.Zhat = SSRNBlock(self.Yhat,self.params.c,self.params.Fo,reuse=reuse) # input: generated mels

    def build_text2mel(self,reuse=None):
        """
        Creates graph for either training or inference. During training, one-previous shifted targets 
        are used as the feedback input S. During inference (synthesis) a variable time-length input 
        consisting of mel frames generated so far is used as input.  
        """
        assert self.mode in ['train_text2mel','val_text2mel','synthesize']
        # building training graph for Text2Mel
        self.logger.info('Building training graph for Text2Mel ...')       

        if self.mode in ['train_text2mel','val_text2mel']:
            self.S = tf.pad(self.Y[:,:-1,:],[[0,0],[1,0],[0,0]]) # feedback input: one-prev-shifted target input) 
        elif self.mode=='synthesize':
            self.S = tf.placeholder(dtype=tf.float32,shape=[None,None,self.params.F]) # mels generated so far
            self.last_attended = tf.placeholder(dtype=tf.int32,shape=[self.params.batch_size]) # batch_size with int indexes 
            self.transcripts = tf.placeholder(dtype=tf.int32,shape=[None,None]) # int encoded input text to synthesize

        self.add_input_embeddings(reuse)
        self.logger.info('Initialized input character embeddings with dim: {}'.format(self.L.shape))

        if self.params.pos_encoding:
            self.add_pos_encodings()
            self.logger.info('Initialized position encodings, shapes: {}, {}'.format(self.K_pos.shape,self.Q_pos.shape))

        self.add_predict_op(reuse)
        if self.mode in ['train_text2mel','val_text2mel']:
            self.add_loss_op()
            self.add_train_op()

    def add_input_embeddings(self,reuse=None):
        """
        Args:
            reuse (bool): indicates whether to use variables already defined (from checkpoints)   
        Returns:
            L (tf.tensor): tf.float32 tensor obtained after looking up embeddings (shape: batch_size, N, e) 
        """
        with tf.variable_scope('InputEmbeddings',reuse=reuse):
            # from Gehring et. al (2017), sizing e the same as d if input embedding is to be added
            e = self.params.d if self.params.local_encoding else self.params.e
            vocab_size = len(self.params.vocab)
            # TODO: Add ability to make padding embedding zero
            embedding_mat = tf.get_variable(name='char_embeddings',shape=[vocab_size,e],
                dtype=tf.float32,trainable=True)
            self.L = tf.nn.embedding_lookup(embedding_mat,self.transcripts)

        return self.L

    def add_pos_encodings(self,reuse=None):
        """
        Args:
            reuse (bool): reuse of variable scope
        Returns:
            K_pos (tf.tensor) (shape: 1, max_N, d)
            Q_pos (tf.tensor) (shape: 1, max_T, d)
        """

        self.K_pos = get_timing_signal_1d(self.params.max_N,self.params.d,self.params.pos_rate)
        self.Q_pos = get_timing_signal_1d(self.params.max_T,self.params.d)

        return self.K_pos, self.Q_pos

    
    def add_predict_op(self,reuse=None):
        
        self.K, self.V = TextEncBlock(self.L,self.params.d)

        if self.params.local_encoding:            # as in Gehring et. al (2017) uses input embedding L in value
            self.V = tf.sqrt(0.5)*(self.L+self.V) # weighted sum with V, L 

        self.logger.info('Encoded K, V with dim: {}'.format(self.K.shape))
        self.Q = AudioEncBlock(self.S,self.params.d)
        self.logger.info('Encoded Q with dim: {}'.format(self.Q.shape))

        if self.params.pos_encoding:             # as in Gehring et. al (2017) adding these to precondition monotonicity
            self.K = self.K + self.K_pos[0,:tf.shape(self.K)[1],:]
            self.Q = self.Q + self.Q_pos[0,:tf.shape(self.Q)[1],:]

        if self.mode=='synthesize':
            self.A , self.R = AttentionBlock(self.K, self.V, self.Q,
                last_attended=self.last_attended,attn_window_size=self.params.attn_window_size)
        else:
            self.A , self.R = AttentionBlock(self.K, self.V, self.Q)
        self.logger.info('Encoded R with dim: {}'.format(self.R.shape))
        self.RQ = tf.concat([self.R, self.Q],axis=2)
        self.logger.info('Concatenated RQ with dim: {}'.format(self.RQ.shape))
        self.Ylogit, self.Yhat, self.YStoplogit = AudioDecBlock(self.RQ,self.params.F)
        self.logger.info('Decoded Yhat with dim: {}'.format(self.Yhat.shape))
    
        return self.Ylogit, self.Yhat
    
    def add_loss_op(self):
    
        if 'ssrn' in self.mode:
            target, pred, logit, sumlabel = self.Z, self.Zhat, self.Zlogit, 'train/Z'
        elif 'text2mel' in self.mode:
            target, pred, logit, sumlabel = self.Y, self.Yhat, self.Ylogit, 'train/Y'
            tf.summary.image('train/A', tf.expand_dims(tf.transpose(self.A[:1], [0, 2, 1]), -1))

        # compute loss (without guided attention loss for now)
        self.L1_loss = tf.reduce_sum(
                tf.abs(target-pred)*self.Y_mask
                )/tf.reduce_sum(self.Y_mask) # padded batches are masked
        assert len(self.L1_loss.shape.as_list())==0,'Loss not scalar, shape: {}'.format(self.L1_loss.shape)
        self.CE_loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=target,logits=logit)*self.Y_mask
            )/tf.reduce_sum(self.Y_mask)     # padded batches are masked
        assert len(self.CE_loss.shape.as_list())==0,'Loss not scalar, shape: {}'.format(self.CE_loss.shape)
        self.stop_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y_stop_labels,logits=self.YStoplogit)
            )                                # stop predictions are not masked 
        self.loss = self.params.l1_loss_weight*self.L1_loss + self.params.CE_loss_weight*self.CE_loss + self.stop_loss

        # guided attention loss from Tachibana et. al (2017)
        if self.params.attention_mode =='guided' and 'ssrn' not in self.mode:
            # A (shape: batch_size, N, T) - these dimensions are fixed for a single padded batch
            N, T = tf.cast(tf.shape(self.A)[1],tf.float32), tf.cast(tf.shape(self.A)[2],tf.float32)
            W = tf.fill(tf.shape(self.A),0.0) # weight matrix to be multiplied with A
            W = W + tf.expand_dims(tf.range(N),1)/N - tf.expand_dims(tf.range(T),0)/T # using broadcasting for mat + col - row
            self.W_att = 1.0 - tf.exp(-tf.square(W)/(2*0.2)**2) # using g=0.2 from paper
            self.att_loss = tf.reduce_mean(tf.multiply(self.A,self.W_att))
            tf.summary.scalar('train/att_loss',self.att_loss)
            self.loss = self.loss + self.att_loss
            self.logger.info('Added guided attention loss over A: {}'.format(self.A))

        tf.summary.image(sumlabel+'_target', tf.expand_dims(tf.transpose(target[:1], [0, 2, 1]), -1))
        tf.summary.image(sumlabel+'_pred', tf.expand_dims(tf.transpose(pred[:1], [0, 2, 1]), -1))
        tf.summary.histogram(sumlabel+'_target',target)
        tf.summary.histogram(sumlabel+'_logit',logit)
        tf.summary.histogram(sumlabel+'_pred',pred)
        tf.summary.scalar('train/L1_loss',self.L1_loss)
        tf.summary.scalar('train/CE_loss',self.CE_loss)
        tf.summary.scalar('train/total_loss',self.loss)
    
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
                try:
                    grad = tf.clip_by_value(grad, -self.params.grad_clip_value, self.params.grad_clip_value)
                    self.clipped.append((grad, var))
                except Exception as e:
                    print(grad)
            self.grad_norm = tf.global_norm([g for g,v in self.clipped])

            tf.summary.scalar('train/lr',self.lr)
            tf.summary.scalar('train/grad_global_norm',self.grad_norm)

        self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step) # increments gs      
        # self.train_op = self.optimizer.minimize(self.loss)


####### test functions (might be slightly deprecated) ########

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
