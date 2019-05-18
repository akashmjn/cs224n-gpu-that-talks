#!/usr/local/bin/python3

import logging
import os,sys
import tensorflow as tf
import numpy as np
import sentencepiece as spm
from tensorflow.python import debug as tf_debug

from .model import TextEncBlock, AudioEncBlock, AudioDecBlock, AttentionBlock, SSRNBlock
from .utils import set_logger, Params, learning_rate_decay, get_timing_signal_1d
from .data_load import TFRecordDataloader, load_tokens_from_text


class ModelGraph(object):

    def __init__(self,params):
        self.params = params
        self.logger = set_logger(os.path.join(self.params.log_dir,
                                    self.params.model_name+'.log') ) # sets path for logging
        self.global_step = tf.train.get_global_step()       
        self._add_data_input()
        self._build()
        self._add_loss_op()
        self._add_train_op()
        self._add_inference_op()
        self._add_tboard_summaries()

    def _add_data_input(self):
        pass
    def _build(self):
        pass
    def _add_loss_op(self):
        pass
    def _add_train_op(self):
        pass
    def _add_inference_op(self):
        pass

    def _add_input_embeddings(self,tokens):
        """
        Args:
            reuse (bool): indicates whether to use variables already defined (from checkpoints)   
        Returns:
            L (tf.tensor): tf.float32 tensor obtained after looking up embeddings (shape: batch_size, N, e) 
        """
        with tf.variable_scope('InputEmbeddings'):
            # from Gehring et. al (2017), sizing e the same as d if input embedding is to be added
            e = self.params.d if self.params.local_encoding else self.params.e
            # TODO: Add ability to make padding embedding zero
            embedding_mat = tf.get_variable(name='char_embeddings',shape=[self.vocab_size,e],
                dtype=tf.float32,trainable=True)
            L = tf.nn.embedding_lookup(embedding_mat,tokens)
        return L

    def _add_text_encoder(self,L):
        K, V = TextEncBlock(L,self.params.d)
        if self.params.local_encoding:            # as in Gehring et. al (2017) uses input embedding L in value
            V = tf.sqrt(0.5)*(L+V) # weighted sum with V, L 
        self.logger.info('Encoded input text to K, V with dim: {}'.format(K.shape))       
        return K,V

    def _add_audio_encoder(self,S):
        Q = AudioEncBlock(S,self.params.d)
        if self.params.dropout_rate > 0:
            Q = tf.nn.dropout(Q,1-self.params.dropout_rate)
        self.logger.info('Encoded input audio to Q with dim: {}'.format(Q.shape))       
        return Q

    def _add_attention(self,K,V,Q):
        if self.params.pos_encoding:             # as in Gehring et. al (2017) adding these to precondition monotonicity
            K_pos, Q_pos = self._get_pos_encodings()
            K = K + K_pos[0,:tf.shape(K)[1],:]
            Q = Q + Q_pos[0,:tf.shape(Q)[1],:]
        A, R = AttentionBlock(K, V, Q)
        self.logger.info('Encoded context vector R with dim: {}'.format(R.shape))       
        return K, V, Q, A, R

    def _add_audio_decoder(self,R,Q,reuse=None):
        RQ = tf.concat([R, Q],axis=2)
        self.logger.info('Concatenated RQ with dim: {}'.format(RQ.shape))
        Ylogit, Yhat, YStoplogit = AudioDecBlock(RQ,self.params.d,self.params.F,reuse=reuse)
        self.logger.info('Decoded Yhat with dim: {}'.format(Yhat.shape))       
        return Ylogit, Yhat, YStoplogit

    def _get_pos_encodings(self):
        """
        Args:
            reuse (bool): reuse of variable scope
        Returns:
            K_pos (tf.tensor) (shape: 1, max_N, d)
            Q_pos (tf.tensor) (shape: 1, max_T, d)
        """
        with tf.variable_scope('PositionalEncodings'):
            K_pos = get_timing_signal_1d(self.params.max_N,self.params.d,self.params.pos_rate)
            Q_pos = get_timing_signal_1d(self.params.max_T,self.params.d)
        return K_pos, Q_pos       

    def _add_tboard_summaries(self):
        pass

class ModelTrainGraph(ModelGraph):

    def _add_data_input(self):
        if self.params.prepro: # reading pre-processed data from .tfrecord (recommended) 
            tfrecord_path = tf.constant(os.path.join(self.params.data_dir,'train.tfrecord'))
            self.dataloader = TFRecordDataloader(self.params,tfrecord_path,self.logger)
            self.iterator_init_op = self.dataloader.iterator_init_op
            self.num_train_batch = self.dataloader.num_batch_train
            self.num_val_batch = self.dataloader.num_batch_val
            self.vocab_size = self.dataloader.vocab_size
            batch = self.dataloader.get_batch()
            self.tokens, self.Y, self.Z, self.Y_mask = batch['tokens'], batch['mels'], batch['mags'], batch['mels_mask']
            self.Y_stop_labels = 1 - self.Y_mask[:,:,0] # 0 when data exists, 1 when padded
        else:
            self.tokens, self.Y, self.Z, self.fnames, self.num_train_batch = get_batch(self.params,mode,self.logger)       

    def _add_loss_op(self):
        # compute loss (without guided attention loss for now)
        with tf.variable_scope('LossOps'):
            self.L1_loss = tf.reduce_sum(
                    tf.abs(self.target-self.pred)*self.Y_mask
                    )/tf.reduce_sum(self.Y_mask) # padded batches are masked
            assert len(self.L1_loss.shape.as_list())==0,'Loss not scalar, shape: {}'.format(self.L1_loss.shape)
            self.CE_loss = tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target,logits=self.logit)*self.Y_mask
                )/tf.reduce_sum(self.Y_mask)     # padded batches are masked
            assert len(self.CE_loss.shape.as_list())==0,'Loss not scalar, shape: {}'.format(self.CE_loss.shape)
            self.stop_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y_stop_labels,logits=self.YStoplogit)
                )                                # stop predictions are not masked 
            self.loss = self.params.l1_loss_weight*self.L1_loss + self.params.CE_loss_weight*self.CE_loss + self.stop_loss
            self.logger.info('Added masked L1, CE and stop loss ops ...')

            # guided attention loss from Tachibana et. al (2017)
            if self.params.attention_mode =='guided' and 'ssrn' not in self.mode:
                # A (shape: batch_size, N, T) - these dimensions are fixed for a single padded batch
                N, T = tf.cast(tf.shape(self.A)[1],tf.float32), tf.cast(tf.shape(self.A)[2],tf.float32)
                W = tf.fill(tf.shape(self.A),0.0) # weight matrix to be multiplied with A
                W = W + tf.expand_dims(tf.range(N),1)/N - tf.expand_dims(tf.range(T),0)/T # using broadcasting for mat + col - row
                self.W_att = 1.0 - tf.exp(-tf.square(W)/(2*0.2)**2) # using g=0.2 from paper
                self.att_loss = tf.reduce_mean(tf.multiply(self.A,self.W_att))
                self.loss = self.loss + self.att_loss
                self.logger.info('Added guided attention loss over A: {}'.format(self.A))       

    def _add_train_op(self):
        with tf.variable_scope('optimizer'):
            self.lr = learning_rate_decay(self.params,self.global_step) 
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=self.params.beta1,beta2=self.params.beta2)
            ## gradient clipping
            self.gvs = self.optimizer.compute_gradients(self.loss)
            self.clipped = []
            self.logger.info('Adding training op only for variables in scope: {}'.format(self.params.trainable_vars))
            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,self.params.trainable_vars)
            self.trainable_gvs = [(g,v) for g,v in self.gvs if v in self.trainable_vars]
            for grad, var in self.trainable_gvs:
                try:
                    grad = tf.clip_by_norm(grad, self.params.grad_clip_value)
                    self.clipped.append((grad, var))
                except Exception as e:
                    print(grad)
        self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step) # increments gs             

    def _add_tboard_summaries(self):
        self._add_basic_summaries()
        self._add_additional_summaries()
        tf.summary.merge_all()

    def _add_basic_summaries(self):
        tf.summary.image(self._tboard_label+'_target', tf.expand_dims(tf.transpose(self.target[:1], [0, 2, 1]), -1))
        tf.summary.image(self._tboard_label+'_pred', tf.expand_dims(tf.transpose(self.pred[:1], [0, 2, 1]), -1))
        tf.summary.histogram(self._tboard_label+'_target',self.target)
        tf.summary.histogram(self._tboard_label+'_logit',self.logit)
        tf.summary.histogram(self._tboard_label+'_pred',self.pred)
        tf.summary.scalar('train/lr',self.lr)
        tf.summary.scalar('train/L1_loss',self.L1_loss)
        tf.summary.scalar('train/CE_loss',self.CE_loss)
        tf.summary.scalar('train/total_loss',self.loss)                     

    def _add_additional_summaries(self):
        pass

class Text2MelTrainGraph(ModelTrainGraph):

    def _build(self):
        self.logger.info('Building training graph for Text2Mel ...')       
        self._add_data_input()
        self.L = self._add_input_embeddings(self.tokens)
        self.S = tf.pad(self.Y[:,:-1,:],[[0,0],[1,0],[0,0]]) # feedback input: one-prev-shifted target input) 
        self.K, self.V = self._add_text_encoder(self.L)
        self.Q = self._add_audio_encoder(self.S)
        self.K, self.V, self.Q, self.A, self.R = self._add_attention(self.K,self.V,self.Q)
        tf.summary.image('train/A', tf.expand_dims(tf.transpose(self.A[:1], [0, 2, 1]), -1))
        self.Ylogit, self.Yhat, self.YStoplogit = self._add_audio_decoder(self.R,self.Q)
        self.target, self.pred, self.logit, self._tboard_label = self.Y, self.Yhat, self.Ylogit, 'train/Y'

    def _add_additional_summaries(self):
        if self.params.attention_mode == 'guided':
            tf.summary.scalar('train/att_loss',self.att_loss)
        
        with tf.variable_scope('Dataloading'):
            tf.summary.scalar('train/len_tokens',tf.shape(self.tokens)[1])
            tf.summary.scalar('train/len_frames',tf.shape(self.Y)[1])
    
        with tf.variable_scope('GradSummaries'):
            embed_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'InputEmbed')
            textenc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'TextEnc')
            audioenc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'AudioEnc')
            audiodec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'AudioDec')
            grad_norm_embed = tf.global_norm([g for g,v in self.clipped if v in embed_vars])
            grad_norm_textenc = tf.global_norm([g for g,v in self.clipped if v in textenc_vars])
            grad_norm_audioenc = tf.global_norm([g for g,v in self.clipped if v in audioenc_vars])
            grad_norm_audiodec = tf.global_norm([g for g,v in self.clipped if v in audiodec_vars])
            tf.summary.scalar('train/grad_norm_embed',grad_norm_embed)
            tf.summary.scalar('train/grad_norm_textenc',grad_norm_textenc)
            tf.summary.scalar('train/grad_norm_audioenc',grad_norm_audioenc)
            tf.summary.scalar('train/grad_norm_audiodec',grad_norm_audiodec)       

class SSRNTrainGraph(ModelTrainGraph):
    
    def _build(self):
        self.logger.info('Building training graph for SSRN ...')
        self._add_data_input()     
        self.Zlogit, self.Zhat = SSRNBlock(self.Y,self.params.c,self.params.Fo) # input, labels: true mels, mags
        self.target, self.pred, self.logit, self._tboard_label = self.Z, self.Zhat, self.Zlogit, 'train/Z'

    def _add_loss_op(self):
        with tf.variable_scope('LossOps'):
            self.L1_loss = tf.reduce_mean(tf.abs(self.target-self.pred)) # batches are now samples patches
            assert len(self.L1_loss.shape.as_list())==0,'Loss not scalar, shape: {}'.format(self.L1_loss.shape)
            self.CE_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target,logits=self.logit)
                )  # padded batches are masked
            assert len(self.CE_loss.shape.as_list())==0,'Loss not scalar, shape: {}'.format(self.CE_loss.shape)
            self.loss = self.params.l1_loss_weight*self.L1_loss + self.params.CE_loss_weight*self.CE_loss 
            self.logger.info('Added L1, CE loss ops ...')

class UnsupervisedTrainGraph(ModelTrainGraph):

    def _build(self):
        self.logger.info('Building training graph for unsupervised training on audio ...')
        self._add_data_input()
        self.S = tf.pad(self.Y[:,:-1,:],[[0,0],[1,0],[0,0]]) # one-prev-shifted target input) 
        self.Q = self._add_audio_encoder(self.S)
        self.R = self._add_unsupervised_attn_input()
        self.Ylogit, self.Yhat, self.YStoplogit = self._add_audio_decoder(self.R,self.Q)
        self.target, self.pred, self.logit, self._tboard_label = self.Y, self.Yhat, self.Ylogit, 'train/Y'

    def _add_unsupervised_attn_input(self):
        with tf.variable_scope('UnsupervisedInput'):
            self.unsupervised_input = tf.get_variable(name='unsupervised_input',shape=[self.params.d],
                dtype=tf.float32,trainable=True)
            R = tf.zeros_like(self.Q) + self.unsupervised_input
        self.logger.info('Added trainable R broadcast to dim: {}'.format(R.shape))       
        return R

class SynthesizeGraph(ModelGraph):

    # Steps for synthesis: 
    # assign(K, V) = TextEncBlock(character sequence) 
    # Iter t, qt = AudioEncBlock(Y:t), rt = Attention(qt,KV), Yt = AudioDecBlock(qt,rt)
    # Z = SSRN(Y:T)
   
    def _add_data_input(self):
        self.spm_model = spm.SentencePieceProcessor()
        self.spm_model.Load(self.params.spm_model)
        self.vocab_size = self.spm_model.GetPieceSize()
        self.S = tf.placeholder(dtype=tf.float32,shape=[None,None,self.params.F]) # mels generated so far
        self.tokens = tf.placeholder(dtype=tf.int32,shape=[None,None]) # int encoded input text to synthesize
        self.K_inp = tf.placeholder(dtype=tf.float32,shape=[None,None,self.params.d]) # pre-computed text encoding 
        self.V_inp = tf.placeholder(dtype=tf.float32,shape=[None,None,self.params.d]) # pre-computed text encoding
        self.t = tf.placeholder(dtype=tf.int32,shape=()) # current timestep (required for pos encodings)

    def _add_attention_t(self,K,V,Qt):
        if self.params.pos_encoding:             # as in Gehring et. al (2017) adding these to precondition monotonicity
            K_pos, Q_pos = self._get_pos_encodings()
            K = K + K_pos[0,:tf.shape(K)[1],:]
            Qt = Qt + Q_pos[0,self.t,:]
        A, R = AttentionBlock(K, V, Qt)
        self.logger.info('Encoded context vector Rt with dim: {}'.format(R.shape))       
        return K, V, Qt, A, R
    
    def _build(self):
        self.logger.info("Building inference graph ...")
        self.params.dropout_rate = 0.0
        # Add embeddings lookup 
        self.L = self._add_input_embeddings(self.tokens)
        self.K_pre, self.V_pre = self._add_text_encoder(self.L)
        self.Q = self._add_audio_encoder(self.S)
        # # Iteration t: qt = AudioEncBlock(S:t)[t], rt = Attention(qt,KV), Yt = AudioDecBlock(qt,rt)
        # self.Qt = self.Q[:,-1:,:] # qt = AudioEncBlock(S:t)[t], need only last context vector
        # self.At, self.Rt = self._add_attention_t(self.K_pre,self.V_pre,self.Qt)
        # _, self.Yt, self.YStopt = self._add_audio_decoder(self.Rt,self.Qt)
        # Full utterance computation for SSRN
        self.K, self.V, self.Q, self.A, self.R = self._add_attention(self.K_inp,self.V_inp,self.Q)
        self.Ylogit, self.Y, self.YStoplogit = self._add_audio_decoder(self.R,self.Q,reuse=False)
        self.Zlogit, self.Zhat = SSRNBlock(self.Y,self.params.c,self.params.Fo) # input, labels: true mels, mags
        self.logger.info('Built SSRN with output dim: {}'.format(self.Zhat.shape))       

####### Old code to be removed #######

class OldModelGraph(object):
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
                batch, self.iterator_init_op,self.num_train_batch, self.num_val_batch = get_batch_prepro(
                        self.tfrecord_path,params,self.logger
                    ) 
                self.tokens, self.Y, self.Z, self.Y_mask = batch['indexes'], batch['mels'], batch['mags'], batch['mels_mask']
                self.Y_stop_labels = 1 - self.Y_mask[:,:,0] # 0 when data exists, 1 when padded
            else:
                self.tokens, self.Y, self.Z, self.fnames, self.num_train_batch = get_batch(params,mode,self.logger)
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
            self.target, self.pred, self.logit, self._tboard_label = self.Z, self.Zhat, self.Zlogit, 'train/Z'
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
            # self.last_attended = tf.placeholder(dtype=tf.int32,shape=[self.params.batch_size]) # batch_size with int indexes 
            self.tokens = tf.placeholder(dtype=tf.int32,shape=[None,None]) # int encoded input text to synthesize

        self.add_input_embeddings(reuse)
        self.logger.info('Initialized input character embeddings with dim: {}'.format(self.L.shape))

        if self.params.pos_encoding:
            self.add_pos_encodings()
            self.logger.info('Initialized position encodings, shapes: {}, {}'.format(self.K_pos.shape,self.Q_pos.shape))

        self.add_predict_op(reuse)
        # self.target, self.pred, self.logit, self.sumlabel = self.Y, self.Yhat, self.Ylogit, 'train/Y'
        # tf.summary.image('train/A', tf.expand_dims(tf.transpose(self.A[:1], [0, 2, 1]), -1))
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
            vocab_size = self.dataloader.vocab_size 
            # TODO: Add ability to make padding embedding zero
            embedding_mat = tf.get_variable(name='char_embeddings',shape=[vocab_size,e],
                dtype=tf.float32,trainable=True)
            self.L = tf.nn.embedding_lookup(embedding_mat,self.tokens)

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
            # self.A , self.R = AttentionBlock(self.K, self.V, self.Q,
            #     last_attended=self.last_attended,attn_window_size=self.params.attn_window_size)
            self.A , self.R = AttentionBlock(self.K, self.V, self.Q)
        else:
            self.A , self.R = AttentionBlock(self.K, self.V, self.Q)
        self.logger.info('Encoded R with dim: {}'.format(self.R.shape))
        self.RQ = tf.concat([self.R, self.Q],axis=2)
        self.logger.info('Concatenated RQ with dim: {}'.format(self.RQ.shape))
        self.Ylogit, self.Yhat, self.YStoplogit = AudioDecBlock(self.RQ,self.params.d,self.params.F)
        self.logger.info('Decoded Yhat with dim: {}'.format(self.Yhat.shape))
    
        return self.Ylogit, self.Yhat
    
    def add_loss_op(self):
    
        # compute loss (without guided attention loss for now)
        # TODO: fix the use of Y_mask (won't work for SSRN currently)
        with tf.variable_scope('LossOps'):
            self.L1_loss = tf.reduce_sum(
                    tf.abs(self.target-self.pred)*self.Y_mask
                    )/tf.reduce_sum(self.Y_mask) # padded batches are masked
            assert len(self.L1_loss.shape.as_list())==0,'Loss not scalar, shape: {}'.format(self.L1_loss.shape)
            self.CE_loss = tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target,logits=self.logit)*self.Y_mask
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

        tf.summary.image(self.sumlabel+'_target', tf.expand_dims(tf.transpose(self.target[:1], [0, 2, 1]), -1))
        tf.summary.image(self.sumlabel+'_pred', tf.expand_dims(tf.transpose(self.pred[:1], [0, 2, 1]), -1))
        tf.summary.histogram(self.sumlabel+'_target',self.target)
        tf.summary.histogram(self.sumlabel+'_logit',self.logit)
        tf.summary.histogram(self.sumlabel+'_pred',self.pred)
        tf.summary.scalar('train/L1_loss',self.L1_loss)
        tf.summary.scalar('train/CE_loss',self.CE_loss)
        tf.summary.scalar('train/total_loss',self.loss)
    
        return self.loss, self.L1_loss, self.CE_loss

    def add_train_op(self):

        with tf.variable_scope('optimizer'):
            self.lr = learning_rate_decay(self.params,self.global_step) 
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=self.params.beta1,beta2=self.params.beta2)
    
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

    tf.logging.set_verbosity(tf.logging.DEBUG)
    test_graph_setup('placeholder')
