#!/usr/local/bin/python3

import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
from model import TextEncBlock, AudioEncBlock, AudioDecBlock, AttentionBlock


def add_predict_op(L,S,d,F,reuse=False):

    # building training graph for Text2Mel
    print('Building training graph for Text2Mel')
    KV = TextEncBlock(L,d)
    print('Encoded KV with dim: {}'.format(KV.shape))
    Q = AudioEncBlock(S,d)
    print('Encoded Q with dim: {}'.format(Q.shape))
    _ , R = AttentionBlock(KV,Q)
    print('Encoded R with dim: {}'.format(R.shape))
    RQ = tf.concat([R,Q],axis=2)
    print('Concatendated RQ with dim: {}'.format(RQ.shape))
    Ylogit, Yhat = AudioDecBlock(RQ,F)
    print('Decoded Yhat with dim: {}'.format(Yhat.shape))

    return Ylogit, Yhat

def add_loss_op(Y,Ylogit,Yhat):

    # compute loss (without guided attention loss for now)
    L1_loss = tf.reduce_mean(tf.abs(Y-Yhat))
    assert len(L1_loss.shape.as_list())==0,'Loss not scalar, shape: {}'.format(L1_loss.shape)
    CE_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=Ylogit))
    assert len(CE_loss.shape.as_list())==0,'Loss not scalar, shape: {}'.format(CE_loss.shape)
    loss = L1_loss + CE_loss

    return loss, L1_loss, CE_loss


if __name__ == '__main__':

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
    loss, L1_loss, CE_loss = add_loss_op(Y,Ylogit,Yhat)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_out, L1_out, CE_out, Y_out = sess.run([loss, L1_loss, CE_loss, Yhat],{L:L_feed,Y:Y_feed})
    
    print('Final loss: {:.2f}, L1: {:.2f}, CE: {:.2f}, with output shape: {}'.format(loss_out, L1_out, CE_out, Y_out.shape))
    np.set_printoptions(precision=3)
    print(Y_out)
