import sys,os
sys.path.append(os.path.abspath("."))
import argparse
import tensorflow as tf
from src.utils import Params
from src.graph import ModelGraph
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint_dir', help="Path to directory with checkpoints to modify")
parser.add_argument('--restore_scope', help="Variable scope of variables to restore")
parser.add_argument('--exclude_scope', help="Variable scope of new variables to exclude in first restore,\
 that are then randomly initialized and saved")
parser.add_argument('--restore_dir',help="Path to directory to restore from")
args = parser.parse_args()

checkpoint_path = tf.train.latest_checkpoint(args.checkpoint_dir)
restore_path = tf.train.latest_checkpoint(args.restore_dir) if args.restore_dir else checkpoint_path
params = Params(os.path.join(args.checkpoint_dir,'params.json'))
gs = tf.train.get_or_create_global_step()
g = ModelGraph(params,'train_text2mel')

save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

pdb.set_trace()
restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,args.restore_scope)
exclude_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,args.exclude_scope)
restore_vars = [var for var in restore_vars if var not in exclude_vars]

restorer = tf.train.Saver(restore_vars)
saver = tf.train.Saver(save_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    restorer.restore(sess,restore_path) # restores only variables outside of excluded scope
    saver.save(sess,checkpoint_path) # saves all variables, with random initialization for excluded scope
