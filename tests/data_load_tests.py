import sys, os
sys.path.append(os.path.abspath("."))
import tensorflow as tf
from src.data_load import parse_tfrecord, get_batch_prepro
from src.utils import Params, set_logger

def test_parse_tfrecord(tfrecord_path,params):

    padded_shapes = (
            tf.TensorShape([None]),
            tf.TensorShape([None,params.F]),
            tf.TensorShape([None,params.Fo]),
            tf.TensorShape([None,params.F])
        )   

    dataset = tf.data.TFRecordDataset([tfrecord_path])\
                    .map(parse_tfrecord,params.num_threads)\
                    .padded_batch(4,padded_shapes)
    it = dataset.make_one_shot_iterator()
    idx, mel, mag, mel_mask = it.get_next()

    print(idx)
    print(mel)
    print(mel_mask)

    with tf.Session() as sess:
        print(sess.run([idx,mel,mel_mask]))   

def test_get_batch_prepro(tfrecord_path,params):

    logger = set_logger("./test.log")
    batch, init_op, nb_train, nb_val = get_batch_prepro(tfrecord_path,params,logger)

    mel, mel_mask = batch['mels'], batch['mels_mask']
    s1, s2 = tf.reduce_mean(mel), tf.reduce_sum(mel*mel_mask)/tf.reduce_sum(mel_mask)

    with tf.Session() as sess:
        sess.run(init_op)
        batch_dict = sess.run(batch)
        print("Mean: {}, with masking: {}".format(*sess.run([s1,s2])))

# test functions
if __name__ == '__main__':
    tfrecord_path = "../data/indic-tts-hindi/hindi-female/train.tfrecord"
    params = Params("./runs/hindi-text2melM4/params.json")

    # test_parse_tfrecord(tfrecord_path,params)
    test_get_batch_prepro(tfrecord_path,params)
