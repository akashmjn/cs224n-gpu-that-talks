import sys, os
sys.path.append(os.path.abspath("."))
import tensorflow as tf
import sentencepiece as spm
from src.data_load import TFRecordDataloader, load_tokens_from_text
from src.utils import Params, set_logger

def test_parse_tfrecord(dl,params):

    print("\n #### Testing: parse_tfrecord #### \n")
    print(dl.spm_model)
    padded_shapes = (
            tf.TensorShape([None]),
            tf.TensorShape([None,params.F]),
            tf.TensorShape([None,params.Fo]),
            tf.TensorShape([None,params.F])
        )   

    dataset = tf.data.TFRecordDataset([dl.tfrecord_path])\
                    .map(dl.parse_tfrecord,params.num_threads)\
                    .padded_batch(4,padded_shapes)
    it = dataset.make_one_shot_iterator()
    tokens, mel, mag, mel_mask = it.get_next()

    print(tokens)
    print(mel)
    print(mel_mask)

    with tf.Session() as sess:
        t, m, ms = sess.run([tokens,mel,mel_mask])   

    print(t.shape,m.shape,ms.shape)
    print_token_array(t,dl)

def print_token_array(token_array,dl):
    for tis in token_array:
        print(" ".join([ dl.spm_model.id_to_piece(int(ti)) for ti in tis ]))    

def test_get_batch(dl):

    print("\n #### Testing: get_batch #### \n")
    logger = set_logger("./test.log")
    batch  = dl.get_batch(logger)

    mel, mel_mask = batch['mels'], batch['mels_mask']
    s1, s2 = tf.reduce_mean(mel), tf.reduce_sum(mel*mel_mask)/tf.reduce_sum(mel_mask)

    with tf.Session() as sess:
        sess.run(dl.iterator_init_op)
        batch_dict = sess.run(batch)
        tokens = batch_dict['tokens']
        print("Tokens, shape: {}".format(tokens.shape))
        print_token_array(tokens,dl)
        print("Mean: {}, with masking: {}".format(*sess.run([s1,s2])))

# test functions
if __name__ == '__main__':
    tfrecord_path = "../data/LJSpeech-1-1/train.tfrecord"
    params = Params("./runs/text2mel/test/params.json")
    params.dict['spm_model'] = "../data/LJSpeech-1-1/spm250unnorm.model"
    params.dict['test_data'] = "./sentences/harvard_sentences.txt"

    dl = TFRecordDataloader(params,tfrecord_path)

    test_parse_tfrecord(dl,params)
    test_get_batch(dl)
    #print(load_tokens_from_text(params))
