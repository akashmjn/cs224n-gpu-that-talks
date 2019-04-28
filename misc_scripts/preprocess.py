import os,sys
import argparse
import numpy as np
import tensorflow as tf

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from src.dsp_utils import load_spectrograms
from src.utils import Params
from src.data_load import process_csv_file

def process_audio_pair(fpath,params,output_path):
    # fpath,params,output_path = arg_tuple
    fname, mel, mag = load_spectrograms(fpath,params,'train_text2mel')
    np.save(os.path.join(output_path,"mels",fname.replace(".wav",".npy")),mel,allow_pickle=False)
    np.save(os.path.join(output_path,"mags",fname.replace(".wav",".npy")),mag,allow_pickle=False)      
    print("Processed: {}".format(fpath))

def process_to_npy(params,input_path,csv_path,output_path):
    # TODO: Parallelize / multiprocess this
    
    # Processes all wav files in csv and saves extracted features as npy arrays
    # get list of file paths
    with open(csv_path) as f:
        lines = f.readlines()
    fpaths = [ os.path.join(input_path,line.split("|")[0]+".wav") for line in lines]
    # args_list = [ (fpath,Params(args.params_path),args.output_path) for fpath in fpaths]
    print(fpaths[:20])

    os.makedirs(os.path.join(output_path,"mels"),exist_ok=True)
    os.makedirs(os.path.join(output_path,"mags"),exist_ok=True)

    for fpath in fpaths:
        process_audio_pair(fpath,params,output_path)

    # with mp.Pool(4) as pool:
    #     res = pool.imap_unordered(process_audio_pair,args_list)
    #     pool.close()
    #     pool.join()   

# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))   

def process_to_tfrecord(params,input_path,csv_path,output_path): #TODO:SP
    # TODO: Parallelize / multiprocess this

    fpaths, text_lengths, transcriptions = process_csv_file(csv_path,params)
    # fpaths = [os.path.join(input_path,f) for f in fpaths]

    if 'train' in csv_path:
        tfrecord_fname, mode = os.path.join(output_path,'train.tfrecord'), 'train'
    elif 'val' in csv_path:
        tfrecord_fname, mode = os.path.join(output_path,'val.tfrecord'), 'val'
    else:
        raise Warning('Check whether passed csv file corresponds to training or validation sets')

    writer = tf.python_io.TFRecordWriter(tfrecord_fname)

    for i,fpath in enumerate(fpaths):

        fname, mel, mag = load_spectrograms(fpath,params,'train_text2mel')
        text = transcriptions[i]
        print(text)

        feature = {
            'fname': _bytes_feature(fname.encode()),
            'transcription': _bytes_feature(text.encode()),
            'mel': _bytes_feature(mel.tostring()),
            'mag': _bytes_feature(mag.tostring()),
            'input-len': _int64_feature(text_lengths[i]),
            'mel-shape': _int64_list_feature(list(mel.shape)),
            'mag-shape': _int64_list_feature(list(mag.shape))
        } 

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())
        print("Processed: {}".format(fname))

    writer.close()
    sys.stdout.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('params_path', help="Path to params.json file containing DSP hyperparameters")
    parser.add_argument('input_path', help="Path to folder containing .wav files")
    parser.add_argument('csv_path', help="Path to file with metadata: text, wav filename")
    parser.add_argument('output_path', help="Path to output folder that will contain mels, mags folders")
    parser.add_argument('--mode',default='tfrecord',help="Format to save processed data files npy/tfrecord (default)")
    args = parser.parse_args()

    params, output_path = Params(args.params_path),args.output_path

    if args.mode=='npy':
        process_to_npy(params,args.input_path,args.csv_path,args.output_path)
    elif args.mode=='tfrecord':
        process_to_tfrecord(params,args.input_path,args.csv_path,args.output_path)
