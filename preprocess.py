import os,sys
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from src.dsp_utils import load_spectrograms
from src.utils import Params

# Input dir, output dir
def process_audio_pair(fpath,params,output_path):
    # fpath,params,output_path = arg_tuple
    fname, mel, mag = load_spectrograms(fpath,params,'train_text2mel')
    np.save(os.path.join(output_path,"mels",fname.replace(".wav",".npy")),mel,allow_pickle=False)
    np.save(os.path.join(output_path,"mags",fname.replace(".wav",".npy")),mag,allow_pickle=False)      
    print("Processed: {}".format(fpath))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('params_path', help="Path to params.json file containing DSP hyperparameters")
    parser.add_argument('input_path', help="Path to folder containing .wav files")
    parser.add_argument('csv_path', help="Path to file with metadata: text, wav filename")
    parser.add_argument('output_path', help="Path to output folder that will contain mels, mags folders")
    args = parser.parse_args()

    params, output_path = Params(args.params_path),args.output_path

    # get list of file paths
    with open(args.csv_path) as f:
        lines = f.readlines()
    fpaths = [ os.path.join(args.input_path,line.split("|")[0]+".wav") for line in lines]
    # args_list = [ (fpath,Params(args.params_path),args.output_path) for fpath in fpaths]
    print(fpaths[:20])

    os.makedirs(os.path.join(args.output_path,"mels"),exist_ok=True)
    os.makedirs(os.path.join(args.output_path,"mags"),exist_ok=True)

    for fpath in fpaths:
        process_audio_pair(fpath,params,output_path)

    # with mp.Pool(4) as pool:
    #     res = pool.imap_unordered(process_audio_pair,args_list)
    #     pool.close()
    #     pool.join()

