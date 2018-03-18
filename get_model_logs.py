#!/usr/bin/env python3

import os,sys,glob

# list of all model checkpoint dirs
ckp_dirs = sys.argv[1:]

# create list of files from each dir to compress 
file_list = []
for ckp_dir in ckp_dirs:
    common_files = ['graph.pbtxt','params.json','checkpoint']
    file_list.extend([os.path.join(ckp_dir,f) for f in common_files])
    file_list.extend(glob.glob(os.path.join(ckp_dir,'*.log')))
    model_files = [ os.path.basename(f) for f in glob.glob(os.path.join(ckp_dir,'model_gs_*')) ]
    ckp_nums = [ int(f.split('gs_')[1].split('k.')[0]) for f in model_files ]
    max_ckp_str = str(max(ckp_nums))
    model_files = [ f for f in model_files if max_ckp_str in f ]
    file_list.extend([os.path.join(ckp_dir,f) for f in model_files])

# compress all files
print("Compressing files to logs.tar.gz")
os.system('tar -zcvf logs.tar.gz '+" ".join(file_list))

