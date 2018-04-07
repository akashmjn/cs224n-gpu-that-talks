## Repository for CS224n project: Attention, I'm Trying to Speak. 

Implementation of a convolutional seq2seq-based model based on [Tachibana et. al. (2017)](https://arxiv.org/abs/1710.08969). 
Given a sequence of characters, the model predicts a sequence of spectrogram frames in two stages (Text2Mel and SSRN). 

![Model Schematic](https://raw.githubusercontent.com/akashmjn/cs224n-gpu-that-talks/master/reports/model-schematic.png)

https://soundcloud.com/akashmjn/sets/m4-tuned-model

## Usage:

### Script files

Run each file with `python script_file.py -h` to see usage details. 

```
python train.py <PATH_PARAMS.JSON> <MODE>
python evaluate.py <PATH_PARAMS.JSON> <MODE> 
python synthesize.py <TEXT2MEL_PARAMS> <SSRN_PARAMS> <SENTENCES.txt> (<N_ITER> <SAMPLE_DIR>)
```

### Notebooks:

*   Evaluation: 
*   Demo
