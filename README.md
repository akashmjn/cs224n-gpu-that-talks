## Repository for CS224n project: Attention, I'm Trying to Speak. 

Implementation of a convolutional seq2seq-based model based on [Tachibana et. al. (2017)](https://arxiv.org/abs/1710.08969). 
Given a sequence of characters, the model predicts a sequence of spectrogram frames in two stages (Text2Mel and SSRN). 

As discussed in the report, we can get fairly decent audio quality with Text2Mel trained for 60k steps, SSRN for 100k steps. This corresponds to about (10+20) hours of training on a single Tesla M60 GPU. 

**Poster**: [[link]](https://akashmjn.github.io/cs224n/cs224n-final-poster.pdf))
**Final Report**: [[link]](https://akashmjn.github.io/cs224n/cs224n-final-project-report.pdf)
**Output Samples (M4 model - 60k steps)**: [[link]](https://soundcloud.com/akashmjn/sets/m4-tuned-model)

![Model Schematic](https://raw.githubusercontent.com/akashmjn/cs224n-gpu-that-talks/master/reports/model-schematic.png)

## Usage:

### Script files

Run each file with `python script_file.py -h` to see usage details. 

```
python train.py <PATH_PARAMS.JSON> <MODE>
python evaluate.py <PATH_PARAMS.JSON> <MODE> 
python synthesize.py <TEXT2MEL_PARAMS> <SSRN_PARAMS> <SENTENCES.txt> (<N_ITER> <SAMPLE_DIR>)
```

### Notebooks:

*   **Evaluation**: Runs model predictions across the entire training and validation sets for different saved model checkpoints and saves the final results. 
*   **Demo**: Interactively type input sentences and listen to the generated output audio. 
