"""
Implementation of CS224n project: Attention, I'm Trying to Speak. 
Speech synthesis using convolutional seq2seq model based on Tachibana et. al (2017)

Utility Code has been referenced from the following sources, all other code is the author's own: 
    - data_load.py, dsp_utils.py (with modifications)
      https://www.github.com/kyubyong/dc_tts, (Author: kyubyong park, kbpark.linguist@gmail.com)
      https://github.com/r9y9/deepvoice3_pytorch/blob/master/audio.py
    - spsi.py (referenced)
      https://github.com/lonce/SPSI_Python
    - utils.py (referenced)
      https://github.com/cs230-stanford/cs230-code-examples
      https://www.github.com/kyubyong/dc_tts
      https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py     

Author: 
    Akash Mahajan
Email:
    akashmjn@stanford.edu
"""

__author__ = "Akash Mahajan"

from . import utils 
from . import dsp_utils
from . import data_load
from . import graph 
from . import model 
from . import spsi 

__all__ = ["utils","dsp_utils","data_load","graph","model","spsi"]
