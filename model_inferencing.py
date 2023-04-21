# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 10:57:53 2023

@author: dreji18
"""

from huggingsound import SpeechRecognitionModel

model_dir = r'xx\xxx\xxx'

model = SpeechRecognitionModel(model_dir)

audio_dir =  r'xx\xxx\xxx'

import os
os.chdir(audio_dir)

audio_paths = ["F02_B1_C17_M6.wav"]

transcriptions = model.transcribe(audio_paths)
