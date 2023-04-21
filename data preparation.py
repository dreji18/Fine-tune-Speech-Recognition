# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 18:44:57 2023

@author: dreji18
"""

from datasets import load_dataset
import numpy as np
import wave
import os

path = r'xx\xxx\xxx'
os.chdir(path)

# loading the dataset from huggingface hub
#--https://huggingface.co/datasets/ngdiana/uaspeech_severity_high
UA = load_dataset("ngdiana/uaspeech_severity_high")

# converting to pandas for easy data handling
UA_df =  UA['train'].to_pandas()
UA_df = UA_df[0:20]

UA_df['filename'] = UA_df['path'].apply(lambda x: x.split("/")[-1])

UA_df.to_csv("UA_df.csv")


## the goal is to convert the speech array to WAV file in bulk

def array2WAV(id):
    
    # Define the sample rate and number of samples
    sample_rate = 16000
    num_samples = 1
    
    # Create a WAV file object
    wav_file = wave.open(UA_df['filename'].iloc[id], "w")
    
    # Set the WAV file parameters
    wav_file.setnchannels(1) # 1 channel (mono)
    wav_file.setsampwidth(2) # 16-bit sample width
    wav_file.setframerate(sample_rate)
    
    # Write the samples to the WAV file as binary data
    
    samples = UA_df['speech'].iloc[id]
    samples = (samples * (2**15 - 1)).astype(np.int16)
    wav_file.writeframes(samples.tobytes())
    
    # Close the WAV file
    wav_file.close()

# set the output directory to save the wav files
out_dir = r'xx\xxx\xxx'

os.chdir(out_dir)

for id in range(0, len(UA_df)):
    array2WAV(id)