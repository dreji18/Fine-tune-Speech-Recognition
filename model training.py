# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 20:01:49 2023

@author: dreji18
"""

from huggingsound import TrainingArguments, ModelArguments, SpeechRecognitionModel, TokenSet
#https://github.com/jonatasgrosman/huggingsound
#!pip install huggingsound
import pandas as pd
import os


audio_dir = r'xx\xxx\xxx'

import torch
torch.cuda.empty_cache()
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
#model = SpeechRecognitionModel("facebook/wav2vec2-large-xlsr-53")
model = SpeechRecognitionModel("facebook/wav2vec2-large-xlsr-53", device=device)
torch.cuda.empty_cache()

# preparing the vocab file
vocab_dict = {'c': 0,
 'q': 1,
 'w': 2,
 'j': 3,
 'r': 4,
 'h': 5,
 'x': 6,
 'm': 7,
 'p': 8,
 'd': 9,
 'f': 10,
 'g': 11,
 'k': 12,
 'u': 13,
 'v': 14,
 'a': 15,
 'n': 16,
 ' ': 17,
 'i': 18,
 's': 19,
 'y': 20,
 'l': 21,
 'e': 22,
 'o': 23,
 'z': 24,
 'b': 25,
 't': 26}

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

tokens = list(vocab_dict.keys())
token_set = TokenSet(tokens)

#%%
training_args = TrainingArguments(
    learning_rate=3e-4,
    max_steps=1000,
    eval_steps=200,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
)
model_args = ModelArguments(
    activation_dropout=0.1,
    hidden_dropout=0.1,
) 

#%%
# preparing the training data
UA_df = pd.read_csv(r'xx\xxx\UA_df.csv')

os.chdir(audio_dir)

train_data = []
for id in range(0, len(UA_df)):
    train_data.append({"path": UA_df['filename'].iloc[id], "transcription": UA_df['target'].iloc[id]})

# for evaluation data    
UA_df[['col1', 'col2', 'col3', 'col4']] = UA_df['filename'].str.split("_", expand=True)
UA_df1 = UA_df.drop_duplicates(subset='col3', keep="first")

eval_data = []
for id in range(0, len(UA_df1)):
    eval_data.append({"path": UA_df1['filename'].iloc[id], "transcription": UA_df1['target'].iloc[id]})

# and finally, fine-tune your model
output_dir = r'xx\xxx\xxx'
model.finetune(
    output_dir, 
    train_data=train_data, 
    eval_data=eval_data, # the eval_data is optional
    token_set=token_set,
    training_args=training_args,
    model_args=model_args,
)
#%%