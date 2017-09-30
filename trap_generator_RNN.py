#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:30:13 2017

@author: frankiezeager

RNN Code to generate trap lyrics based on scraped trap lyric data
input: trap lyrics scraped from AZ Lyrics using the lyric_scraper.py file
output: generated lyrics produced by a recurrent neural network
"""

import pickle
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding,TimeDistributed,Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.utils import np_utils
import math
import string
import random
#import lyrics
lyrics=pickle.load(open('clean_lyrics.p','rb'))

chars = list(set(lyrics))
vocab_size=len(chars)

#create mapping from characters to indexes and indexes to characters
ix_to_char = {ix:char for ix, char in enumerate(chars)}
char_to_ix = {char:ix for ix, char in enumerate(chars)}

#create hyperparameters 
seq_length=100
hidden_dimension=500
layer_num=3
batch_size=50000
generate_length=850

#prep data
X_input=[]
y_input=[]
for i in range(0,math.floor(len(lyrics)/seq_length)): #for i in number of sequences
    #create sequence of lyrics from full raw text
    beg=i*seq_length #so this will start at 0 for i=0
    end=(i+1)*seq_length
    x_sample=lyrics[beg:end]
    y_sample=lyrics[end] #this will be the next character after the x sequence (the true predicted value)    
    X_input.append([char_to_ix[char] for char in x_sample]) #encode for each character in the sequence and append to X
    y_input.append(char_to_ix[y_sample])

#translate features into the form [sample, t, features]
X=np.reshape(X_input,(len(X_input),seq_length,1))
#normalize
X=X/float(vocab_size)
#one hot encoding the data
y=np_utils.to_categorical(y_input)
 
       
def output_text(model, length=500):
"""
Function to output text trained by the neural network. Starts with a randomly selected capital letter.
Input:
    model: fit keras model object
    length: int. how long the output text should be. Default is 500 characters.
Global variables:
    vocab_size: int. How long the vocab size is
"""

    #get random sample to start generating text
    ix=np.random.randint(0,len(X)-1)
    sample=X_input[ix]
    print("\n")
    print("Random Text Seed from corpus: \n")
    print(''.join(ix_to_char[index] for index in sample)) #print all the letters from the sample
    print()
    seed_text=''.join(ix_to_char[index] for index in sample)
    for i in range(length):
        x=np.reshape(sample,(1,len(sample),1)) #get in form [samples,time_step,features]
        x=x/float(vocab_size) #normalize
        pred=model.predict(x,verbose=0) #make prediction for next letter
        index=np.argmax(pred)
        result=ix_to_char[index]
        seed_text=seed_text+result #append to text
    print("Final Output: \n")
    print(seed_text)

#define model
model=Sequential()
model.add(LSTM(hidden_dimension,input_shape=(None,vocab_size), return_sequences=True))
for i in range(layer_num-1):
    model.add(LSTM(hidden_dimension,return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

num_epoch=0

while True:
    print('\n\n')
    model.fit(X, y, batch_size=batch_size, verbose=1, epochs=1)
    num_epoch += 1
    print("Epoch number ", num_epoch)
    output_text(model, generate_length)
    ModelCheckpoint('model_chceckpoint_epoch_'+num_epoch+'.hdf5',save_best_only=True)