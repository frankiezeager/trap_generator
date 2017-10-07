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
from keras.models import Sequential
from keras.layers import Dense, Embedding,TimeDistributed,Activation
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.utils import np_utils
import math
#import lyrics
lyrics=pickle.load(open('clean_lyrics.p','rb'))

chars = sorted(list(set(lyrics)))
vocab_size=len(chars)

#create mapping from characters to indexes and indexes to characters
#ix_to_char = {ix:char for ix, char in enumerate(chars)}
#char_to_ix = {char:ix for ix, char in enumerate(chars)}
char_to_ix = dict((c, i) for i, c in enumerate(chars))
ix_to_char = dict((i, c) for i, c in enumerate(chars))
#create hyperparameters 
seq_length=100
hidden_dimension=256
layer_num=3
batch_size=50000
generate_length=850

#prep data
X=np.zeros((math.floor(len(lyrics)/seq_length),seq_length,vocab_size))
y=np.zeros((math.floor(len(lyrics)/seq_length),seq_length,vocab_size))

#given a sequence X, target sequence will be shifted by one space
for i in range(0,math.floor(len(lyrics)/seq_length)):
    #create sequence of lyrics from full raw text
    x_seq=lyrics[i*seq_length:(i+1)*seq_length]
    y_seq=lyrics[i*seq_length+1:(i+1)*seq_length+1]
    X_ix=[char_to_ix[char] for char in x_seq] #convert each letter in sequence to ix
    y_ix=[char_to_ix[char]for char in y_seq] #do the same for y
    input_sequence=np.zeros((seq_length,vocab_size))
    for j in range(seq_length):
        input_sequence[j][X_ix[j]]=1
    X[i]=input_sequence
    target_sequence=np.zeros((seq_length,vocab_size))
    for j in range(seq_length):
        target_sequence[j][y_ix[j]]=1
    y[i]=target_sequence

#translate features into the form [sample, sequence, features]
#X=np.reshape(X_input,(len(X_input),seq_length,1))
#X=X_input.reshape(1,seq_length,1)
#y=Y_input.reshape
#normalize
#X=X/float(vocab_size)
#one hot encoding the data
#y=np_utils.to_categorical(y_input)
 
       
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

    ix=np.random.randint(0,len(X)-1) #get random integer
    sample=X_input[ix] #get the text from that point
    print("\n")
    print("Random Text Seed from corpus: \n")
    print(''.join(ix_to_char[index] for index in sample)) #print all the letters from the sample
    print()
    seed_text=''.join(ix_to_char[index] for index in sample)
    for i in range(length):
        x=np.reshape(sample,(1,len(sample),1)) #get in form [samples,sequence,features]
        x=x/float(vocab_size) #normalize
        pred=model.predict(x)[0] #make prediction for next letter
        index=np.argmax(pred,axis=1)[-1]
        result=ix_to_char[index]
        seed_text=seed_text+result#append to text
        sample.append(index)#add to the sample 
        sample=sample[1:] #removing first character to keep consistent length
        
    print("Final Output: \n")
    print(seed_text)
    return seed_text
vocab_size=len(chars)
#define model
model=Sequential()
model.add(LSTM(hidden_dimension,input_shape=(None,vocab_size), return_sequences=True))
for i in range(layer_num-1):
    model.add(LSTM(hidden_dimension,return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size))) #add if you want to try many to many output (instead of character by character)
#model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam")


#see how bad the original model is
print("/n initial text output: ")
output_text(model)

num_epoch=0

text_file=open('text_file.txt','w')
text_file.close()
while True:
    print('\n\n')
    model.fit(X, y, batch_size=batch_size, verbose=1)
    num_epoch += 1
    print("Epoch number ", num_epoch)
    predicted_text=output_text(model, generate_length)
    text_file=open('text_file.txt','a')
    text_file.write('epoch number '+ num_epoch+ '\n\n\n' + predicted_text)
    text_file.close()
    #pickle.dump(predicted_text,open('text_'+str(num_epoch)+'.p','wb'))
    ModelCheckpoint('model_chceckpoint_epoch_'+str(num_epoch)+'.hdf5',save_best_only=True)