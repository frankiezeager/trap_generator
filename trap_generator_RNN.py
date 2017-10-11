#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:30:13 2017
@author: frankiezeager
RNN Code to generate trap lyrics based on scraped trap lyric data
input: trap lyrics scraped from AZ Lyrics using the lyric_scraper.py file
output: generated lyrics produced by a recurrent neural network
Thanks to this tutorial for a lot of help along the way: https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/
"""
import pickle
from keras.models import Sequential
from keras.layers import Dense, Embedding,TimeDistributed,Activation
from keras.layers import LSTM
from keras.callbacks import History,Callback
import numpy as np
from keras.utils import np_utils
import math

#import lyrics
lyrics=pickle.load(open('clean_lyrics.p','rb'))

#split into train and test (validation) to prevent overfitting to training set ~30000 training examples (characters)
lyrics_train=lyrics[:-100000]
end=len(lyrics)
lyrics_test=lyrics[end-100000:]

chars = sorted(list(set(lyrics)))
vocab_size=len(chars)

#create mapping from characters to indexes and indexes to characters
char_to_ix = dict((c, i) for i, c in enumerate(chars))
ix_to_char = dict((i, c) for i, c in enumerate(chars))

#create hyperparameters 
seq_length=40
hidden_dimension=64
layer_num=3
batch_size=2000
generate_length=850

#prep data so as to run faster
X=np.zeros((math.floor(len(lyrics_train)/seq_length),seq_length,vocab_size))
y=np.zeros((math.floor(len(lyrics_train)/seq_length),seq_length,vocab_size))
X_test=np.zeros((math.floor(len(lyrics_test)/seq_length),seq_length,vocab_size))
y_test=np.zeros((math.floor(len(lyrics_test)/seq_length),seq_length,vocab_size))

#given a sequence X, target sequence will be shifted by one space
for i in range(0,math.floor(len(lyrics_train)/seq_length)):
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
    
#create test data in same way
for i in range(0,math.floor(len(lyrics_test)/seq_length)):
    #create sequence of lyrics from full raw text
    x_test_seq=lyrics[i*seq_length:(i+1)*seq_length]
    y_test_seq=lyrics[i*seq_length+1:(i+1)*seq_length+1]
    X_test_ix=[char_to_ix[char] for char in x_test_seq] #convert each letter in sequence to ix
    y_test_ix=[char_to_ix[char]for char in y_test_seq] #do the same for y
    input_sequence=np.zeros((seq_length,vocab_size))
    for j in range(seq_length):
        input_sequence[j][X_ix[j]]=1
    X_test[i]=input_sequence
    target_sequence=np.zeros((seq_length,vocab_size))
    for j in range(seq_length):
        target_sequence[j][y_ix[j]]=1
    y_test[i]=target_sequence
    
      
def output_text(model, length=500,seed="Rain drop (drip), drop top"):
    """
    Function to output text trained by the neural network. Starts with a randomly selected capital letter.
    Input:
        model: fit keras model object
        length: int. how long the output text should be. Default is 500 characters.
        seed: the beginning of the generated lyrics. Set to be Migos' viral phrase "rain drop, drop top",
            but will eventually be user input when hosted
    Global variables:
        vocab_size: int. How long the vocab size is
        seq_length: int. Input size for model
    """
    generated=''
    if len(seed)>seq_length:
        sequence=seed[:-seq_length] #truncate to input size
    if len(seed)<seq_length:
        #pad to input size
        sequence=seed.rjust(seq_length)
    else:
        sequence=seed       
    generated+=seed
    
    for i in range(length):
        inputs=np.zeros((1,seq_length,vocab_size)) #initialize matrix
        #input_ix=[char_to_ix[char] for char in sequence]
        for t,char in enumerate(sequence):
            inputs[0,t,char_to_ix[char]]=1 #one hot encoding
  
        #predicted=np.argmax(model.predict(inputs[:,:i+1:],verbose=0)[0]) #predict next character using max softmax probability
        predicted=np.argmax(model.predict(inputs[:, :i+1, :])[0], 1)
        predicted_letter=ix_to_char[predicted[-1]]
        generated+=predicted_letter #add predicted letter to the generated output
        
        #use only the last sequence_length number of letters as input back into model to predict
        #next characters
        sequence=sequence[1:]+predicted_letter
    return generated





#define model
model=Sequential()
model.add(LSTM(hidden_dimension,input_shape=(None,vocab_size), return_sequences=True))
for i in range(layer_num-1):
    model.add(LSTM(hidden_dimension,return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")


#see how bad the original model is
print("/n initial text output: ")
output_text(model)

num_epoch=0

#define history loss function
class LossHistory(Callback):
    def on_train_begin(self,logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
history=LossHistory()

#write initial text file where output will be appended
text_file=open('text_file.txt','w')
text_file.close()
while True:
    print('\n\n')
    #fit LSTM Model
    model.fit(X, y, batch_size=batch_size, verbose=2,nb_epoch=1,callbacks=[history],validation_data=(X_test,y_test))
    num_epoch += 1
    print("Epoch number ", num_epoch)
    predicted_text=output_text(model, generate_length)
    text_file=open('text_file.txt','a')
    text_file.write('\n\n epoch number '+ str(num_epoch)+ '\n'+'loss: '+str(history.losses)+'\n'+ 'predicted text: '+'\n'+ predicted_text)
    text_file.close()
    #print predicted output so as to measure effectiveness of model
    print("predicted text: ", predicted_text)
    #for every 20th epoch, save the model (to save hd space)
    if num_epoch%20==0:
        model.save('model_checkpoint_epoch_'+str(num_epoch)+'.hdf5')
