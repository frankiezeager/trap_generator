#%matplotlib inline
import pickle
from gensim.models import Word2Vec
import numpy as np
from keras.utils import np_utils
import math
import os
import timeit
import seaborn as sns
import collections
import pandas as pd
from keras.preprocessing import sequence
#import more_itertools as mit
import re
import random
#lines = [line for line in lyrics.split('\n') if line!='']
#len(lines)
#2156871/272010 #about 8 words per line
#if i want 3 lines per sequence, about 24 words
random.seed(12345)
#import lyrics
lyrics=pickle.load(open('clean_lyrics.p','rb'))

words = lyrics.split()
print('number of words in the corpus: {:,}'.format(len(words)))
#>2,000,000 words, but only 35,000 unique words
print('number of unique words in the corpus: {:,}'.format(len(set(words))))

#>1,000,000 words, but only 38,000 unique words

#most common words
word_freq =collections.Counter(words)
word_freq = pd.DataFrame.from_dict(word_freq,orient='index')
word_freq = word_freq.rename(columns={0:'count'})
word_freq.head(10)
word_freq.sort_values(by='count',ascending=False).head(20)

#not too different from most english corpuses except for the curse words

#make word a column not the index
word_freq['word']=word_freq.index
word_freq=word_freq.reset_index(drop=True)
word_freq.head()

sns.set(color_codes=True)

sns.barplot(x='word',y='count', data=word_freq.sort_values(by='count',ascending=False).head(10))

list_lyrics = pickle.load(open('clean_list_lyrics.p','rb'))


#Word2Vec Embedding
len(list_lyrics) #4584 songs
list_lyrics[0:5]
max_seq_length = 10
#create lines/sentences
#lines = lyrics.split('\n')
#split into list of words for each line if the line is not empty, making sure
#it's less than max length
#lines = [line.split()[:max_seq_length] for line in lines if len(line.split())!=0]
sequences=[]
for lyric in list_lyrics:
    #add spaces around all new line characters
    lyric = re.sub(r'\n([a-zA-Z])',r'\n \1',lyric)
    lyric = re.sub(r'([a-zA-Z])\n',r'\1 \n',lyric)
    #deal with multiple new line characters, but keeping single new line characters
    lyric = re.sub('\n\n','\n',lyric)
    lyric = re.sub('\n\n\n','\n',lyric)
    lyric = re.sub('\n\n\n\n','\n',lyric)
    lyric = re.sub('\n\n\n\n\n','\n',lyric)
    #remove return characters
    lyric = re.sub('\r',' ',lyric)

    #split into words
    words = lyric.split(' ')

    #split into sequences
    lines = [words[i:i+max_seq_length] for i in range(0,len(words), max_seq_length)]
    sequences.append(lines)

len(sequences)

#now we have a list of each song, with a list containing the words in the song, split by sequence seq_length

#unnest the list
lines = [s for sequence in sequences for s in sequence]

print('Number of examples: ', len(lines))

w2vmodel = Word2Vec(lines,min_count=1,iter=200)

w2vmodel.wv.most_similar('dollar')

w2vmodel.wv.most_similar('\n')


w2vmodel.wv.most_similar(positive=['girl','homie'], negative=['man'])
w2vmodel.wv.most_similar(positive=['lady','homie'], negative=['man'])
w2vmodel.wv['homie']

w2v_weights = w2vmodel.wv.syn0

vocab_size, embedding_size = w2v_weights.shape
print('vocab size: ', vocab_size)
print('embedding size: ', embedding_size)

#add a token for unseen and words used for padding to max sequence length (the model will learn to ignore these)
w2vmodel.build_vocab([['_UNSEEN_','_UNSEEN_']], update=True)

w2vmodel.wv.syn0[w2vmodel.wv.vocab['_UNSEEN_'].index] = vocab_size+1

w2vmodel.save('word2vec_model')
