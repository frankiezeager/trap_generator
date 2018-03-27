#trap generator training
import pickle
import time
from keras.models import Sequential
from keras.layers import Dense, Embedding, TimeDistributed, Activation, Dropout, Lambda
from keras.layers import LSTM
from keras.callbacks import History, Callback, TensorBoard, LambdaCallback
from keras.optimizers import RMSprop
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
import more_itertools as mit
import re
from gensim.models import Word2Vec
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import timeit

max_seq_length = 6
num_layers = 2
batch_size = 64
epochs=150
dropout=0.05
layer_size = 512

def lstm_model(num_layers, dropout, layer_size, w2v_weights, max_seq_length):
    vocab_size, embedding_size = w2v_weights.shape
    ## create model
    optimizer=optimizers.Adam()
    model=Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=max_seq_length,weights=[w2v_weights],trainable=False))
    for layer in range(num_layers-1):
        model.add(LSTM(layer_size, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(LSTM(layer_size))
    model.add(Dropout(dropout))
    model.add(Dense(vocab_size, activation = 'softmax'))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return model

def word_to_ix(word,w2vmodel):
    return w2vmodel.wv.vocab[word].index

def ix_to_word(ix,w2vmodel):
    return w2vmodel.wv.index2word[ix]

def generate_text(model, w2vmodel, nb_epoch, length=75, max_seq_length=20, seed="Rain drop drop top"):
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

        ADD additional loop for lines (basically change seed each time)
    """
    global sample
    generated = ''
    sequences = seed

    generated += seed

    #clean seed
    seed=re.sub(r'<[^<]+?>', '', seed)
    #remove encoding characters like \x86
    seed=re.sub(r'[^\x00-\x7f]','',seed)
    seed=re.sub(r'\#','',seed)
    #remove punctuation
    seed=re.sub(r'[^A-Za-z0-9\s]','',seed)

    #shorten if longer than max_seq_length
    seed = seed.split(' ')[:max_seq_length]

    word_ix_list = []
    for word in seed:
        try:
            word = word_to_ix(word,w2vmodel)
        except:
            #since we're using -1 as a null word (why we also pad with the not in vocab index), we'll use that for words that aren't in the word2vec model
            print('Warning: {0} not contained in training vocabulary. It will be ignored when computing output.'.format(word))
            word = word_to_ix('_UNSEEN_',w2vmodel)
        word_ix_list.append(word)

    #pad word_list with the unseen word2vec if shorter than max_seq_length
    word_ix_list = [word_to_ix('_UNSEEN_',w2vmodel)] * (max_seq_length-len(word_ix_list)) + word_ix_list

    for temp in [0.2, 0.5, .75, 1.0]:
        print('temperature: ', temp)
        for word in range(length):
            #reshape wordlist
            word_ix_list = np.asarray(word_ix_list).reshape(1,max_seq_length)

            #prediction = model.predict(x=word_ix_list)
            #next_ix = np.argmax(prediction)
            prediction = model.predict(x=word_ix_list,verbose=0)[0]
            next_ix = sample(prediction, temp)
            predicted_word = ix_to_word(next_ix,w2vmodel)

            generated += (' ' + predicted_word) #add predicted word to the generated output

            #remove first word from the word list to reduce the array for the max sequence length for the model
            word_ix_list = np.append(word_ix_list,next_ix)
            word_ix_list.shape
            word_ix_list = np.delete(word_ix_list,0,0)
        print(generated)
        print('-----')
    #print(generated)
    return

def sample(a, temp=1.0):
    try:
        a = np.log(a) / temp
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))
            # prediction = np.asarray(a).astype('float64')
            # prediction = np.log(prediction) / temp
            # exp_prediction= np.exp(prediction)
            # prediction = exp_prediction / np.sum(exp_prediction)
            # probabilities = np.random.multinomial(1, prediction, 1)
            # return np.argmax(probabilities)
    except:
        #print('Temperature cannot be 0. Temperature set to 1.')
        return np.argmax(a)


def on_epoch_end(epoch, logs):
    """ This callback is invoked at the end of each epoch. """
    global w2vmodel
    global max_seq_length
    global word_to_ix
    global ix_to_word
    global sample
    if epoch % 5 == 0:
        generate_text(model, w2vmodel, epoch, length=75, max_seq_length=max_seq_length,
         seed="Rain drop, drop top\n")
    return

def load_w2v_model():
    #load word 2 vec model
    w2vmodel = Word2Vec.load('word2vec_model')

    w2v_weights = w2vmodel.wv.syn0

    #prepare data for training
    vocab_size, embedding_size = w2v_weights.shape

    return w2vmodel, w2v_weights

def create_training_data(w2vmodel):
    #import lyrics
    lyrics_train=pickle.load(open('lines_train_'+str(max_seq_length)+'.p','rb'))
    lyrics_test=pickle.load(open('lines_test_'+str(max_seq_length)+'.p','rb'))
    #createtraining Data
    X_train=[]
    y_train=[]
    for line in lyrics_train:
        X_train.append([word_to_ix(word,w2vmodel) for word in line[:-1] if word!=''])
        y_train.append(word_to_ix(line[-1],w2vmodel))

    #create testing data
    X_test=[]
    y_test=[]
    for line in lyrics_test:
        X_test.append([word_to_ix(word,w2vmodel) for word in line[:-1] if word!=''])
        y_test.append(word_to_ix(line[-1],w2vmodel))

    #pad training and testing X data with the unseen word2vec (hopefully the model will learn this is useless)
    X_train=sequence.pad_sequences(X_train, maxlen=max_seq_length, value=word_to_ix('_UNSEEN_',w2vmodel))
    X_test=sequence.pad_sequences(X_test, maxlen=max_seq_length, value=word_to_ix('_UNSEEN_',w2vmodel))

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    w2vmodel, w2v_weights = load_w2v_model()

    X_train, y_train, X_test, y_test = create_training_data(w2vmodel)

    model = lstm_model(num_layers, dropout, layer_size, w2v_weights, max_seq_length)

    # Construct a hyperparameter string for each model
    model_name = 'model_dropout' + str(dropout) + '_num_layers_'+ str(num_layers) +'_layersize_' + str(layer_size) + '_batch_size_' + str(batch_size) +'max_seq_length' + str(max_seq_length)

    print_callback=LambdaCallback(on_epoch_end=on_epoch_end)
    tbCallBack = TensorBoard(log_dir='./' + model_name +'/logs', histogram_freq=0, write_graph=True, write_images=True)
    checkpointer = ModelCheckpoint(filepath='./' + model_name + '/' + model_name + '_weights.h5', verbose=1, save_best_only=False)

    print('##'*50)
    print('Starting training for %s' % model_name)
    #add epochs
    model.fit(X_train, y_train, validation_data=[X_test,y_test], batch_size=batch_size,
              epochs=epochs, callbacks=[print_callback, tbCallBack, checkpointer], verbose=2, shuffle=False)


    print('Done training!')
    print('Run `tensorboard --logdir=%s` to see the results.' % './' + model_name)
