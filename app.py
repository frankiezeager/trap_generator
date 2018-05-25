import numpy as np
import flask
from flask import request, render_template, flash
import io
import keras
from gensim.models import Word2Vec
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Embedding, TimeDistributed, Activation, Dropout, Lambda
from keras.layers import LSTM
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import re

application = flask.Flask(__name__)
application.config.from_object(__name__)

class ReusableForm(Form):
    seed = TextField('Seed:', validators=[validators.required()])
    length = TextField('Length:',validators=[validators.required()])
    temperature = TextField('Temperature:',validators=[validators.required()])
#model = None

num_layers = 2
dropout=.2
layer_size = 512
max_seq_length = 10

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

def load_model():
    #load word 2 vec model
    global w2vmodel
    w2vmodel = Word2Vec.load('word2vec_model')
    w2v_weights = w2vmodel.wv.syn0
    global model
    model = lstm_model(num_layers, dropout, layer_size, w2v_weights, max_seq_length)
    model.load_weights('model/model_dropout0.05_num_layers_2_layersize_512_batch_size_64max_seq_length6_weights.h5', by_name=True)

def sample(a, temp=1.0):
    try:
        a = np.log(a) / temp
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))
    except:
        #print('Temperature cannot be 0. Temperature set to 1.')
        return np.argmax(a)

def clean_seed(seed):
    #clean seed
    seed=re.sub(r'<[^<]+?>', '', seed)
    #remove encoding characters like \x86
    seed=re.sub(r'[^\x00-\x7f]','',seed)
    seed=re.sub(r'\#','',seed)
    #remove punctuation
    seed=re.sub(r'[^A-Za-z0-9\s]','',seed)
    return seed

@application.route("/", methods=["GET"])
def serve_form():
    return render_template("form.html", form=ReusableForm(request.form))

@application.route("/", methods=['POST'])
def generate_text(): # model, w2vmodel, length=75, max_seq_length=20, temp=1, seed="Rain drop drop top"):
    load_model()
    global w2vmodel
    w2v_weights = w2vmodel.wv.syn0

    form = ReusableForm(request.form)
    print(form.errors)
    #initialize data dictionary that will be returned by app
    data = {"success": False}

    if flask.request.method =='POST':
        seed=request.form['seed']

        length=int(request.form['length'])
        temp=float(request.form['temperature'])
        #add some catches for text in those boxes
        print('Starting with seed: ', seed)

        generated = '\n'

        generated += seed

        seed = clean_seed(seed)

        #shorten if longer than max_seq_length
        seed = seed.split(' ')[:max_seq_length]

        word_ix_list = []
        for word in seed:
            try:
                word = word_to_ix(word,w2vmodel)
            except:
                #since we're using -1 as a null word (why we also pad with the not in vocab), we'll use that for words that aren't in the word2vec model
                print('Warning: {0} not contained in training vocabulary. It will be ignored when computing output.'.format(word))
                word = word_to_ix('_UNSEEN_',w2vmodel)
            word_ix_list.append(word)

        #pad word_list with the unseen word2vec if shorter than max_seq_length
        word_ix_list = [word_to_ix('_UNSEEN_',w2vmodel)] * (max_seq_length-len(word_ix_list)) + word_ix_list

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

        return render_template('success.html', generated=generated)

if __name__ == '__main__':
    print("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started")
    load_model()
    application.run()
