from flask import Flask
from flask import request
from flask import json
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
#from keras import backend as K
#from random import randint
import tensorflow as tf
import os

MODEL_FILE_NAME = 'model_test.h5'
TOKENIZER_FILE_NAME = 'tokenizer_test.pkl'
modelObj = None
tokenizerObj = None
global graph

# generate a sequence from a language model
def generate_seq(seq_length, seed_text, n_words):
    with graph.as_default():
        result = list()
        in_text = seed_text
        # generate a fixed number of words
        for _ in range(n_words):
            # encode the text as integer
            encoded = tokenizerObj.texts_to_sequences([in_text])[0]
            # truncate sequences to a fixed length
            encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
            # predict probabilities for each word
            yhat = modelObj.predict_classes(encoded, verbose=0)
            # map predicted word index to word
            out_word = ''
            for word, index in tokenizerObj.word_index.items():
                if index == yhat:
                    out_word = word
                    break
            # append to input
            in_text += ' ' + out_word
            result.append(out_word)
        return ' '.join(result)

app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
    payload = json.loads(request.get_data().decode('utf-8'))
    prediction = predict(payload['text'])
    data = {}
    data['data'] = prediction
    return json.dumps(data)

def loadModel(modelFileName) :
    # load the model
    return load_model(modelFileName)

def loadTokenizer(tokenizerFileName) :
    # load the tokenizer
    return load(open(tokenizerFileName, 'rb'))


def predict(seed_text) :
    seq_length = 5
    generated = generate_seq(seq_length, seed_text, 2)
    #K.clear_session()
    return generated

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    try:
        modelObj = loadModel(MODEL_FILE_NAME)
        print('Model loaded')
        tokenizerObj = loadTokenizer(TOKENIZER_FILE_NAME)
        print('Tokenizer loaded')
        graph = tf.get_default_graph()

    except Exception as e:
        print('No model here')
        print('Train first')
        print(e)
        modelObj = None
        tokenizerObj = None
    app.run(debug=True, port=port)
