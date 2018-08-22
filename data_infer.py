from flask import Flask
from flask import request
from flask import json
from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
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
    MODEL_FILE_NAME = 'model_test.h5'
    TOKENIZER_FILE_NAME = 'tokenizer_test.pkl'
    generated = generate_seq(loadModel(MODEL_FILE_NAME), loadTokenizer(TOKENIZER_FILE_NAME), seq_length, seed_text, 2)
    K.clear_session()
    return generated

if __name__ == "__main__":
    app.run()
