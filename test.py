import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
from keras.models import load_model
import random
import json
import pickle
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

stemmer = LancasterStemmer()

data = pickle.load(open("training_data", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

with open('intents.json') as json_data:
    intents = json.load(json_data)

model = load_model('model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return np.array(bag)


context = {}

ERROR_THRESHOLD = 0.25


def classify(sentence):
    results = model.predict(bow(sentence, words).reshape(1, -1))[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list


def response(sentence, user_id='123'):
    results = classify(sentence)
    if not results or results[0][1] < ERROR_THRESHOLD:
        return 'Sorry i did not get that'
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    if 'context_set' in i:
                        context[user_id] = i['context_set']

                    if 'context_filter' not in i or \
                            (user_id in context and 'context_filter' in i and i['context_filter'] == context[user_id]):
                        return random.choice(i['responses'])

            results.pop(0)


print("Don't be shy, start the conversation :)")
while True:
    item = input()
    if response(item) is None:
        print('I did not get that.')
    else:
        print(response(item))
