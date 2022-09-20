import json
import os
import pickle
import random

import nltk
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

MODEL_PATH = 'data/models/en'
INTENTS_PATH = 'data/intents/en'
MODEL_SAVE_PATH = os.path.join(MODEL_PATH, 'chatbot_model.h5')
WORDS_SAVE_PATH = os.path.join(MODEL_PATH, 'words.pkl')
CLASSES_SAVE_PATH = os.path.join(MODEL_PATH, 'classes.pkl')

# logging.basicConfig(filename=constant.LOG_FILE_PATH,
#                      format='%(asctime)s %(levelname)-8s %(message)s',
#                      level=logging.DEBUG,
#                      datefmt='%Y-%m-%d %H:%M:%S')
# logging.getLogger('matplotlib.font_manager').disabled = True

lemmatizer = WordNetLemmatizer()
model = load_model(MODEL_SAVE_PATH)
intents = json.loads(open(os.path.join(INTENTS_PATH, 'Common.json'), encoding='utf-8').read())
words = pickle.load(open(WORDS_SAVE_PATH,'rb'))
classes = pickle.load(open(CLASSES_SAVE_PATH,'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
        else:
            result = "You must ask the right questions"
    return result

def chatbot_response(msg):
    #logging.info("[processor.py] predict_class >>>>>")
    ints = predict_class(msg, model)
    #logging.info("[processor.py] getResponse >>>>>")
    res = getResponse(ints, intents)
    return res
