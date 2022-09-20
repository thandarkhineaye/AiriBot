import json
import os
import pickle
import random

import numpy as np
from janome.analyzer import Analyzer
from janome.tokenfilter import POSStopFilter
from janome.tokenizer import Tokenizer
from tensorflow import keras

MODEL_PATH = 'data/models/jp'
INTENTS_PATH = 'data/intents/jp'
MODEL_SAVE_PATH = os.path.join(MODEL_PATH, 'chatbot_model.h5')
WORDS_SAVE_PATH = os.path.join(MODEL_PATH, 'words.pkl')
CLASSES_SAVE_PATH = os.path.join(MODEL_PATH, 'classes.pkl')

error_threshold = 0.25

tokenizer = Tokenizer()
# 読み捨てるトークンの品詞を指定する
# token_filters = [POSStopFilter(['記号','助詞','助動詞','動詞'])]
token_filters = [POSStopFilter(['記号', '助詞', '助動詞'])]
anal = Analyzer(tokenizer=tokenizer, token_filters=token_filters)
# 学習したモデルを読み込む
model = keras.models.load_model(MODEL_SAVE_PATH)

# 教師データに出現した単語の目録リスト
with open(WORDS_SAVE_PATH, 'rb') as handle:
    words = pickle.load(handle)

# 教師データに出現した意図ラベルの目録リスト
with open(CLASSES_SAVE_PATH, 'rb') as enc:
    classes = pickle.load(enc)


# 対象テキスト文字列を基本形の分かち書きに変換する
def str_wakati(w):
    wakati = ''
    tokens = anal.analyze(w)
    wakati = ' '.join([t.base_form for t in tokens])
    return wakati


# 対象テキスト文字列を基本形の分かち書きリストに変換する
def str_wakati_list(w):
    wakati = []
    tokens = anal.analyze(w)
    wakati.append([t.base_form for t in tokens])
    return wakati[0]


# 意図ファイルを読み込む
with open(os.path.join(INTENTS_PATH, 'intents.json'), encoding="utf-8") as file:
    intents = json.load(file)


def chat(input_msg: str):
    # ユーザ入力文章を分かち書き・レンマ化してリスト化する
    w_inp_list = str_wakati_list(input_msg)
    # one-hot表現の配列に変換する
    bag = [0] * len(words)
    for s in w_inp_list:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    # 応答文を獲得する
    res = model.predict(np.array([bag]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    tag = return_list[0]['intent']
    list_of_intents = intents['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
        else:
            result = "うまくお答えできません"
    return result
