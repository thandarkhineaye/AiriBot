import json
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from janome.analyzer import Analyzer
from janome.tokenfilter import POSStopFilter
from janome.tokenizer import Tokenizer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

MODEL_PATH = 'data/models/jp'
INTENTS_PATH = 'data/intents/jp'
MODEL_SAVE_PATH = os.path.join(MODEL_PATH, 'chatbot_model.h5')
WORDS_SAVE_PATH = os.path.join(MODEL_PATH, 'words.pkl')
CLASSES_SAVE_PATH = os.path.join(MODEL_PATH, 'classes.pkl')

# 意図ファイルを読み込む
with open(os.path.join(INTENTS_PATH, 'Common.json'), "r", encoding="utf-8") as file:
    data = json.load(file)
print(data)
print(type(data))

# 形態素解析のためのインスタンスの生成
tokenizer = Tokenizer()
# 読み捨てるトークンの品詞を指定する
# token_filters = [POSStopFilter(['記号','助詞','助動詞','動詞'])]
token_filters = [POSStopFilter(['記号','助詞','助動詞'])]
anal = Analyzer(tokenizer=tokenizer, token_filters=token_filters)


# 辞書データからある階層の内容を取り出す
def nested_item_value(parrent_object, nest_list):
    """ return nested data """

    if not nest_list: return parrent_object

    result = ""
    for nest_key in nest_list:
        object_type = type(parrent_object)
        if object_type is not dict and object_type is not list:
            result = None
            break
        elif object_type is list:
            if type(nest_key) is not int:
                result = None
                break
            result = parrent_object[nest_key] if nest_key < len(parrent_object) else None
        else:
            result = parrent_object.get(nest_key, None)

        if result is None: break

        parrent_object = result

    return result


kw_list = nested_item_value(data,['intents',3,'responses'])
print(kw_list)


# 対象テキスト文字列を基本形の分かち書きに変換する
def str_wakati(w):
    wakati=''
    tokens = anal.analyze(w)
    wakati = ' '.join([t.base_form for t in tokens])
    return wakati


# 対象テキスト文字列を基本形の分かち書きリストに変換する
def str_wakati_list(w):
    wakati=[]
    tokens = anal.analyze(w)
    wakati.append([t.base_form for t in tokens])
    return wakati[0]


print(str_wakati_list(kw_list[1]))

# 作業用リスト
words = []
classes = []
documents = []

for intent in data['intents']:
    # 各々の教師文(pattern)について繰り返す
    for pattern in intent['patterns']:
        # 教師文を分かち書きリストにする
        w = str_wakati_list(pattern)
        # 分かち書きにした教師文のリスト
        words.extend(w)
        # リスト化した教師文とラベルの対
        documents.append((w, intent['tag']))
        # 教師文に存在する意図ラベルの目録リスト
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(words)
print(documents)
print(classes)

words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(words)
print(classes)

# 目録をpickleファイルに保存する
pickle.dump(words, open(WORDS_SAVE_PATH, 'wb'))
pickle.dump(classes, open(CLASSES_SAVE_PATH, 'wb'))

training = []
output_empty = [0] * len(classes)
for doc in documents:

    bag = []

    pattern_words = doc[0]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

# 訓練用配列に格納する
train_x = list(training[:,0])
train_y = list(training[:,1])

# Create model - 3 layers.
# First layer 128 neurons,
# second layer 64 neurons,
# third output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))   # First
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))                                   # Second
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))                                    # Third
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))                # Last

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=120, batch_size=5, verbose=1)
model.save(MODEL_SAVE_PATH, hist)

fig, ax2 = plt.subplots(1, figsize=(15, 5))
ax2.plot(hist.history['loss'])
ax2.plot(hist.history['accuracy'])
ax2.legend(["train", "test"], loc="upper right")
ax2.set_xlabel("Loss")
ax2.set_ylabel("Accuracy")
plt.show()

