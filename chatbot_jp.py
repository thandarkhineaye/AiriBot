import logging
import random

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

from common.config import load_config
from common.file_helper import FileHelper
from common.logger import set_log_conf
from common.sentence_preprocessor import SentencePreProcessor


# Log File configuration
set_log_conf()
config = load_config()

LANGUAGE = config["model"]["language"]["japan"]

file_helper = FileHelper(config)
sentence_preprocessor = SentencePreProcessor(config, LANGUAGE)

# 意図ファイルを読み込む
logging.info("Json Data file Open and Load")
data = file_helper.load_intents(LANGUAGE)
logging.info(f"intents {data}")


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

# Data Tokenization and Preparation
tokenized_data = sentence_preprocessor.tokenize_data(data)
words = tokenized_data["words"]
classes = tokenized_data["classes"]
documents = tokenized_data["documents"]

# 目録をpickleファイルに保存する
file_helper.dump_words_file(words, LANGUAGE)
file_helper.dump_classes_file(classes, LANGUAGE)

logging.info("initializing training data")
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

logging.info("create train and test lists")
# 訓練用配列に格納する
train_x = list(training[:,0])
train_y = list(training[:,1])
logging.info("Training data created")

logging.info("Create model - 3 layers")
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

logging.info("Compile model - 3 layers")
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

logging.info("fitting and saving the model")
hist = model.fit(np.array(train_x), np.array(train_y), epochs=120, batch_size=5, verbose=1)
file_helper.save_model(model, hist, LANGUAGE)

fig, ax2 = plt.subplots(1, figsize=(15, 5))
ax2.plot(hist.history['loss'])
ax2.plot(hist.history['accuracy'])
ax2.legend(["train", "test"], loc="upper right")
ax2.set_xlabel("Loss")
ax2.set_ylabel("Accuracy")
plt.show()

logging.info("Model Created")
