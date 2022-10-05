import logging
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import nltk
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
# from keras.optimizers import SGD
from tensorflow.keras.optimizers import SGD

from common.file_helper import FileHelper
from common.logger import set_log_conf
from common.sentence_preprocessor import SentencePreProcessor

# Log File configuration
set_log_conf()

file_helper = FileHelper()
sentence_preprocessor = SentencePreProcessor()

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

logging.info("Json Data file Open and Load >>>>>")
# Json Data file Open and Load
intents = file_helper.load_intents()
logging.info(f"intents {intents}")

# Data Tokenization and Preparation
logging.info(f"Lemmatize word using WordNet's built-in morphy function")
tokenized_data = sentence_preprocessor.tokenize_data(intents)
words = tokenized_data["words"]
classes = tokenized_data["classes"]
documents = tokenized_data["documents"]

file_helper.dump_words_file(words)
file_helper.dump_classes_file(classes)

logging.info("initializing training data >>>>>")
# initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:

    bag = []

    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

logging.info("create train and test lists  >>>>>")
# create train and test lists. 
# X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
logging.info("Training data created")


logging.info("Create model - 3 layers >>>>>")
# Create model - 3 layers. 
# First layer 128 neurons, 
# second layer 64 neurons, 
# third output layer contains number of neurons
# equal to number of intents to predict output intent with softmax

# model = Sequential()
# model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))   # First
# model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))                                   # Second
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))                                    # Third
# model.add(Dropout(0.5))
# model.add(Dense(len(train_y[0]), activation='softmax'))                # Last

model_test = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape = (len(train_x[0]),)),
  tf.keras.layers.Dense(256, activation = "relu"),
  tf.keras.layers.Dense(128, activation = "relu"),
  tf.keras.layers.Dense(64, activation = "relu"),
  tf.keras.layers.Dense(len(train_y[0]), activation = "softmax")
])

logging.info("Compile model - 3 layers >>>>>")
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

#model_test.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model_test.compile(loss = 'categorical_crossentropy',
                optimizer = tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

logging.info("fitting and saving the model >>>>>")
#fitting and saving the model
hist = model_test.fit(np.array(train_x), np.array(train_y), epochs=17, batch_size=5, verbose=1)
file_helper.save_model(model_test, hist)

fig, ax2 = plt.subplots(1, figsize=(15, 5))
ax2.plot(hist.history['loss'])
ax2.plot(hist.history['accuracy'])
ax2.legend(["train", "test"], loc="upper right")
ax2.set_xlabel("Loss")
ax2.set_ylabel("Accuracy")
plt.show()

logging.info("Model Created >>>>>")