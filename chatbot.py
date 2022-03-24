import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
#from keras.optimizers import SGD
from tensorflow.keras.optimizers import SGD
import random
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Initial Declaration
words=[]                    
classes = []
documents = []
ignore_words = ['?', '!']

# Json Data file Open and Load
data_file = open('data/Common.json', encoding='utf-8').read()
intents = json.loads(data_file)

# Data Tokenization and Preparation
for intent in intents['intents']:
    for pattern in intent['patterns']:

        w = nltk.word_tokenize(pattern)
        words.extend(w)

        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize word using WordNet's built-in morphy function. Returns the input word unchanged if it cannot be found in WordNet- Ignore words.
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

# Sorting words and classes array
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique lemmatized words", words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

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

# create train and test lists. 
# X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


# Create model - 3 layers. 
# First layer 128 neurons, 
# second layer 64 neurons, 
# third output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))   # First
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))                                    # Second
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))                    # Third

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")

