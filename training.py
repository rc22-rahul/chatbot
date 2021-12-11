import json
import random
import numpy as np
import nltk


import pickle

from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

words = []
classes = []
documents = []
ignore_letters = ['?', '.', ',', '!']

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)

        documents.append((word_list, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open("words.pkl", "wb")) #serialinzing the data and storing it
pickle.dump(classes, open("classes.pkl", "wb"))

training = []
output_empty = [0]*len(classes)  # one hot encoding[0,1,0,1]

for document in documents:
    bag = []
    word_pattern = document[0]
    word_pattern = [lemmatizer.lemmatize(word.lower()) for word in word_pattern]

    for word in words:
        bag.append(1) if word in word_pattern else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training=np.array(training)

#splitting data in training and set

train_x = list(training[:, 0])

train_y = list(training[:, 1])
print('Training data created')


'''In short, a dropout layer ignores a set of neurons (randomly) as one can see in the picture below.
 This normally is used to prevent the net from overfitting.
  The Dense layer is a normal fully connected layer in a neuronal network.'''


model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))  #128 neurons relu for positve otherwise zero
model.add(Dropout(0.5))   #prevent overfitting
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))   # neuron as many as classes and softmax for likelihood for that class

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('Chatbotmodel.h5',hist)
print('model created')

