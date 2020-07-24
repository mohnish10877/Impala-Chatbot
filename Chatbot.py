import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras import backend
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
word_data=[]
class_data = []
documents = []
ig_words = ['?', '!']
data_file = open('D:\Academics\Projects\Impala Chatbot\intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        word_data.extend(w)
        # adding documents
        documents.append((w, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in class_data:
            class_data.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in word_data if w not in ig_words]
words = sorted(list(set(word_data)))

classes = sorted(list(set(class_data)))

print (len(documents), "documents")

print (len(class_data), "classes", class_data)

print (len(word_data), "unique lemmatized words", word_data)


pickle.dump(word_data,open('words.pkl','wb'))
pickle.dump(class_data,open('classes.pkl','wb'))

training = []
output_empty = [0] * len(class_data)

for doc in documents:
    bag = []
    pattern_words = doc[0]

    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    for w in word_data:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[class_data.index(doc[1])] = 1

    training.append([bag, output_row])
random.shuffle(training)
training = np.array(training)
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('D:\Academics\Projects\Impala Chatbot\chatbot_model.h5', hist)

print("model created")