import os
import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD


nltk.download('punkt')
nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()

script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, "intents.json")


with open(json_path, "r") as file:
    intents = json.load(file)


words = []
classes = []
documents = []
ignore_symbols = ['?', '!', '.', ',']


for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_symbols]
words = sorted(set(words))


classes = sorted(set(classes))


if not os.path.exists('model'):
    os.makedirs('model')


pickle.dump(words, open('model/words.pkl', 'wb'))
pickle.dump(classes, open('model/classes.pkl', 'wb'))


training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns if word not in ignore_symbols]


    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)


    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    

    if len(bag) == len(words) and len(output_row) == len(classes):
        training.append([bag, output_row])
    else:
        print(f"Skipping document: {document[0]} - Bag length: {len(bag)}, Output length: {len(output_row)}")


training = np.array(training, dtype=object)


train_x = list(training[:, 0])
train_y = list(training[:, 1])


model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

model.save('model/chatbot_model.keras')

print("Model training completed and saved successfully!")
