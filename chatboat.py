# chatbot_full.py
# Complete Chatbot with NLP, Memory, and Database support

import random
import json
import sqlite3
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

lemmatizer = WordNetLemmatizer()

# Step 1: Load intents
with open('intents.json') as file:
    intents = json.load(file)

# Step 2: Preprocess text data
words, classes, documents = [], [], []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]))
classes = sorted(set(classes))

# Step 3: Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    for word in words:
        bag.append(1 if word in word_patterns else 0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

np.random.shuffle(training)
train_x = np.array([x[0] for x in training])
train_y = np.array([x[1] for x in training])

# Step 4: Model building
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# Step 5: Train the chatbot
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save("chatbot_model.h5")
pickle.dump({'words': words, 'classes': classes}, open('training_data.pkl', 'wb'))

# Step 6: Database setup
conn = sqlite3.connect('chatbot_memory.db')
cursor = conn.cursor()
cursor.execute('CREATE TABLE IF NOT EXISTS chat_memory (user_input TEXT, bot_response TEXT)')
conn.commit()

# Step 7: Define helper functions
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(w.lower()) for w in tokens]

def bag_of_words(text):
    tokens = clean_text(text)
    bag = [1 if w in tokens else 0 for w in words]
    return np.array(bag)

def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure, please try asking differently."

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    if results:
        return classes[results[0][0]]
    else:
        return None

def chatbot_response(msg):
    tag = predict_class(msg)
    response = get_response(tag) if tag else "Sorry, I didn’t get that."
    cursor.execute("INSERT INTO chat_memory (user_input, bot_response) VALUES (?, ?)", (msg, response))
    conn.commit()
    return response

# Step 8: Chat loop
print("Chatbot ready! Type 'quit' to exit.")
while True:
    message = input("You: ")
    if message.lower() == 'quit':
        break
    print("Bot:", chatbot_response(message))
