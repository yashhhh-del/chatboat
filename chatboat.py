# app.py — Streamlit Chatbot (NLP + SQLite + Memory)

import sys

# ---- FIX SQLITE IMPORT FOR STREAMLIT ----
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except:
    import sqlite3

import streamlit as st
import json
import random
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ---- STREAMLIT UI SETUP ----
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 Smart ChatBot with NLP + Memory")

lemmatizer = WordNetLemmatizer()

# ---- NLTK RESOURCE DOWNLOAD FIX ----
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

# ---- DATABASE SETUP ----
conn = sqlite3.connect('chat_memory.db')
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS chat_memory (user_input TEXT, bot_response TEXT)")
conn.commit()

# ---- LOAD INTENTS ----
with open("intents.json", "r") as file:
    intents = json.load(file)

# ---- TRAIN OR LOAD MODEL ----
try:
    model = pickle.load(open('trained_chatbot.pkl', 'rb'))
    words, classes = pickle.load(open('training_data.pkl', 'rb'))
except:
    words, classes, documents = [], [], []
    ignore_letters = ['?', '!', '.', ',']

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            documents.append((tokens, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]))
    classes = sorted(classes)

    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        token_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
        bag = [1 if w in token_words else 0 for w in words]
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    import random
    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    model = Sequential([
        Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(train_y[0]), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
    model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=0)

    pickle.dump((words, classes), open('training_data.pkl', 'wb'))
    pickle.dump(model, open('trained_chatbot.pkl', 'wb'))

# ---- NLP PROCESSING ----
def clean_text(sentence):
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in tokens]

def bag_of_words(sentence):
    tokens = clean_text(sentence)
    return np.array([1 if w in tokens else 0 for w in words])

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return classes[results[0][0]] if results else None

def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

def chatbot_response(message):
    tag = predict_class(message)
    response = get_response(tag)
    cursor.execute("INSERT INTO chat_memory VALUES (?, ?)", (message, response))
    conn.commit()
    return response

# ---- CHAT DISPLAY + MEMORY ----
st.sidebar.header("📁 Options")
if st.sidebar
