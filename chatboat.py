import sys

# ---- SQLITE FIX ----
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ModuleNotFoundError:
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

# ---- PAGE SETUP ----
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 Smart ChatBot with NLP and Memory")

# ---- NLTK FIX FOR STREAMLIT ----
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

lemmatizer = WordNetLemmatizer()

# ---- DB SETUP ----
conn = sqlite3.connect('chat_memory.db')
cursor = conn.cursor()
cursor.execute("""CREATE TABLE IF NOT EXISTS chat_memory (
    user_input TEXT,
    bot_response TEXT
)""")
conn.commit()

# ---- LOAD INTENTS ----
with open("intents.json", "r") as f:
    intents = json.load(f)

# ---- LOAD OR TRAIN MODEL ----
try:
    words, classes = pickle.load(open('training_data.pkl', 'rb'))
    model = pickle.load(open('trained_chatbot.pkl', 'rb'))
except:
    words, classes, documents = [], [], []
    ignore = ['?', '!', '.', ',']

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            documents.append((tokens, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore]))
    classes = sorted(classes)

    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = [1 if w in [lemmatizer.lemmatize(w.lower()) for w in doc[0]] else 0 for w in words]
        row = list(output_empty)
        row[classes.index(doc[1])] = 1
        training.append([bag, row])

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

# ---- FUNCTIONS ----
def clean(sentence):
    return [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(sentence)]

def bag_of_words(sentence):
    tokens = clean(sentence)
    return np.array([1 if w in tokens else 0 for w in words])

def respond(msg):
    bow = bag_of_words(msg)
    res = model.predict(np.array([bow]))[0]
    thresh = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > thresh]
    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
        reply = "Sorry, I didn’t understand that."
    else:
        tag = classes[results[0][0]]
        for intent in intents['intents']:
            if intent['tag'] == tag:
                reply = random.choice(intent['responses'])

    cursor.execute("INSERT INTO chat_memory VALUES (?, ?)", (msg, reply))
    conn.commit()
    return reply

# ---- UI ----
st.sidebar.header("📁 Options")
if st.sidebar.button("🗑 Clear Chat"):
    cursor.execute("DELETE FROM chat_memory")
    conn.commit()
    st.session_state.messages = []
    st.success("✅ Chat Cleared!")

if "messages" not in st.session_state:
    st.session_state.messages = []

user = st.chat_input("Say something...")

if user:
    bot = respond(user)
    st.session_state.messages.append(("You", user))
    st.session_state.messages.append(("Bot", bot))

for sender, text in st.session_state.messages:
    if sender == "You":
        st.chat_message("user").markdown(text)
    else:
        st.chat_message("assistant").markdown(text)
