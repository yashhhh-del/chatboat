import sys
import os

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

# ---- STREAMLIT SETUP ----
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 Smart ChatBot with NLP + Memory")

lemmatizer = WordNetLemmatizer()

# ---- NLTK DOWNLOAD FIX ----
for resource in ["punkt", "punkt_tab", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

# ---- DATABASE SETUP ----
conn = sqlite3.connect('chat_memory.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""CREATE TABLE IF NOT EXISTS chat_memory (
    user_input TEXT,
    bot_response TEXT
)""")
conn.commit()

# ---- LOAD INTENTS ----
if not os.path.exists("intents.json"):
    st.error("❌ intents.json missing! Place it in the same folder as app.py")
    st.stop()

with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

# ---- LOAD OR TRAIN MODEL ----
if os.path.exists("trained_chatbot.pkl") and os.path.exists("training_data.pkl"):
    model = pickle.load(open("trained_chatbot.pkl", "rb"))
    words, classes = pickle.load(open("training_data.pkl", "rb"))
else:
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
        row = list(output_empty)
        row[classes.index(doc[1])] = 1
        training.append([bag, row])

    training = np.array(training, dtype=object)
    train_x = np.array(list(training[:,0])).astype(np.float32)
    train_y = np.array(list(training[:,1])).astype(np.float32)

    model = Sequential([
        Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(train_y[0]), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=100, batch_size=5, verbose=0)

    pickle.dump((words, classes), open("training_data.pkl", "wb"))
    pickle.dump(model, open("trained_chatbot.pkl", "wb"))

# ---- HELPER FUNCTIONS ----
def clean_text(sentence):
    return [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(sentence)]

def bag_of_words(sentence):
    tokens = clean_text(sentence)
    return np.array([1 if w in tokens else 0 for w in words])

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    thresh = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > thresh]
    results.sort(key=lambda x: x[1], reverse=True)
    return classes[results[0][0]] if results else None

def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn’t understand that."

def chatbot_response(msg):
    tag = predict_class(msg)
    response = get_response(tag) if tag else "Sorry, I didn’t get that."
    cursor.execute("INSERT INTO chat_memory VALUES (?, ?)", (msg, response))
    conn.commit()
    return response

# ---- STREAMLIT UI ----
st.sidebar.header("📁 Options")
if st.sidebar.button("🗑 Clear Chat History"):
    cursor.execute("DELETE FROM chat_memory")
    conn.commit()
    st.session_state.messages = []
    st.success("✅ Chat history cleared!")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Type your message...")

if user_input:
    reply = chatbot_response(user_input)
    st.session_state.messages.append(("You", user_input))
    st.session_state.messages.append(("Bot", reply))

for sender, msg in st.session_state.messages:
    if sender == "You":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)
