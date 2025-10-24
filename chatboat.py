# app.py
# Streamlit Chatbot with Full-Featured Backend Integration

import streamlit as st
import sqlite3
import joblib
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

# Initialize database and model
conn = sqlite3.connect('chatbot_memory.db')
cursor = conn.cursor()
cursor.execute('CREATE TABLE IF NOT EXISTS chat_memory (user_input TEXT, bot_response TEXT)')
conn.commit()

lemmatizer = WordNetLemmatizer()
model = load_model("chatbot_model.h5")
data = joblib.load("training_data.pkl")

words = data['words']
classes = data['classes']

# Load intents
with open("intents.json") as f:
    intents = json.load(f)

# Utility functions
def clean_text(sentence):
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in tokens]

def bag_of_words(sentence):
    tokens = clean_text(sentence)
    bag = [1 if w in tokens else 0 for w in words]
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    if results:
        return classes[results[0][0]]
    return None

def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return np.random.choice(intent["responses"])
    return "Sorry, I’m not sure I understand that."

def chatbot_reply(user_message):
    tag = predict_class(user_message)
    response = get_response(tag) if tag else "Could you clarify that?"
    cursor.execute("INSERT INTO chat_memory VALUES (?, ?)", (user_message, response))
    conn.commit()
    return response

# Streamlit layout
st.title("🤖 Advanced AI Chatbot")
st.info("Type your message below to chat with the AI Assistant.")

# Chat history management
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input
user_input = st.chat_input("Say something...")

if user_input:
    response = chatbot_reply(user_input)
    st.session_state.messages.append(("You", user_input))
    st.session_state.messages.append(("Bot", response))

# Display conversation
for sender, msg in st.session_state.messages:
    if sender == "You":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)

# Sidebar: memory and controls
with st.sidebar:
    st.header("⚙️ Chat Controls")
    if st.button("🗑️ Clear Chat Memory"):
        cursor.execute("DELETE FROM chat_memory")
        conn.commit()
        st.session_state.messages = []
        st.success("Chat memory cleared!")

    st.markdown("**Chat History (Last 10 entries):**")
    cursor.execute("SELECT * FROM chat_memory ORDER BY rowid DESC LIMIT 10")
    rows = cursor.fetchall()
    for row in rows:
        st.markdown(f"🧍‍♂️ *You:* {row[0]}\n🤖 *Bot:* {row[1]}")

