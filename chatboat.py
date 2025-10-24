# app.py — full ready-to-run chatbot using Streamlit + OpenAI or local NLP
# Requirements: streamlit, openai, nltk, joblib, sqlite3

import streamlit as st
import openai
import sqlite3
import time

# ====== SETTINGS ======
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 Smart Chatbot")
st.caption("Built with Streamlit and OpenAI | October 2025")

# ====== DATABASE SETUP ======
conn = sqlite3.connect('chat_memory.db')
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS chat_history (user TEXT, bot TEXT)")
conn.commit()

# ====== API KEY CONFIG ======
with st.sidebar:
    st.header("⚙️ Configuration")
    openai.api_key = st.text_input("Enter your OpenAI API Key", type="password")
    st.markdown("[Get your key here](https://platform.openai.com/account/api-keys)")
    if st.button("🗑️ Clear Chat History"):
        cursor.execute("DELETE FROM chat_history")
        conn.commit()
        st.session_state.messages = []
        st.success("Chat history cleared!")

# ====== INIT SESSION ======
if "messages" not in st.session_state:
    st.session_state.messages = []

# ====== CHAT LOOP ======
user_input = st.chat_input("Type your message here...")

def generate_response(prompt):
    """Use OpenAI API to create chat responses."""
    try:
        with st.spinner("Thinking... 🤯"):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful and friendly chatbot."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
        message = response.choices[0].message["content"].strip()
        return message
    except Exception as e:
        return f"Error: {e}"

if user_input:
    st.session_state.messages.append(("You", user_input))
    bot_reply = generate_response(user_input)
    st.session_state.messages.append(("Bot", bot_reply))
    cursor.execute("INSERT INTO chat_history VALUES (?, ?)", (user_input, bot_reply))
    conn.commit()
    time.sleep(0.2)

# ====== DISPLAY MESSAGES ======
for sender, msg in st.session_state.messages:
    if sender == "You":
        with st.chat_message("user"):
            st.markdown(msg)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg)

# ====== SIDEBAR CHAT LOG ======
with st.sidebar.expander("🕘 Recent Conversations", expanded=False):
    cursor.execute("SELECT * FROM chat_history ORDER BY rowid DESC LIMIT 10")
    for user, bot in cursor.fetchall():
        st.markdown(f"🧍‍♂️ **You:** {user}")
        st.markdown(f"🤖 **Bot:** {bot}")
        st.divider()
