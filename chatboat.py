import sys
import os
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
import sqlite3
import pandas as pd
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import base64
from io import BytesIO
import re

# ---- SQLITE FIX ----
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ModuleNotFoundError:
    import sqlite3

# ---- STREAMLIT SETUP ----
st.set_page_config(
    page_title="ChatBot PRO MAX - AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- PERFECT COLOR SCHEME WITH EXCELLENT CONTRAST ----
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main App Styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .block-container {
        padding: 2rem 1rem;
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    }
    
    /* Title Styling */
    h1 {
        color: #1a202c !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h3 {
        color: #4a5568 !important;
        text-align: center;
        font-weight: 500 !important;
        margin-top: 0 !important;
    }
    
    /* Chat Messages - User */
    .stChatMessage[data-testid="chat-message-user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 18px !important;
        padding: 16px 20px !important;
        margin: 10px 0 !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stChatMessage[data-testid="chat-message-user"] p,
    .stChatMessage[data-testid="chat-message-user"] span,
    .stChatMessage[data-testid="chat-message-user"] div {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 15px !important;
    }
    
    /* Chat Messages - Assistant */
    .stChatMessage[data-testid="chat-message-assistant"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
        border-radius: 18px !important;
        padding: 16px 20px !important;
        margin: 10px 0 !important;
        box-shadow: 0 4px 12px rgba(245, 87, 108, 0.3) !important;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] p,
    .stChatMessage[data-testid="chat-message-assistant"] span,
    .stChatMessage[data-testid="chat-message-assistant"] div,
    .stChatMessage[data-testid="chat-message-assistant"] strong {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 15px !important;
    }
    
    /* Avatar Icons */
    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        box-shadow: 0 2px 8px rgba(79, 172, 254, 0.4);
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%) !important;
        box-shadow: 0 2px 8px rgba(250, 112, 154, 0.4);
    }
    
    /* Caption Text - High Contrast */
    [data-testid="stCaptionContainer"] {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        opacity: 0.95 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Badges */
    .feature-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: #ffffff !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
        margin: 4px;
        box-shadow: 0 3px 10px rgba(245, 87, 108, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Confidence Colors - High Contrast */
    .confidence-high {
        color: #10b981 !important;
        font-weight: 700 !important;
        background: #d1fae5;
        padding: 4px 10px;
        border-radius: 12px;
        display: inline-block;
    }
    
    .confidence-medium {
        color: #f59e0b !important;
        font-weight: 700 !important;
        background: #fef3c7;
        padding: 4px 10px;
        border-radius: 12px;
        display: inline-block;
    }
    
    .confidence-low {
        color: #ef4444 !important;
        font-weight: 700 !important;
        background: #fee2e2;
        padding: 4px 10px;
        border-radius: 12px;
        display: inline-block;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar Sections */
    .sidebar-section {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 16px;
        margin: 16px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .sidebar-section h2,
    .sidebar-section h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        margin-bottom: 12px !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 700;
        font-size: 14px;
        box-shadow: 0 4px 12px rgba(245, 87, 108, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(245, 87, 108, 0.4);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(79, 172, 254, 0.3);
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 28px !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Chat Input */
    .stChatInput textarea {
        background-color: #ffffff !important;
        color: #1a202c !important;
        border: 2px solid #667eea !important;
        border-radius: 16px !important;
        padding: 12px 16px !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    .stChatInput textarea:focus {
        border-color: #764ba2 !important;
        box-shadow: 0 6px 20px rgba(118, 75, 162, 0.3) !important;
    }
    
    /* Artifacts */
    .artifact-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 3px solid #667eea;
        border-radius: 16px;
        padding: 24px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }
    
    .artifact-container h3 {
        color: #667eea !important;
        font-weight: 700 !important;
        margin-bottom: 16px !important;
        text-align: left !important;
    }
    
    .artifact-container p,
    .artifact-container div,
    .artifact-container span {
        color: #1a202c !important;
        font-size: 14px !important;
    }
    
    /* Code Blocks */
    code {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        padding: 3px 8px !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    
    pre {
        background: #1a202c !important;
        border: 2px solid #667eea !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }
    
    pre code {
        background: transparent !important;
        color: #a5f3fc !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 12px 16px !important;
    }
    
    .streamlit-expanderContent {
        background: #ffffff;
        border: 2px solid #667eea;
        border-radius: 0 0 12px 12px;
        padding: 16px;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* Toggle Switch */
    .stCheckbox label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 16px !important;
        font-weight: 600 !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 16px !important;
        font-weight: 600 !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 16px !important;
        font-weight: 600 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #667eea !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 16px;
        border: 2px dashed rgba(255, 255, 255, 0.4);
    }
    
    [data-testid="stFileUploader"] label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Smart ChatBot PRO MAX")
st.markdown("### *Advanced AI Assistant with Claude-like Intelligence*")

lemmatizer = WordNetLemmatizer()

# ---- NLTK DOWNLOAD ----
for resource in ["punkt", "punkt_tab", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

# ---- DATABASE SETUP ----
conn = sqlite3.connect('chat_memory.db', check_same_thread=False)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_memory'")
table_exists = cursor.fetchone()

if table_exists:
    cursor.execute("PRAGMA table_info(chat_memory)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if len(columns) == 2:
        cursor.execute("ALTER TABLE chat_memory RENAME TO chat_memory_old")
        cursor.execute("""CREATE TABLE chat_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_input TEXT,
            bot_response TEXT,
            confidence REAL,
            features_used TEXT
        )""")
        cursor.execute("""
            INSERT INTO chat_memory (timestamp, user_input, bot_response, confidence, features_used)
            SELECT datetime('now'), user_input, bot_response, 0.0, ''
            FROM chat_memory_old
        """)
        cursor.execute("DROP TABLE chat_memory_old")
        conn.commit()
else:
    cursor.execute("""CREATE TABLE IF NOT EXISTS chat_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        user_input TEXT,
        bot_response TEXT,
        confidence REAL,
        features_used TEXT
    )""")
    conn.commit()

# ---- SESSION STATE ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "artifacts" not in st.session_state:
    st.session_state.artifacts = []
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = []
if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = True
if "file_analysis_enabled" not in st.session_state:
    st.session_state.file_analysis_enabled = True

# ---- LOAD INTENTS ----
if not os.path.exists("intents.json"):
    st.error("❌ intents.json missing! Place it in the same folder as app.py")
    st.stop()

with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

# ---- LOAD OR TRAIN MODEL ----
@st.cache_resource
def load_or_train_model():
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
    
    return model, words, classes

model, words, classes = load_or_train_model()

# ---- ADVANCED FEATURES ----
def web_search_simulation(query):
    try:
        search_results = {
            "title": f"Search results for: {query}",
            "snippet": f"Based on current information about '{query}', here are some insights...",
            "source": "Web Search",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return search_results
    except Exception as e:
        return {"error": str(e)}

def detect_code_request(text):
    code_keywords = ['code', 'program', 'script', 'function', 'write', 'create', 'develop', 
                     'python', 'javascript', 'html', 'css', 'algorithm', 'implement']
    return any(keyword in text.lower() for keyword in code_keywords)

def detect_web_search_request(text):
    search_keywords = ['search', 'find', 'look up', 'what is', 'who is', 'latest', 
                       'current', 'news', 'weather', 'price', 'information about']
    return any(keyword in text.lower() for keyword in search_keywords)

def detect_file_request(text):
    file_keywords = ['file', 'document', 'upload', 'analyze', 'read', 'csv', 'excel', 
                     'pdf', 'data', 'spreadsheet']
    return any(keyword in text.lower() for keyword in file_keywords)

def generate_code_artifact(description, language="python"):
    templates = {
        "python": f"""# Generated Python Code for: {description}

def main():
    # Your implementation here
    print("Code generated based on: {description}")
    pass

if __name__ == "__main__":
    main()
""",
        "javascript": f"""// Generated JavaScript Code for: {description}

function main() {{
    // Your implementation here
    console.log("Code generated based on: {description}");
}}

main();
"""
    }
    return templates.get(language, templates["python"])

def create_artifact(content, artifact_type="code", title="Generated Artifact"):
    artifact = {
        "id": len(st.session_state.artifacts) + 1,
        "title": title,
        "type": artifact_type,
        "content": content,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.artifacts.append(artifact)
    return artifact

def analyze_file_content(file):
    try:
        if file.type == "text/csv":
            df = pd.read_csv(file)
            analysis = f"""
**File Analysis:**
- **Rows:** {len(df)}
- **Columns:** {len(df.columns)}
- **Column Names:** {', '.join(df.columns.tolist())}
- **First 5 rows preview:**

{df.head().to_markdown()}
"""
            return analysis, df
        elif file.type == "text/plain":
            content = file.read().decode("utf-8")
            return f"**Text File Content:**\n\n{content[:1000]}{'...' if len(content) > 1000 else ''}", None
        else:
            return f"File type: {file.type} uploaded successfully.", None
    except Exception as e:
        return f"Error analyzing file: {str(e)}", None

# ---- HELPER FUNCTIONS ----
def clean_text(sentence):
    return [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(sentence)]

def bag_of_words(sentence):
    tokens = clean_text(sentence)
    return np.array([1 if w in tokens else 0 for w in words])

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    thresh = 0.6
    results = [[i, r] for i, r in enumerate(res) if r > thresh]
    results.sort(key=lambda x: x[1], reverse=True)
    if not results:
        return None, 0
    return classes[results[0][0]], results[0][1]

def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure about that. Could you rephrase?"

def chatbot_response(msg, uploaded_file=None):
    features_used = []
    additional_content = ""
    
    if st.session_state.web_search_enabled and detect_web_search_request(msg):
        search_results = web_search_simulation(msg)
        features_used.append("web_search")
        additional_content += f"\n\n🔍 **Web Search Results:**\n{search_results.get('snippet', '')}"
    
    if detect_code_request(msg):
        features_used.append("code_generation")
        code = generate_code_artifact(msg)
        artifact = create_artifact(code, "code", f"Code for: {msg[:50]}")
        additional_content += f"\n\n💻 **Generated Code Artifact (ID: {artifact['id']})**"
    
    if uploaded_file and st.session_state.file_analysis_enabled:
        features_used.append("file_analysis")
        file_analysis, df = analyze_file_content(uploaded_file)
        additional_content += f"\n\n📄 **File Analysis:**\n{file_analysis}"
    
    tag, confidence = predict_class(msg)
    
    if confidence is None:
        confidence = 0.0
    
    base_response = get_response(tag) if tag else "I'm not sure about that. Could you rephrase?"
    full_response = base_response + additional_content
    
    try:
        cursor.execute(
            "INSERT INTO chat_memory (timestamp, user_input, bot_response, confidence, features_used) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg, full_response, float(confidence), ", ".join(features_used))
        )
        conn.commit()
    except sqlite3.OperationalError as e:
        pass
    
    st.session_state.conversation_context.append({"user": msg, "bot": full_response})
    if len(st.session_state.conversation_context) > 10:
        st.session_state.conversation_context.pop(0)
    
    return full_response, confidence, features_used

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.header("⚙️ Settings")
    
    st.subheader("🎛️ Features")
    st.session_state.web_search_enabled = st.checkbox("🔍 Web Search", value=True)
    st.session_state.file_analysis_enabled = st.checkbox("📁 File Analysis", value=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("📤 File Upload")
    uploaded_file = st.file_uploader("Upload a file (CSV, TXT, etc.)", type=['csv', 'txt', 'json'])
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("📊 Chat Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.artifacts = []
            st.rerun()
    
    with col2:
        if st.button("💾 Clear DB", use_container_width=True):
            cursor.execute("DELETE FROM chat_memory")
            conn.commit()
            st.success("Database cleared!")
    
    if st.button("⬇️ Download History", use_container_width=True):
        cursor.execute("SELECT * FROM chat_memory")
        data = cursor.fetchall()
        if data:
            df = pd.DataFrame(data, columns=["ID", "Timestamp", "User", "Bot", "Confidence", "Features"])
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "chat_history.csv", "text/csv", use_container_width=True)
        else:
            st.info("No history available")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("📈 Statistics")
    cursor.execute("SELECT COUNT(*) FROM chat_memory")
    total_messages = cursor.fetchone()[0]
    st.metric("Total Messages", total_messages)
    st.metric("Artifacts Created", len(st.session_state.artifacts))
    st.metric("Current Session", len(st.session_state.messages))
    st.markdown('</div>', unsafe_allow_html=True)

# ---- MAIN CHAT INTERFACE ----
user_input = st.chat_input("💬 Ask me anything...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    
    with st.spinner("🤔 Thinking..."):
        reply, confidence, features = chatbot_response(user_input, uploaded_file)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "confidence": confidence,
        "features": features,
        "timestamp": datetime.now().strftime("%H:%M")
    })

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**{message['content']}**")
            st.caption(f"🕐 {message['timestamp']}")
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])
            
            conf = message.get("confidence", 0)
            conf_class = "confidence-high" if conf > 0.8 else "confidence-medium" if conf > 0.6 else "confidence-low"
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if message.get("features"):
                    for feature in message["features"]:
                        st.markdown(f'<span class="feature-badge">{feature}</span>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<span class="{conf_class}">Confidence: {conf:.0%}</span>', unsafe_allow_html=True)
            
            st.caption(f"🕐 {message['timestamp']}")

# Display Artifacts
if st.session_state.artifacts:
    st.markdown("---")
    st.subheader("📦 Generated Artifacts")
    
    for artifact in st.session_state.artifacts:
        with st.expander(f"🔹 {artifact['title']} (ID: {artifact['id']})"):
            st.markdown(f"**Type:** {artifact['type']} | **Created:** {artifact['timestamp']}")
            
            if artifact['type'] == 'code':
                st.code(artifact['content'], language='python')
                st.download_button(
                    "⬇️ Download Code",
                    artifact['content'],
                    f"artifact_{artifact['id']}.py",
                    "text/plain"
                )
            else:
                st.text_area("Content", artifact['content'], height=200)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #667eea; font-size: 16px; font-weight: 600;'>
    🤖 <b style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>ChatBot PRO MAX</b> - Enhanced with Claude.ai-like features<br>
    <span style='color: #764ba2;'>🔍 Web Search | 💻 Code Generation | 📄 File Analysis | 📦 Artifacts</span>
</div>
""", unsafe_allow_html=True)
