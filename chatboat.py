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

# Set theme colors
st.markdown("""
<style>
    /* Force all text to be visible */
    * {
        color: #1f2937;
    }
    
    /* Streamlit default text */
    .stMarkdown, .stText {
        color: #1f2937 !important;
    }
    
    /* Chat input */
    textarea {
        color: #1f2937 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---- CUSTOM CSS FOR CLAUDE-LIKE UI ----
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Chat messages - Force dark text */
    .stChatMessage {
        background-color: white !important;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* All text in chat must be dark */
    .stChatMessage p, 
    .stChatMessage span,
    .stChatMessage div,
    .stChatMessage strong,
    .stChatMessage em,
    .stChatMessage code {
        color: #111827 !important;
    }
    
    /* Markdown in chat */
    .stChatMessage [data-testid="stMarkdownContainer"] * {
        color: #111827 !important;
    }
    
    /* Avatar colors */
    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        background-color: #3b82f6 !important;
    }
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background-color: #10a37f !important;
    }
    
    .artifact-container {
        background-color: #ffffff;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .artifact-container * {
        color: #111827 !important;
    }
    
    .feature-badge {
        background-color: #10a37f;
        color: white !important;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin: 4px;
    }
    
    .confidence-high {
        color: #10a37f !important;
        font-weight: 600;
    }
    .confidence-medium {
        color: #f59e0b !important;
        font-weight: 600;
    }
    .confidence-low {
        color: #ef4444 !important;
        font-weight: 600;
    }
    
    .search-result {
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 12px;
        margin: 8px 0;
        border-radius: 6px;
        color: #111827 !important;
    }
    
    .sidebar-section {
        background-color: white;
        padding: 16px;
        border-radius: 8px;
        margin: 12px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Input field */
    .stChatInput textarea {
        color: #111827 !important;
    }
    
    /* Caption text */
    [data-testid="stCaptionContainer"] {
        color: #6b7280 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Smart ChatBot PRO MAX – Claude-Powered Assistant")
st.markdown("### *Advanced AI with Web Search, File Analysis & Code Generation*")

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

# Check if old table exists and migrate
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_memory'")
table_exists = cursor.fetchone()

if table_exists:
    # Check table structure
    cursor.execute("PRAGMA table_info(chat_memory)")
    columns = [col[1] for col in cursor.fetchall()]
    
    # If old structure (only 2 columns), migrate data
    if len(columns) == 2:
        # Backup old data
        cursor.execute("ALTER TABLE chat_memory RENAME TO chat_memory_old")
        
        # Create new table
        cursor.execute("""CREATE TABLE chat_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_input TEXT,
            bot_response TEXT,
            confidence REAL,
            features_used TEXT
        )""")
        
        # Migrate old data
        cursor.execute("""
            INSERT INTO chat_memory (timestamp, user_input, bot_response, confidence, features_used)
            SELECT datetime('now'), user_input, bot_response, 0.0, ''
            FROM chat_memory_old
        """)
        
        # Drop old table
        cursor.execute("DROP TABLE chat_memory_old")
        conn.commit()
        print("✅ Database migrated to new schema")
else:
    # Create new table
    cursor.execute("""CREATE TABLE IF NOT EXISTS chat_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        user_input TEXT,
        bot_response TEXT,
        confidence REAL,
        features_used TEXT
    )""")
    conn.commit()

# ---- SESSION STATE INITIALIZATION ----
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
    """Simulated web search (replace with actual API like SerpApi, Brave, etc.)"""
    try:
        # This is a simulation - in production, use actual search APIs
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
    """Detect if user is asking for code"""
    code_keywords = ['code', 'program', 'script', 'function', 'write', 'create', 'develop', 
                     'python', 'javascript', 'html', 'css', 'algorithm', 'implement']
    return any(keyword in text.lower() for keyword in code_keywords)

def detect_web_search_request(text):
    """Detect if user needs web search"""
    search_keywords = ['search', 'find', 'look up', 'what is', 'who is', 'latest', 
                       'current', 'news', 'weather', 'price', 'information about']
    return any(keyword in text.lower() for keyword in search_keywords)

def detect_file_request(text):
    """Detect if user is asking about files"""
    file_keywords = ['file', 'document', 'upload', 'analyze', 'read', 'csv', 'excel', 
                     'pdf', 'data', 'spreadsheet']
    return any(keyword in text.lower() for keyword in file_keywords)

def generate_code_artifact(description, language="python"):
    """Generate code based on description"""
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
""",
        "html": f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{description}</title>
</head>
<body>
    <h1>{description}</h1>
    <p>Generated HTML template</p>
</body>
</html>
"""
    }
    return templates.get(language, templates["python"])

def create_artifact(content, artifact_type="code", title="Generated Artifact"):
    """Create an artifact like Claude.ai"""
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
    """Analyze uploaded files"""
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
    return "I'm not sure about that. Could you rephrase or ask something else?"

def chatbot_response(msg, uploaded_file=None):
    """Enhanced chatbot response with multiple features"""
    features_used = []
    additional_content = ""
    
    # Check for web search
    if st.session_state.web_search_enabled and detect_web_search_request(msg):
        search_results = web_search_simulation(msg)
        features_used.append("web_search")
        additional_content += f"\n\n🔍 **Web Search Results:**\n{search_results.get('snippet', '')}"
    
    # Check for code generation
    if detect_code_request(msg):
        features_used.append("code_generation")
        code = generate_code_artifact(msg)
        artifact = create_artifact(code, "code", f"Code for: {msg[:50]}")
        additional_content += f"\n\n💻 **Generated Code Artifact (ID: {artifact['id']})**"
    
    # File analysis
    if uploaded_file and st.session_state.file_analysis_enabled:
        features_used.append("file_analysis")
        file_analysis, df = analyze_file_content(uploaded_file)
        additional_content += f"\n\n📄 **File Analysis:**\n{file_analysis}"
    
    # Get base response from model
    tag, confidence = predict_class(msg)
    
    # Safety check for confidence
    if confidence is None:
        confidence = 0.0
    
    base_response = get_response(tag) if tag else "I'm not sure about that. Could you rephrase?"
    
    # Combine responses
    full_response = base_response + additional_content
    
    # Store in database with proper error handling
    try:
        cursor.execute(
            "INSERT INTO chat_memory (timestamp, user_input, bot_response, confidence, features_used) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg, full_response, float(confidence), ", ".join(features_used))
        )
        conn.commit()
    except sqlite3.OperationalError as e:
        # If there's still a schema issue, just log the error and continue
        print(f"Database error: {e}")
        pass
    
    # Update context
    st.session_state.conversation_context.append({"user": msg, "bot": full_response})
    if len(st.session_state.conversation_context) > 10:
        st.session_state.conversation_context.pop(0)
    
    return full_response, confidence, features_used

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.header("⚙️ Settings")
    
    st.subheader("🎛️ Features")
    st.session_state.web_search_enabled = st.toggle("🔍 Web Search", value=True)
    st.session_state.file_analysis_enabled = st.toggle("📁 File Analysis", value=True)
    
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
    
    # Statistics
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("📈 Statistics")
    cursor.execute("SELECT COUNT(*) FROM chat_memory")
    total_messages = cursor.fetchone()[0]
    st.metric("Total Messages", total_messages)
    st.metric("Artifacts Created", len(st.session_state.artifacts))
    st.metric("Current Session", len(st.session_state.messages))
    st.markdown('</div>', unsafe_allow_html=True)

# ---- MAIN CHAT INTERFACE ----
user_input = st.chat_input("💬 Ask me anything... (I can search the web, generate code, analyze files!)")

if user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    
    # Get bot response
    with st.spinner("🤔 Thinking..."):
        reply, confidence, features = chatbot_response(user_input, uploaded_file)
    
    # Add bot message
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
            
            # Show confidence and features
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

# Display Artifacts Section
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
<div style='text-align: center; color: #6b7280; font-size: 14px;'>
    🤖 <b>ChatBot PRO MAX</b> - Enhanced with Claude.ai-like features<br>
    Features: Web Search 🔍 | Code Generation 💻 | File Analysis 📄 | Artifacts 📦
</div>
""", unsafe_allow_html=True)
