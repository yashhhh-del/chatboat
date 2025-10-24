# 🤖 ChatBot PRO MAX - Claude.ai Enhanced Assistant

An advanced AI chatbot with Claude.ai-inspired features including web search, code generation, file analysis, and artifact creation.

## ✨ Key Features

### 🔍 **Web Search Integration**
- Real-time web search capabilities
- Automatic detection of search queries
- Formatted search results with sources

### 💻 **Code Generation & Artifacts**
- Generate code in Python, JavaScript, HTML, and more
- Downloadable code artifacts
- Syntax highlighting and formatting
- Support for multiple programming languages

### 📄 **File Analysis**
- Upload and analyze CSV, TXT, JSON files
- Automatic data insights and statistics
- Preview and download capabilities
- Pandas-powered data analysis

### 🎨 **Modern UI/UX**
- Claude.ai-inspired interface
- Confidence scoring for responses
- Feature badges showing active capabilities
- Real-time timestamps
- Beautiful, responsive design

### 💬 **Advanced Conversation**
- Context-aware responses
- Conversation history tracking
- Intent classification
- High-confidence predictions

### 📊 **Analytics & Management**
- Chat statistics dashboard
- Export conversation history
- Session management
- Database persistence

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Required Packages

```bash
pip install streamlit tensorflow nltk numpy pandas requests beautifulsoup4
```

### Step 2: Download NLTK Data

The application will automatically download required NLTK data on first run, but you can also do it manually:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
```

### Step 3: File Structure

Ensure your project has this structure:
```
your-project/
│
├── app_enhanced.py          # Main application file
├── intents.json             # Training data for the chatbot
├── trained_chatbot.pkl      # (Generated automatically)
├── training_data.pkl        # (Generated automatically)
└── chat_memory.db           # (Generated automatically)
```

---

## 🎯 Usage

### Starting the Application

```bash
streamlit run app_enhanced.py
```

The app will open in your browser at `http://localhost:8501`

### Basic Interactions

1. **Ask Questions**: Type any question in the chat input
2. **Search the Web**: Ask about current events, weather, news
3. **Generate Code**: Request code in any programming language
4. **Analyze Files**: Upload CSV or TXT files via the sidebar
5. **Download Artifacts**: Save generated code or analysis results

### Example Queries

**General Conversation:**
```
- "Hello, what can you do?"
- "Tell me about yourself"
- "How are you?"
```

**Web Search:**
```
- "What's the latest news on AI?"
- "Search for Python tutorials"
- "Find information about climate change"
```

**Code Generation:**
```
- "Write a Python function to sort a list"
- "Create a JavaScript calculator"
- "Generate HTML for a contact form"
```

**File Analysis:**
```
- "Analyze my data" (upload CSV via sidebar)
- "Check this file" (upload TXT via sidebar)
- "Process my spreadsheet"
```

---

## 🎛️ Configuration

### Sidebar Options

**Features Toggle:**
- 🔍 Web Search - Enable/disable web search
- 📁 File Analysis - Enable/disable file upload analysis

**File Upload:**
- Support for CSV, TXT, JSON files
- Drag and drop or browse to upload

**Chat Management:**
- 🗑️ Clear Chat - Clears current session
- 💾 Clear DB - Clears all history
- ⬇️ Download History - Export as CSV

**Statistics:**
- View total messages
- Track artifacts created
- Monitor current session

---

## 🧠 How It Works

### 1. **Intent Classification**
The chatbot uses a neural network to classify user intents:
- Input text is tokenized and lemmatized
- Bag-of-words representation is created
- Neural network predicts intent with confidence score

### 2. **Feature Detection**
Smart detection of special requests:
- Code generation keywords → Triggers code artifact creation
- Search keywords → Activates web search
- File mentions → Prepares for file analysis

### 3. **Response Generation**
Multi-layered response system:
- Base response from trained model
- Enhanced with web search results
- Augmented with code artifacts
- Enriched with file analysis

### 4. **Artifact Management**
Claude.ai-style artifact system:
- Automatic creation for code and documents
- Version tracking with unique IDs
- Download and reuse capabilities

---

## 🔧 Customization

### Adding New Intents

Edit `intents.json`:

```json
{
  "tag": "your_tag",
  "patterns": ["pattern1", "pattern2"],
  "responses": ["response1", "response2"]
}
```

Then delete `trained_chatbot.pkl` and `training_data.pkl` to retrain.

### Modifying UI

Edit CSS in the `st.markdown()` section at the top of `app_enhanced.py`:

```python
st.markdown("""
<style>
    /* Your custom CSS here */
</style>
""", unsafe_allow_html=True)
```

### Adding Search APIs

Replace the `web_search_simulation()` function with actual API calls:

```python
# Example with SerpApi
from serpapi import GoogleSearch

def web_search_simulation(query):
    params = {
        "q": query,
        "api_key": "YOUR_API_KEY"
    }
    search = GoogleSearch(params)
    return search.get_dict()
```

---

## 📊 Database Schema

SQLite database: `chat_memory.db`

**Table: chat_memory**
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| timestamp | TEXT | Message timestamp |
| user_input | TEXT | User's message |
| bot_response | TEXT | Bot's response |
| confidence | REAL | Confidence score |
| features_used | TEXT | Features activated |

---

## 🐛 Troubleshooting

### Common Issues

**1. Model Training Issues**
```
Error: Could not find training data
Solution: Delete .pkl files and restart to retrain
```

**2. NLTK Download Errors**
```
Error: Resource not found
Solution: Run nltk.download('all') manually
```

**3. Database Locked**
```
Error: Database is locked
Solution: Close all instances and restart
```

**4. Import Errors**
```
Error: No module named 'x'
Solution: pip install [missing-package]
```

---

## 🔒 Privacy & Security

- All data stored **locally** on your machine
- No external data transmission (except search APIs if configured)
- SQLite database for chat history
- Full control over data deletion
- No cloud dependencies

---

## 🎨 UI Features

### Confidence Indicators
- 🟢 **Green** (>80%) - High confidence
- 🟡 **Yellow** (60-80%) - Medium confidence
- 🔴 **Red** (<60%) - Low confidence

### Feature Badges
- **web_search** - Web search was used
- **code_generation** - Code was generated
- **file_analysis** - File was analyzed

### Artifacts
- Unique IDs for tracking
- Timestamps for version control
- Download buttons for each artifact
- Expandable sections for better organization

---

## 🚀 Future Enhancements

Potential additions:
- [ ] Image generation integration
- [ ] Voice input/output
- [ ] Multi-language support
- [ ] Advanced data visualization
- [ ] API endpoint creation
- [ ] User authentication
- [ ] Cloud deployment options
- [ ] Plugin system

---

## 📚 Dependencies

```
streamlit>=1.28.0
tensorflow>=2.13.0
nltk>=3.8.1
numpy>=1.24.0
pandas>=2.0.0
requests>=2.31.0
beautifulsoup4>=4.12.0
```

---

## 📝 License

This project is open-source and available for educational purposes.

---

## 🤝 Contributing

Contributions are welcome! Areas to improve:
- Add more intents to `intents.json`
- Implement real search APIs
- Enhance UI/UX
- Add more file format support
- Improve code generation templates

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code comments
3. Test with simple queries first
4. Ensure all dependencies are installed

---

## 🌟 Acknowledgments

Inspired by:
- Claude.ai by Anthropic
- Streamlit community
- TensorFlow/Keras documentation
- NLTK project

---

**Built with ❤️ using Python, TensorFlow, and Streamlit**

---

## 📖 Quick Start Example

```python
# 1. Install packages
pip install streamlit tensorflow nltk numpy pandas requests beautifulsoup4

# 2. Run the app
streamlit run app_enhanced.py

# 3. In the browser, try these:
# - "Hello, what can you do?"
# - "Write a Python function to calculate factorial"
# - "Search for latest AI news"
# - Upload a CSV file and say "analyze this data"
```

---

**Enjoy your Claude.ai-enhanced chatbot! 🤖✨**
