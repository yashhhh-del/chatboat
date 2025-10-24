# 🔧 Troubleshooting Guide - ChatBot PRO MAX

## Common Errors and Solutions

### ❌ Error 1: Database Schema Mismatch

**Error Message:**
```
sqlite3.OperationalError: table chat_memory has 2 columns but 5 values were supplied
```

**Cause:** Your old database has a different structure than the new enhanced version.

**Solutions:**

#### Option 1: Automatic Migration (Recommended)
The enhanced app now includes automatic migration. Just restart the app:
```bash
streamlit run app_enhanced.py
```

#### Option 2: Manual Database Reset
Use the provided reset script:
```bash
python reset_database.py
```

#### Option 3: Delete Old Database
Simply delete the old database file (you'll lose chat history):
```bash
rm chat_memory.db
```
Then restart the app.

---

### ❌ Error 2: Module Not Found

**Error Message:**
```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solution:**
Install all required dependencies:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install streamlit tensorflow nltk numpy pandas requests beautifulsoup4
```

---

### ❌ Error 3: NLTK Data Not Found

**Error Message:**
```
LookupError: Resource punkt not found
```

**Solution:**
The app will auto-download NLTK data, but if it fails, manually download:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
```

Or download all:
```python
import nltk
nltk.download('all')
```

---

### ❌ Error 4: Port Already in Use

**Error Message:**
```
OSError: [Errno 98] Address already in use
```

**Solution:**
Kill the existing Streamlit process:
```bash
# Find the process
lsof -i :8501

# Kill it
kill -9 <PID>

# Or use a different port
streamlit run app_enhanced.py --server.port 8502
```

---

### ❌ Error 5: TensorFlow GPU Issues

**Error Message:**
```
Could not load dynamic library 'libcudart.so.11.0'
```

**Solution:**
This is just a warning. TensorFlow will use CPU instead. To suppress:
```bash
export TF_CPP_MIN_LOG_LEVEL=2
streamlit run app_enhanced.py
```

Or install CPU-only version:
```bash
pip uninstall tensorflow
pip install tensorflow-cpu
```

---

### ❌ Error 6: Model Training Takes Too Long

**Issue:** First run takes several minutes

**Solution:**
This is normal for first run. The model is being trained and will be cached. Subsequent runs will be instant.

To speed up:
- The trained model is saved as `trained_chatbot.pkl`
- The training data is saved as `training_data.pkl`
- Don't delete these files unless you change intents.json

---

### ❌ Error 7: Confidence Always Shows 0%

**Cause:** Database migration issue or prediction error

**Solution:**
1. Clear the model cache:
```bash
rm trained_chatbot.pkl training_data.pkl
```

2. Reset database:
```bash
python reset_database.py
```

3. Restart app:
```bash
streamlit run app_enhanced.py
```

---

### ❌ Error 8: File Upload Not Working

**Issue:** Uploaded files not being processed

**Solutions:**
1. Check file format (CSV, TXT, JSON only)
2. Ensure file size is reasonable (<200MB)
3. Check sidebar toggle "📁 File Analysis" is ON
4. Try re-uploading the file

---

### ❌ Error 9: Artifacts Not Appearing

**Issue:** Generated code not showing in artifacts section

**Solutions:**
1. Check if code keywords are in your message
2. Verify artifacts in session state:
   - Refresh the page
   - Check at bottom of chat interface
3. Try explicit request: "Write a Python function..."

---

### ❌ Error 10: Web Search Not Working

**Issue:** Web search toggle doesn't seem to work

**Current Status:** 
- Web search is **simulated** by default
- To enable real search, you need to integrate an API

**To Integrate Real Search:**
1. Get API key from SerpApi, Brave, or Google
2. Edit the `web_search_simulation()` function in app_enhanced.py
3. Replace simulation with actual API calls

Example with SerpApi:
```python
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

## 🚀 Quick Fixes

### Reset Everything
```bash
# Delete all generated files
rm chat_memory.db trained_chatbot.pkl training_data.pkl

# Restart fresh
streamlit run app_enhanced.py
```

### Check Installation
```bash
# Verify Python version (should be 3.8+)
python --version

# Verify all packages
pip list | grep -E "streamlit|tensorflow|nltk|pandas"

# Check Streamlit
streamlit --version
```

### Test Basic Functionality
```python
# Test imports
python -c "import streamlit; import tensorflow; import nltk; print('All imports OK!')"
```

---

## 📞 Still Having Issues?

1. **Check the README.md** for detailed setup instructions
2. **Review the code** - all functions are well-commented
3. **Start with original app.py** - if enhanced version has issues, fall back to original
4. **Check system requirements:**
   - Python 3.8 or higher
   - 4GB RAM minimum
   - Internet connection (for first-time setup)

---

## 🔍 Debug Mode

To run in debug mode:
```bash
streamlit run app_enhanced.py --logger.level=debug
```

This will show detailed logs of what's happening.

---

## 📊 Verify Database Structure

Check if your database has the correct structure:
```python
import sqlite3

conn = sqlite3.connect('chat_memory.db')
cursor = conn.cursor()

# Check table structure
cursor.execute("PRAGMA table_info(chat_memory)")
columns = cursor.fetchall()

print("Database columns:")
for col in columns:
    print(f"  {col[1]} ({col[2]})")

conn.close()
```

**Expected output:**
```
Database columns:
  id (INTEGER)
  timestamp (TEXT)
  user_input (TEXT)
  bot_response (TEXT)
  confidence (REAL)
  features_used (TEXT)
```

---

## ✅ Verification Checklist

Before reporting issues, verify:
- [ ] Python 3.8+ installed
- [ ] All packages from requirements.txt installed
- [ ] intents.json file exists in same folder
- [ ] No other Streamlit apps running on port 8501
- [ ] Database reset if migrating from old version
- [ ] NLTK data downloaded
- [ ] Sufficient disk space for model files

---

## 🎯 Common User Mistakes

1. **Not using the enhanced file** - Make sure you're running `app_enhanced.py`, not the original `app.py`
2. **Missing intents.json** - Both files must be in the same directory
3. **Wrong Python version** - Use Python 3.8 or higher
4. **Not waiting for training** - First run takes time, be patient
5. **Forgetting to toggle features** - Check sidebar settings

---

**If all else fails, start fresh with a clean installation! 🔄**
