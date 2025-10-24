"""
Database Reset Utility for ChatBot PRO MAX
Run this script if you encounter database schema errors
"""

import sqlite3
import os

def reset_database():
    """Reset the database to the new schema"""
    
    db_file = 'chat_memory.db'
    
    # Check if database exists
    if os.path.exists(db_file):
        print(f"📁 Found existing database: {db_file}")
        
        # Ask for confirmation
        confirm = input("⚠️  Do you want to reset the database? This will delete all chat history. (yes/no): ")
        
        if confirm.lower() in ['yes', 'y']:
            # Backup old database
            backup_file = f"{db_file}.backup"
            if os.path.exists(backup_file):
                os.remove(backup_file)
            os.rename(db_file, backup_file)
            print(f"✅ Backup created: {backup_file}")
        else:
            print("❌ Database reset cancelled.")
            return
    
    # Create new database with correct schema
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    cursor.execute("""CREATE TABLE IF NOT EXISTS chat_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        user_input TEXT,
        bot_response TEXT,
        confidence REAL,
        features_used TEXT
    )""")
    
    conn.commit()
    conn.close()
    
    print("✅ Database reset successfully!")
    print("📋 New schema:")
    print("   - id (INTEGER PRIMARY KEY)")
    print("   - timestamp (TEXT)")
    print("   - user_input (TEXT)")
    print("   - bot_response (TEXT)")
    print("   - confidence (REAL)")
    print("   - features_used (TEXT)")
    print("\n🚀 You can now run the chatbot with: streamlit run app_enhanced.py")

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 ChatBot PRO MAX - Database Reset Utility")
    print("=" * 60)
    print()
    
    reset_database()
