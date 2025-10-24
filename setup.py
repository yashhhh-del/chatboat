#!/usr/bin/env python3
"""
Quick Setup Script for ChatBot PRO MAX
This script will set up everything you need to run the chatbot
"""

import subprocess
import sys
import os

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")

def run_command(command, description):
    """Run a command and show progress"""
    print(f"⏳ {description}...")
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"✅ {description} - Done!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed!")
        print(f"   Error: {e}")
        return False

def main():
    print_header("🤖 ChatBot PRO MAX - Quick Setup")
    
    # Check Python version
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    else:
        print(f"❌ Python 3.8+ required. You have {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    
    # Check for pip
    print("\n📦 Checking pip...")
    if run_command("pip --version", "Verifying pip"):
        print("✅ pip is installed")
    else:
        print("❌ pip not found. Please install pip first.")
        sys.exit(1)
    
    # Install requirements
    print_header("📥 Installing Dependencies")
    
    packages = [
        "streamlit",
        "tensorflow",
        "nltk",
        "numpy",
        "pandas",
        "requests",
        "beautifulsoup4"
    ]
    
    if os.path.exists("requirements.txt"):
        run_command("pip install -r requirements.txt", "Installing from requirements.txt")
    else:
        for package in packages:
            run_command(f"pip install {package}", f"Installing {package}")
    
    # Download NLTK data
    print_header("📚 Downloading NLTK Data")
    
    try:
        import nltk
        print("⏳ Downloading NLTK punkt...")
        nltk.download('punkt', quiet=True)
        print("✅ punkt downloaded")
        
        print("⏳ Downloading NLTK punkt_tab...")
        nltk.download('punkt_tab', quiet=True)
        print("✅ punkt_tab downloaded")
        
        print("⏳ Downloading NLTK wordnet...")
        nltk.download('wordnet', quiet=True)
        print("✅ wordnet downloaded")
    except Exception as e:
        print(f"⚠️  NLTK data download had issues: {e}")
        print("   The app will try to download on first run.")
    
    # Check for required files
    print_header("📁 Checking Required Files")
    
    required_files = {
        "app_enhanced.py": "Main application",
        "intents.json": "Training data"
    }
    
    all_files_present = True
    for filename, description in required_files.items():
        if os.path.exists(filename):
            print(f"✅ {filename} - {description}")
        else:
            print(f"❌ {filename} - {description} (MISSING!)")
            all_files_present = False
    
    if not all_files_present:
        print("\n⚠️  Some required files are missing!")
        print("   Please ensure app_enhanced.py and intents.json are in this directory.")
    
    # Database check
    print_header("🗄️  Database Setup")
    
    if os.path.exists("chat_memory.db"):
        print("📊 Existing database found: chat_memory.db")
        reset = input("\n⚠️  Do you want to reset the database? (yes/no): ").lower()
        if reset in ['yes', 'y']:
            try:
                os.remove("chat_memory.db")
                print("✅ Database reset!")
            except Exception as e:
                print(f"❌ Could not reset database: {e}")
    else:
        print("✅ No existing database (will be created on first run)")
    
    # Final summary
    print_header("🎉 Setup Complete!")
    
    print("✨ Your ChatBot PRO MAX is ready to launch!")
    print("\n🚀 To start the chatbot, run:")
    print("   \033[1mstreamlit run app_enhanced.py\033[0m")
    
    print("\n💡 Tips:")
    print("   • First run will train the model (takes a few minutes)")
    print("   • Subsequent runs will be instant")
    print("   • Use sidebar to toggle features")
    print("   • Try: 'Write a Python function' or 'Search for AI news'")
    
    print("\n📖 Documentation:")
    print("   • README.md - Full documentation")
    print("   • TROUBLESHOOTING.md - Common issues and solutions")
    print("   • FEATURES.md - Feature comparison")
    
    # Ask if user wants to run now
    print()
    run_now = input("🚀 Would you like to start the chatbot now? (yes/no): ").lower()
    if run_now in ['yes', 'y']:
        print("\n🎬 Launching ChatBot PRO MAX...")
        print("   Press Ctrl+C to stop")
        print()
        try:
            subprocess.run("streamlit run app_enhanced.py", shell=True)
        except KeyboardInterrupt:
            print("\n\n👋 Chatbot stopped. See you next time!")
    else:
        print("\n👋 Setup complete! Run 'streamlit run app_enhanced.py' when ready.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled. You can run this script again anytime!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("   Please check TROUBLESHOOTING.md for help")
