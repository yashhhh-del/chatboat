# chatbot_full.py — Complete NLP Chatbot with Memory & Database Support
# Compatible with Python 3.10+, TensorFlow 2.16+, and NLTK 3.9+

import random
import json
import sqlite3
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Step 1: Setup NLTK tools
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Step 2: Load intents
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Step 3: Data preprocessing
words, classes, documents = [], [], []
ignore_symbols = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        tokenized = nltk.word_tokenize(pattern)
        words.extend(tokenized)
        documents.append((tokenized, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_symbols]))
classes = sorted(set(classes))

# Step 4: Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    for w in words:
        bag.append(1 if w in pattern_words else 0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
train_x = np.array([x[0] for x in training])
train_y = np.array([x[1] for x in training])

# Step 5: Model creation
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.01),
              metrics=['accuracy'])

# Step 6: Train model
print("Training model... please wait ⏳")
model.fit(train_x, train_y, epochs=150, batch_size=8, verbose=1)
model.save('chatbot_model.h5')

pickle.dump({'words': words, 'classes': classes}, open('training_data.pkl', 'wb'))
print("Model training complete ✅")

# Step 7: SQLite setup
conn = sqlite3.connect('chatbot_memory.db')
cursor = conn.cursor()
cursor.execute('CREATE TABLE IF NOT EXISTS chat_memory (user_input TEXT, bot_response TEXT)')
conn.commit()

# Step 8: Helper functions
def clean_text(sentence):
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in tokens]

def bag_of_words(sentence):
    tokens = clean_text(sentence)
    bag = [1 if w in tokens else 0 for w in words]
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    preds = model.predict(np.array([bow]))[0]
    threshold = 0.25
    results = [[i, p] for i, p in enumerate(preds) if p > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return classes[results[0][0]] if results else None

def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm sorry, could you rephrase that?"

def chatbot_response(message):
    tag = predict_class(message)
    response = get_response(tag) if tag else "Sorry, I didn't understand that."
    cursor.execute("INSERT INTO chat_memory (user_input, bot_response) VALUES (?, ?)", (message, response))
    conn.commit()
    return response

# Step 9: Chat interface
print("\nChatbot ready! Type 'quit' to exit 🗨️")
while True:
    user_message = input("You: ")
    if user_message.lower() == 'quit':
        print("Bot: Goodbye 👋")
        break
    print("Bot:", chatbot_response(user_message))
