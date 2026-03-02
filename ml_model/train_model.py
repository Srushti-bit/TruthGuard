import pandas as pd
import numpy as np
import re
import joblib
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading dataset...")
fake_df = pd.read_csv(os.path.join(BASE_DIR, 'Fake.csv'))
true_df = pd.read_csv(os.path.join(BASE_DIR, 'True.csv'))

fake_df['label'] = 0
true_df['label'] = 1

df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset size: {len(df)} articles")
print(f"Fake: {len(fake_df)} | Real: {len(true_df)}")

df['content'] = df['title'] + ' ' + df['text']

print("Preprocessing text...")
df['processed'] = df['content'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['processed'], df['label'], test_size=0.2, random_state=42
)

print("Vectorizing...")
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

joblib.dump(model, os.path.join(BASE_DIR, 'model.pkl'))
joblib.dump(vectorizer, os.path.join(BASE_DIR, 'vectorizer.pkl'))
print("\n✅ Model saved to ml_model/model.pkl")
print("✅ Vectorizer saved to ml_model/vectorizer.pkl")