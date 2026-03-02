import joblib
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, 'vectorizer.pkl'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

def predict_news(text):
    processed = preprocess_text(text)
    vectorized = vectorizer.transform([processed])
    prediction_num = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    fake_prob = round(probabilities[0] * 100, 2)
    real_prob = round(probabilities[1] * 100, 2)
    prediction = 'REAL' if prediction_num == 1 else 'FAKE'
    confidence = real_prob if prediction == 'REAL' else fake_prob
    return {
        'prediction': prediction,
        'confidence': confidence,
        'real_prob': real_prob,
        'fake_prob': fake_prob,
    }