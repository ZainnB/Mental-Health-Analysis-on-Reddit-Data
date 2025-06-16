import pandas as pd
import numpy as np
import string
import re
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import torch

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# === CONFIG ===
INPUT_FILE = 'reddit_mental_health_posts.xlsx'
OUTPUT_FILE = 'reddit_with_sentiment_emotion.xlsx'

# === Load Data ===
df = pd.read_excel(INPUT_FILE)
df['created_utc'] = pd.to_datetime(df['created_utc'], errors='coerce')
df = df.dropna(subset=['created_utc'])
df['year'] = df['created_utc'].dt.year
df['month'] = df['created_utc'].dt.month
df['day'] = df['created_utc'].dt.date
df['year_month'] = df['created_utc'].dt.to_period('M')

# === Clean Text ===
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['text'].astype(str).apply(clean_text)
df['full_text'] = df['title'].fillna('') + ' ' + df['clean_text'].fillna('')

# === Sentiment Analysis ===
print("🔍 Loading sentiment model...")
sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)

def get_sentiment(text):
    try:
        return sentiment_pipeline(text[:512])[0]['label'].lower()
    except:
        return 'neutral'

print("📊 Analyzing sentiment...")
tqdm.pandas(desc="Sentiment")
df['sentiment'] = df['full_text'].progress_apply(get_sentiment)

# === Emotion Detection ===
print("🔍 Loading emotion model...")
emotion_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
emotion_model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
emotion_pipeline = pipeline("text-classification", model=emotion_model, tokenizer=emotion_tokenizer, top_k=1)

def get_emotion(text):
    try:
        return emotion_pipeline(text[:512])[0][0]['label'].lower()
    except:
        return 'neutral'

print("📊 Detecting emotion...")
tqdm.pandas(desc="Emotion")
df['emotion'] = df['full_text'].progress_apply(get_emotion)

# === Export Final Data ===
df.to_excel(OUTPUT_FILE, index=False)
print(f"✅ All processing complete. Output saved to: {OUTPUT_FILE}")
