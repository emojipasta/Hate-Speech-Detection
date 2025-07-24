# scripts/hate_speech_pipeline.py

import pandas as pd
import numpy as np
import re
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# -------------------------
# Paths
# -------------------------
DATA_PATH = "data/hate_speech.csv"
CLEANED_DATA_PATH = "data/cleaned_data.csv"
MODEL_PATH = "data/best_model_rf.pkl"
VECTORIZER_PATH = "data/tfidf_vectorizer.pkl"

# -------------------------
# Text Cleaning Function
# -------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

# -------------------------
# Loading and Cleaning Data
# -------------------------
def load_and_clean_data():
    df = pd.read_csv(DATA_PATH)
    df['clean_tweet'] = df['tweet'].apply(clean_text)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    return df

# -------------------------
# Vectorizing
# -------------------------
def vectorize_text(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_tweet'].fillna(""))
    joblib.dump(vectorizer, VECTORIZER_PATH)
    return X, vectorizer

# -------------------------
# Training Model
# -------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Saving model
    joblib.dump(model, MODEL_PATH)

    # Evaluating
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("✅ Accuracy:", acc)
    print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))
    print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------
# Pipeline Runner
# -------------------------
def run_pipeline():
    print("🚀 Starting Hate Speech Detection Pipeline...")
    df = load_and_clean_data()
    print("✅ Data cleaned and saved.")

    X, vectorizer = vectorize_text(df)
    print("✅ Text vectorized and vectorizer saved.")

    y = df['class']
    train_model(X, y)
    print("✅ Model trained and saved to:", MODEL_PATH)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    run_pipeline()
