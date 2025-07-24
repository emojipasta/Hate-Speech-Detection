# model_predictor.py
import joblib

# Load model and vectorizer once
model = joblib.load("/Users/ratulmukherjee/Desktop/Hate Speech Detection/data/best_model_rf.pkl")
vectorizer = joblib.load("/Users/ratulmukherjee/Desktop/Hate Speech Detection/data/tfidf_vectorizer.pkl")

def predict(text):
    vect_text = vectorizer.transform([text])
    label = model.predict(vect_text)[0]
    
    label_map = {
        0: "Hate Speech",
        1: "Offensive Language",
        2: "Neither"
    }
    return label_map[label]