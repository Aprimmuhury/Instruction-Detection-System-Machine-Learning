#!/usr/bin/env python3
"""
Test script for the instruction detection model
"""
import sys
import os
import joblib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split

def load_or_train_model():
    """Load existing model or train a new one"""
    model_path = 'models/text_model.joblib'
    vectorizer_path = 'models/vectorizer.joblib'

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            print("Loaded existing model and vectorizer")
            return model, vectorizer
        except Exception as e:
            print(f"Error loading model: {e}")

    # Train new model if no saved model exists
    return train_text_model()

def train_text_model():
    """Train the text classification model"""
    try:
        # Try to load the instruction detection dataset
        if os.path.exists('instruction_detection_large.csv'):
            df = pd.read_csv('instruction_detection_large.csv')
            if 'sentence' in df.columns and 'label' in df.columns:
                X = df['sentence']
                y = df['label'].map({'instruction': 1, 'not instruction': 0})
            else:
                raise ValueError("Dataset doesn't have required columns")
        else:
            # Create sample data if no dataset exists
            print("No dataset found, creating sample data...")
            instructions = [
                "Please open the window",
                "Could you close the door",
                "Turn on the lights",
                "Please help me with this",
                "Can you show me the way",
                "Would you mind passing the salt",
                "The weather is beautiful today",
                "I like this restaurant",
                "The movie was great",
                "This book is interesting",
                "Please bring me a glass of water",
                "Can you call the doctor",
                "Help me carry this bag",
                "Show me your phone",
                "The sky is blue",
                "I enjoy reading",
                "Music sounds nice",
                "The food tastes good"
            ]
            labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]

            X = instructions
            y = labels

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Vectorize
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_vec, y_train)

        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/text_model.joblib')
        joblib.dump(vectorizer, 'models/vectorizer.joblib')

        print(f"Model trained successfully.")
        return model, vectorizer

    except Exception as e:
        print(f"Error training model: {e}")
        return None, None

def test_predictions():
    """Test the model with sample inputs"""
    model, vectorizer = load_or_train_model()

    if model is None or vectorizer is None:
        print("Failed to load or train model")
        return

    test_cases = [
        "Please open the window",
        "Could you close the door",
        "Turn on the lights",
        "The weather is beautiful today",
        "I like this restaurant",
        "Please bring me a glass of water",
        "Can you call the doctor",
        "The sky is blue",
        "Help me carry this bag",
        "This book is interesting"
    ]

    print("Testing Instruction Detection Model:")
    print("=" * 50)

    for text in test_cases:
        # Vectorize input
        text_vec = vectorizer.transform([text])

        # Get prediction and probability
        prediction = model.predict(text_vec)[0]
        probabilities = model.predict_proba(text_vec)[0]

        confidence = float(max(probabilities))
        label = 'instruction' if prediction == 1 else 'not instruction'

        status = "✅" if (text in ["Please open the window", "Could you close the door", "Turn on the lights", "Please bring me a glass of water", "Can you call the doctor", "Help me carry this bag"] and label == 'instruction') or (text in ["The weather is beautiful today", "I like this restaurant", "The sky is blue", "This book is interesting"] and label == 'not instruction') else "❌"

        print(f"{status} '{text}' -> {label.upper()} ({confidence:.1%})")

if __name__ == "__main__":
    test_predictions()