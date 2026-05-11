from flask import Flask, request, render_template, jsonify
import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import json
from datetime import datetime
import nltk

app = Flask(__name__)

# Global variables for model and vectorizer
model = None
vectorizer = None

# History file path
HISTORY_FILE = 'data/history.json'

def load_history():
    """Load prediction history from JSON file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading history: {e}")
    return []

def save_history(history):
    """Save prediction history to JSON file"""
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving history: {e}")

def add_to_history(text, prediction, confidence):
    """Add a prediction to history"""
    history = load_history()
    entry = {
        'id': len(history) + 1,
        'timestamp': datetime.now().isoformat(),
        'text': text,
        'prediction': prediction,
        'confidence': confidence,
        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    history.insert(0, entry)  # Add to beginning for newest first

    # Keep only last 100 entries
    if len(history) > 100:
        history = history[:100]

    save_history(history)
    return entry

def load_or_train_model():
    """Load existing model or train a new one"""
    global model, vectorizer

    model_path = 'models/text_model.joblib'
    vectorizer_path = 'models/vectorizer.joblib'

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            print("Loaded existing model and vectorizer")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")

    # Train new model if no saved model exists
    return train_text_model()

def train_text_model():
    """Train the text classification model"""
    global model, vectorizer

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

        # Calculate accuracy
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Model trained successfully. Test accuracy: {accuracy:.3f}")
        print(classification_report(y_test, y_pred))

        return accuracy

    except Exception as e:
        print(f"Error training model: {e}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Vectorize input
        text_vec = vectorizer.transform([text])

        # Get prediction and probability
        prediction = model.predict(text_vec)[0]
        probabilities = model.predict_proba(text_vec)[0]

        confidence = float(max(probabilities))
        label = 'instruction' if prediction == 1 else 'not instruction'

        # Save to history
        add_to_history(text, label, confidence)

        return jsonify({
            'prediction': label,
            'confidence': confidence,
            'raw_prediction': int(prediction)
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/stats')
def stats():
    try:
        # This would ideally load pre-computed stats, but for now return placeholder
        return jsonify({
            'accuracy': 0.85,  # Placeholder - in real app, load from saved metrics
            'model_type': 'LogisticRegression',
            'vectorizer_type': 'TfidfVectorizer'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def get_history():
    """Get prediction history"""
    try:
        history = load_history()
        return jsonify(history)
    except Exception as e:
        print(f"Error getting history: {e}")
        return jsonify({'error': 'Failed to load history'}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear all prediction history"""
    try:
        save_history([])
        return jsonify({'success': True, 'message': 'History cleared successfully'})
    except Exception as e:
        print(f"Error clearing history: {e}")
        return jsonify({'error': 'Failed to clear history'}), 500

@app.route('/analyze_paragraph', methods=['POST'])
def analyze_paragraph():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not loaded'}), 500

        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')

        sentences = nltk.sent_tokenize(text)
        results = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            text_vec = vectorizer.transform([sentence])
            prediction = model.predict(text_vec)[0]
            probabilities = model.predict_proba(text_vec)[0]
            confidence = float(max(probabilities))
            label = 'instruction' if prediction == 1 else 'not instruction'
            results.append({
                'sentence': sentence,
                'prediction': label,
                'confidence': confidence
            })

        # Add to history
        if len(results) == 1:
            prediction_label = results[0]['prediction']
            confidence_val = results[0]['confidence']
        else:
            prediction_label = f"{len(results)} sentences analyzed"
            confidence_val = sum(r['confidence'] for r in results) / len(results)
        
        add_to_history(text, prediction_label, confidence_val)

        return jsonify(results)

    except Exception as e:
        print(f"Paragraph analysis error: {e}")
        return jsonify({'error': 'Analysis failed'}), 500

if __name__ == '__main__':
    print("Loading or training model...")
    success = load_or_train_model()

    if success:
        print("Starting web server...")
        print("Open http://localhost:5000 in your browser")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load or train model. Exiting.")