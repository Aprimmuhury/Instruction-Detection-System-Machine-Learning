from flask import Flask, request, render_template_string, jsonify
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

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Instruction Detection System</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .tabs {
            display: flex;
            margin-bottom: 30px;
            border-bottom: 2px solid #eee;
        }
        .tab {
            flex: 1;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            background: #f8f9fa;
            border: none;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s;
            position: relative;
        }
        .tab.active {
            background: #667eea;
            color: white;
        }
        .tab:hover:not(.active) {
            background: #e9ecef;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .input-group {
            margin-bottom: 20px;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            resize: vertical;
            min-height: 100px;
            transition: border-color 0.3s;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .btn-secondary {
            background: #6c757d;
        }
        .btn-danger {
            background: #dc3545;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 8px;
            display: none;
        }
        .instruction {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
        }
        .not-instruction {
            background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
            color: white;
        }
        .confidence {
            font-size: 14px;
            opacity: 0.9;
        }
        .stats {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .loading {
            display: none;
            text-align: center;
            color: #667eea;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-right: 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* History styles */
        .history-controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .search-box {
            flex: 1;
            min-width: 200px;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 14px;
        }
        .filter-buttons {
            display: flex;
            gap: 5px;
        }
        .filter-btn {
            padding: 8px 15px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        .filter-btn.active {
            background: #667eea;
            color: white;
        }
        .filter-btn:hover {
            background: #667eea;
            color: white;
        }
        .history-list {
            max-height: 600px;
            overflow-y: auto;
            border: 1px solid #eee;
            border-radius: 8px;
        }
        .history-item {
            padding: 15px;
            border-bottom: 1px solid #eee;
            transition: background-color 0.2s;
        }
        .history-item:hover {
            background-color: #f8f9fa;
        }
        .history-item:last-child {
            border-bottom: none;
        }
        .history-text {
            font-weight: bold;
            margin-bottom: 8px;
            word-wrap: break-word;
        }
        .history-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 14px;
            color: #666;
        }
        .history-prediction {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .history-prediction.instruction {
            background: #d4edda;
            color: #155724;
        }
        .history-prediction.not-instruction {
            background: #f8d7da;
            color: #721c24;
        }
        .no-history {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        .clear-history-btn {
            margin-left: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Instruction Detection System</h1>

        <div class="tabs">
            <button class="tab active" onclick="showTab('analyzer')">Analyzer</button>
            <button class="tab" onclick="showTab('history')">History</button>
        </div>

        <div id="analyzer" class="tab-content active">
            <div class="input-group">
                <label for="instruction-input" style="display: block; margin-bottom: 10px; font-weight: bold; color: #333;">
                    Enter your instruction:
                </label>
                <textarea
                    id="instruction-input"
                    placeholder="Type your instruction here... (e.g., 'Please open the window' or 'The weather is nice today')"
                    maxlength="500"
                ></textarea>
            </div>

            <button onclick="analyzeInstruction()">Analyze Instruction</button>

            <div id="loading" class="loading">
                <div class="spinner"></div>
                Analyzing...
            </div>

            <div id="result" class="result">
                <h3>Analysis Result:</h3>
                <p id="prediction"></p>
                <p id="confidence" class="confidence"></p>
            </div>

            <div class="stats">
                <h3>Model Statistics</h3>
                <p><strong>Training Accuracy:</strong> <span id="train-accuracy">Loading...</span></p>
                <p><strong>Model Type:</strong> Logistic Regression with TF-IDF</p>
                <p><strong>Features:</strong> Text vectorization using TF-IDF</p>
            </div>
        </div>

        <div id="history" class="tab-content">
            <div class="history-controls">
                <input type="text" id="search-input" class="search-box" placeholder="Search history..." onkeyup="filterHistory()">
                <div class="filter-buttons">
                    <button class="filter-btn active" onclick="setFilter('all')">All</button>
                    <button class="filter-btn" onclick="setFilter('instruction')">Instructions</button>
                    <button class="filter-btn" onclick="setFilter('not-instruction')">Not Instructions</button>
                </div>
                <button class="btn-danger clear-history-btn" onclick="clearHistory()">Clear History</button>
            </div>

            <div id="history-list" class="history-list">
                <div class="no-history">
                    <p>No prediction history yet. Try analyzing some instructions first!</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentFilter = 'all';
        let historyData = [];

        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });

            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');

            // Load history if switching to history tab
            if (tabName === 'history') {
                loadHistory();
            }
        }

        async function analyzeInstruction() {
            const input = document.getElementById('instruction-input').value.trim();
            if (!input) {
                alert('Please enter an instruction to analyze.');
                return;
            }

            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const button = document.querySelector('#analyzer button');

            // Show loading
            loading.style.display = 'block';
            result.style.display = 'none';
            button.disabled = true;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: input })
                });

                const data = await response.json();

                // Hide loading
                loading.style.display = 'none';
                result.style.display = 'block';
                button.disabled = false;

                // Display result
                const predictionElement = document.getElementById('prediction');
                const confidenceElement = document.getElementById('confidence');

                if (data.prediction === 'instruction') {
                    result.className = 'result instruction';
                    predictionElement.innerHTML = '<strong>✅ This is an INSTRUCTION</strong>';
                } else {
                    result.className = 'result not-instruction';
                    predictionElement.innerHTML = '<strong>❌ This is NOT an instruction</strong>';
                }

                confidenceElement.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

                // Clear input after successful analysis
                document.getElementById('instruction-input').value = '';

                // Refresh history if history tab is active
                if (document.getElementById('history').classList.contains('active')) {
                    loadHistory();
                }

            } catch (error) {
                loading.style.display = 'none';
                button.disabled = false;
                alert('Error analyzing instruction. Please try again.');
                console.error('Error:', error);
            }
        }

        async function loadHistory() {
            try {
                const response = await fetch('/history');
                historyData = await response.json();
                displayHistory(historyData);
            } catch (error) {
                console.error('Error loading history:', error);
                document.getElementById('history-list').innerHTML = '<div class="no-history"><p>Error loading history.</p></div>';
            }
        }

        function displayHistory(history) {
            const historyList = document.getElementById('history-list');

            if (!history || history.length === 0) {
                historyList.innerHTML = '<div class="no-history"><p>No prediction history yet. Try analyzing some instructions first!</p></div>';
                return;
            }

            const filteredHistory = history.filter(item => {
                const matchesFilter = currentFilter === 'all' || item.prediction === currentFilter;
                const matchesSearch = !document.getElementById('search-input').value ||
                    item.text.toLowerCase().includes(document.getElementById('search-input').value.toLowerCase());
                return matchesFilter && matchesSearch;
            });

            if (filteredHistory.length === 0) {
                historyList.innerHTML = '<div class="no-history"><p>No items match your current filters.</p></div>';
                return;
            }

            const historyHtml = filteredHistory.map(item => `
                <div class="history-item">
                    <div class="history-text">"${item.text}"</div>
                    <div class="history-meta">
                        <span class="history-prediction ${item.prediction.replace(' ', '-')}">${item.prediction.toUpperCase()}</span>
                        <span>Confidence: ${(item.confidence * 100).toFixed(1)}%</span>
                        <span>${item.datetime}</span>
                    </div>
                </div>
            `).join('');

            historyList.innerHTML = historyHtml;
        }

        function setFilter(filter) {
            currentFilter = filter;
            document.querySelectorAll('.filter-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
            displayHistory(historyData);
        }

        function filterHistory() {
            displayHistory(historyData);
        }

        async function clearHistory() {
            if (confirm('Are you sure you want to clear all prediction history? This action cannot be undone.')) {
                try {
                    const response = await fetch('/clear_history', {
                        method: 'POST'
                    });

                    if (response.ok) {
                        historyData = [];
                        displayHistory(historyData);
                        alert('History cleared successfully!');
                    } else {
                        alert('Error clearing history.');
                    }
                } catch (error) {
                    console.error('Error clearing history:', error);
                    alert('Error clearing history.');
                }
            }
        }

        // Load model stats on page load
        window.onload = async function() {
            try {
                const response = await fetch('/stats');
                const data = await response.json();
                document.getElementById('train-accuracy').textContent = `${(data.accuracy * 100).toFixed(1)}%`;
            } catch (error) {
                document.getElementById('train-accuracy').textContent = 'N/A';
            }
        };

        // Allow Enter key to submit
        document.getElementById('instruction-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                analyzeInstruction();
            }
        });
    </script>
</body>
</html>
"""

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
    return render_template_string(HTML_TEMPLATE)

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

if __name__ == '__main__':
    print("Loading or training model...")
    success = load_or_train_model()

    if success:
        print("Starting web server...")
        print("Open http://localhost:5000 in your browser")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load or train model. Exiting.")