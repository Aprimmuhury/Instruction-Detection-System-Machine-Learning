import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv('dataset.csv') # Ensure you have sentence and label columns

# 2. Preprocess
X = df['sentence']
y = df['label'].map({'instruction': 1, 'not': 0})

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train classifier
clf = LogisticRegression()
clf.fit(X_train_vec, y_train)

# 6. Evaluate
y_pred = clf.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# 7. Inference
def predict_instruction(text):
    vec = vectorizer.transform([text])
    pred = clf.predict(vec)[0]
    return "instruction" if pred == 1 else "not instruction"

print(predict_instruction("Please open the window."))