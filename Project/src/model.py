from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os


def build_pipeline(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)),
    ])


def train_and_save(X, y, out_path: str, test_size: float = 0.2, seed: int = 42):
    # Validate inputs
    if len(X) == 0:
        raise ValueError('Empty feature matrix')
    if y.isna().any():
        raise ValueError('Label contains NaN values')
    
    # Use stratify only if there are enough samples per class
    stratify_param = y if len(y.unique()) > 1 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify_param)
    except ValueError as e:
        # Fall back to non-stratified split if stratification fails
        print(f'Warning: Stratified split failed ({e}), using regular split')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    
    pipe = build_pipeline(seed=seed)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    report = classification_report(y_test, preds)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(pipe, out_path)
    return report


def load_model(path: str):
    return joblib.load(path)
