# Intrusion/Instruction Detection ML Project

This repository provides a minimal scaffold to build an intrusion-detection (or instruction-detection) ML pipeline that can load up to 200 CSV datasets, preprocess them, train a classifier, and evaluate it.

## Features

- **Data Processing**: Load and preprocess CSV datasets with automatic label detection
- **Model Training**: Train RandomForest classifier on tabular data
- **Model Evaluation**: Evaluate trained models with detailed metrics
- **Web Interface**: Interactive web application for real-time instruction detection
- **Text Classification**: TF-IDF based text classification for instruction detection

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train Tabular Model (Original Pipeline)

```bash
# Train on datasets (defaults to current dir, reads up to 200 CSVs)
python train.py --data-dir . --out models/model.joblib --max-files 200

# Evaluate using a saved model on a CSV file
python evaluate.py --model models/model.joblib --data-dir . --max-files 10
```

### 2. Web Interface for Text Classification

```bash
# Start the web application
python app.py
```

Then open http://localhost:5000 in your browser to use the interactive instruction detection interface.

### 4. Test the Model

```bash
# Run prediction tests
python test_model.py
```

## Web Interface Features

- **Real-time Analysis**: Input text instructions and get instant predictions
- **Confidence Scores**: See how confident the model is in its prediction
- **Responsive Design**: Works on desktop and mobile devices
- **Visual Feedback**: Color-coded results (green for instructions, red for non-instructions)
- **Model Statistics**: View training accuracy and model information
- **Prediction History**: View, search, and filter all previous predictions
- **History Management**: Clear history, filter by prediction type, search through past analyses

## Project Structure

```
├── train.py              # Main training script for tabular data
├── evaluate.py           # Model evaluation script
├── app.py                # Web interface for text classification
├── convert_instruction_file.py  # Template conversion utility
├── instruction_detection_large.csv  # Dataset file
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── src/
    ├── data_loader.py   # CSV loading utilities
    ├── preprocess.py    # Data preprocessing
    └── model.py         # Model training and loading
```

## Notes

- The loader concatenates CSVs found recursively under `--data-dir` (up to `--max-files`).
- The preprocessing assumes a label column named `label`, `Label`, `target`, or `class`. If none are present, it uses the last column as label.
- The web interface uses TF-IDF vectorization and LogisticRegression for text classification.
- The example pipeline uses a `RandomForestClassifier` for tabular data. Swap to other models in `src/model.py` if desired.

## Next steps

- Add dataset-specific feature engineering and balancing for class imbalance.
- Add cross-validation, hyperparameter search, and logging.
- Optionally create a notebook demonstrating EDA on `instruction_detection_large.csv`.
