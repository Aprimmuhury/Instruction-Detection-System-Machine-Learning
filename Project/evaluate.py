import argparse
from src.preprocess import preprocess
from src.model import load_model
from src.data_loader import load_from_dir, find_csv_files
from sklearn.metrics import classification_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Path to model.joblib')
    p.add_argument('--data-dir', default='.', help='Directory with CSV test files')
    p.add_argument('--max-files', type=int, default=50)
    args = p.parse_args()

    model = load_model(args.model)
    paths = find_csv_files(args.data_dir, max_files=args.max_files)
    if not paths:
        raise SystemExit('No CSV files found for evaluation')
    # load and use first file for quick evaluation
    import pandas as pd
    df = pd.read_csv(paths[0])
    X, y = preprocess(df)
    preds = model.predict(X)
    print(classification_report(y, preds))


if __name__ == '__main__':
    main()
