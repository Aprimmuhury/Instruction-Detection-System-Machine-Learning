import argparse
from src.data_loader import load_from_dir
from src.preprocess import preprocess
from src.model import train_and_save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='.', help='Directory containing CSV datasets')
    p.add_argument('--out', default='models/model.joblib', help='Output model path')
    p.add_argument('--max-files', type=int, default=200)
    args = p.parse_args()

    print(f'Loading up to {args.max_files} CSV files from {args.data_dir}...')
    df = load_from_dir(args.data_dir, max_files=args.max_files)
    print('Preprocessing...')
    X, y = preprocess(df)
    print('Training model...')
    report = train_and_save(X, y, args.out)
    print('Training complete. Evaluation report:')
    print(report)


if __name__ == '__main__':
    main()
