from typing import Tuple
import pandas as pd


def infer_label_column(df: pd.DataFrame) -> str:
    candidates = ['label', 'Label', 'target', 'class']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: last column
    return df.columns[-1]


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    label_col = infer_label_column(df)
    y = df[label_col].copy()
    
    # Drop label and source file columns
    cols_to_drop = [label_col]
    if '_source_file' in df.columns:
        cols_to_drop.append('_source_file')
    X = df.drop(columns=cols_to_drop)

    # Remove rows with NaN labels
    valid_idx = y.notna()
    X = X[valid_idx].reset_index(drop=True)
    y = y[valid_idx].reset_index(drop=True)
    
    if len(X) == 0:
        raise ValueError('No valid data after removing NaN labels')

    # Basic cleaning: drop constant columns
    nunique = X.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    X = X.drop(columns=const_cols)

    # Fill numeric NaNs with median
    num_cols = X.select_dtypes(include='number').columns
    for c in num_cols:
        median_val = X[c].median()
        if pd.notna(median_val):
            X[c] = X[c].fillna(median_val)
        else:
            X[c] = X[c].fillna(0)

    # One-hot encode object columns
    obj_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(obj_cols):
        X = pd.get_dummies(X, columns=obj_cols, dummy_na=True)

    return X, y
