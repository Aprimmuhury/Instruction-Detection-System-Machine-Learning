import os
from typing import List
import pandas as pd


def find_csv_files(data_dir: str, max_files: int = 200) -> List[str]:
    files = []
    for root, _, filenames in os.walk(data_dir):
        for f in filenames:
            if f.lower().endswith('.csv'):
                files.append(os.path.join(root, f))
                if len(files) >= max_files:
                    return files
    return files


def load_and_concat(csv_paths: List[str]) -> pd.DataFrame:
    if not csv_paths:
        raise ValueError('No CSV files found')
    dfs = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            df['_source_file'] = os.path.basename(p)
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        raise ValueError('No readable CSV files')
    return pd.concat(dfs, ignore_index=True)


def load_from_dir(data_dir: str, max_files: int = 200) -> pd.DataFrame:
    paths = find_csv_files(data_dir, max_files=max_files)
    return load_and_concat(paths)
