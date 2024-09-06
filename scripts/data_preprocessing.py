# scripts/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath, low_memory=False)

def standardize_column_names(data):
    """Standardize column names by converting to lowercase and replacing spaces with underscores."""
    data.columns = [col.strip().lower().replace(' ', '_') for col in data.columns]
    return data

def ensure_unique_column_names(data):
    """Ensure column names are unique by appending suffixes to duplicates."""
    cols = pd.Series(data.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))]
    data.columns = cols
    return data

def preprocess_data(data, required_columns):
    """Convert columns to numeric and handle missing values."""
    data[required_columns] = data[required_columns].apply(pd.to_numeric, errors='coerce')
    data[required_columns] = data[required_columns].fillna(data[required_columns].mean())
    return data

def scale_data(data, columns):
    """Standardize columns using StandardScaler."""
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

def create_combined_score(data: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """Create a combined score from two columns."""
    data['combined_score'] = data[col1] + data[col2]
    return data


