# scripts/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

# Define combined score functions
def create_simple_sum_score(data: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """Create a simple sum combined score."""
    data['simple_sum_score'] = data[col1] + data[col2]
    return data

def create_weighted_combined_score(data: pd.DataFrame, col1: str, col2: str, weight1: float, weight2: float) -> pd.DataFrame:
    """Create a weighted combined score."""
    data['weighted_score'] = (data[col1] * weight1) + (data[col2] * weight2)
    return data

def create_geometric_mean_score(data: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """Create a geometric mean combined score."""
    data['geometric_mean_score'] = np.sqrt(data[col1] * data[col2])
    return data

def create_z_score_combined(data: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """Create a Z-score combined score."""
    data['z_score_combined'] = (data[col1] - data[col1].mean()) / data[col1].std() + (data[col2] - data[col2].mean()) / data[col2].std()
    return data

def create_pca_combined_score(data: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Create a PCA-based combined score."""
    pca = PCA(n_components=1)
    data['pca_score'] = pca.fit_transform(data[cols])
    return data

def create_harmonic_mean_score(data: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """Create a harmonic mean combined score."""
    data['harmonic_mean_score'] = 2 / ((1 / data[col1]) + (1 / data[col2]))
    return data

def create_custom_metric(data: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """Create a custom combined score."""
    data['custom_score'] = data[col1] + data[col2]
    # Add custom rules, e.g., bonus points
    data['custom_score'] += np.where(data[col1] > 1.5, 0.2, 0)  # Bonus for high playduration
    data['custom_score'] += np.where(data[col2] > 1.5, 0.2, 0)  # Bonus for high matchshare
    return data

# Apply all combined score functions
def apply_all_scores(data: pd.DataFrame, col1: str, col2: str, weight1: float = 0.6, weight2: float = 0.4):
    """Apply all scoring methods and add them as new columns."""
    data = create_simple_sum_score(data, col1, col2)
    data = create_weighted_combined_score(data, col1, col2, weight1, weight2)
    data = create_geometric_mean_score(data, col1, col2)
    data = create_z_score_combined(data, col1, col2)
    data = create_pca_combined_score(data, [col1, col2])
    data = create_harmonic_mean_score(data, col1, col2)
    data = create_custom_metric(data, col1, col2)
    return data

def create_ultimate_score(data: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Create an ultimate score based on weighted combination of all scores."""
    # Ensure all score columns are present
    required_columns = [
        'simple_sum_score', 'weighted_score', 'geometric_mean_score',
        'z_score_combined', 'pca_score', 'harmonic_mean_score', 'custom_score'
    ]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' is missing from data")

    # Apply weights to each score column
    data['ultimate_score'] = (
        data['simple_sum_score'] * weights.get('simple_sum_score', 0) +
        data['weighted_score'] * weights.get('weighted_score', 0) +
        data['geometric_mean_score'] * weights.get('geometric_mean_score', 0) +
        data['z_score_combined'] * weights.get('z_score_combined', 0) +
        data['pca_score'] * weights.get('pca_score', 0) +
        data['harmonic_mean_score'] * weights.get('harmonic_mean_score', 0) +
        data['custom_score'] * weights.get('custom_score', 0)
    )
    return data
