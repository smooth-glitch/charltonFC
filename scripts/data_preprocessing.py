import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

# Load and Preprocess Data
def load_data(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath, low_memory=False)

def standardize_column_names(data):
    """Convert column names to lowercase and replace spaces with underscores."""
    data.columns = data.columns.str.lower().str.replace(' ', '_').str.strip()
    return data

def ensure_unique_column_names(data):
    """Ensure column names are unique by appending a suffix to duplicates."""
    cols = pd.Series(data.columns)
    for dup in cols[cols.duplicated()].unique():  # Find duplicates
        dup_indices = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dup_indices):
            if i == 0:
                continue
            cols[idx] = f"{dup}_{i}"
    data.columns = cols
    return data

def preprocess_data(data, required_columns):
    """Convert to numeric, fill missing values."""
    data[required_columns] = data[required_columns].apply(pd.to_numeric, errors='coerce').fillna(data[required_columns].mean())
    return data

def scale_data(data, columns):
    """Scale selected columns using StandardScaler."""
    data[columns] = StandardScaler().fit_transform(data[columns])
    return data

# Scoring Methods
def create_combined_scores(data, col1, col2, weight1=0.4, weight2=0.6):
    """Apply all scoring methods."""
    data[col1] = data[col1].apply(lambda x: max(x, 1e-5))
    data[col2] = data[col2].apply(lambda x: max(x, 1e-5))
    data['simple_sum_score'] = data[col1] + data[col2]
    data['weighted_score'] = (data[col1] * weight1) + (data[col2] * weight2)
    data['geometric_mean_score'] = np.sqrt(data[col1] * data[col2])
    data['z_score_combined'] = (
        (data[col1] - data[col1].mean()) / data[col1].std() + 
        (data[col2] - data[col2].mean()) / data[col2].std()
    )
    data = create_pca_combined_score(data, [col1, col2])
    data['harmonic_mean_score'] = 2 / ((1 / data[col1]) + (1 / data[col2]))
    return data

def create_pca_combined_score(data, cols):
    """PCA-based score calculation."""
    pca = PCA(n_components=1)
    data['pca_score'] = pca.fit_transform(data[cols])
    return data

def apply_all_scores(data: pd.DataFrame, col1: str, col2: str, weight1: float = 0.6, weight2: float = 0.4):
    """Apply all scoring methods and add them as new columns."""
    data = create_combined_scores(data, col1, col2, weight1, weight2)
    data = create_ai_based_score(data, col1, col2)
    return data

def create_ai_based_score(data, col1, col2, model_type='random_forest'):
    """Create AI-based score using RandomForest or KMeans."""
    if model_type == 'kmeans':
        kmeans = KMeans(n_clusters=3, random_state=42)
        data['ai_score'] = kmeans.fit_predict(data[[col1, col2]])
    
    elif model_type == 'random_forest':
        # If 'performance_metric' is not available, create a proxy metric
        # Here we use a simple average of the two features as a target
        data['performance_metric'] = data[[col1, col2]].mean(axis=1)
        X = data[[col1, col2]]
        y = data['performance_metric']
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        data['ai_score'] = model.predict(X)
        # Optionally remove the proxy metric after prediction
        data.drop(columns=['performance_metric'], inplace=True)
    
    return data

def prepare_data_for_model(data):
    """Prepare features and target variable for model training."""
    X = data[['simple_sum_score', 'weighted_score', 'geometric_mean_score',
              'z_score_combined', 'pca_score', 'harmonic_mean_score', 'custom_score', 'ai_score']]
    y = data['performance_metric']  # Target column must exist
    return X, y

def train_ai_model(data):
    """Train a RandomForest to optimize weights for the ultimate score."""
    X, y = prepare_data_for_model(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

    importance = model.feature_importances_
    optimal_weights = {col: importance[i] for i, col in enumerate(X.columns)}
    
    total_importance = sum(optimal_weights.values())
    return {key: val / total_importance for key, val in optimal_weights.items()}

def create_ultimate_score(data, weights):
    """Combine all scores into one ultimate score."""
    required_columns = ['simple_sum_score', 'weighted_score', 'geometric_mean_score',
                        'z_score_combined', 'pca_score', 'harmonic_mean_score', 'ai_score']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' is missing from data")
    
    data['ultimate_score'] = sum(data[col] * weights.get(col, 0) for col in required_columns)
    return data

# Usage Example
if __name__ == "__main__":
    # Load and preprocess data
    file_path = 'data/DataScientistInternTask.csv'
    data = load_data(file_path)
    data = standardize_column_names(data)
    data = ensure_unique_column_names(data)
    required_columns = ['playduration', 'matchshare']
    data = preprocess_data(data, required_columns)
    data = scale_data(data, required_columns)

    # Apply all scores
    data = apply_all_scores(data, 'playduration', 'matchshare')

    # Train AI model and get optimal weights
    optimal_weights = train_ai_model(data)

    # Add AI-based score (choose between 'kmeans' or 'random_forest')
    data = create_ai_based_score(data, 'playduration', 'matchshare', model_type='random_forest')

    # Create ultimate score using the optimized weights
    data = create_ultimate_score(data, optimal_weights)

    # Save results or continue further analysis
    print("Data processed and ultimate scores created.")
