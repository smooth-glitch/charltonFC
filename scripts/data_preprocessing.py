import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

# Loading and Preprocessing Data
def load_data(filepath):
    """Loading the dataset from a CSV file."""
    return pd.read_csv(filepath, low_memory=False)

def standardize_column_names(data):
    """Cleaning up column names by making them lowercase and replacing spaces with underscores."""
    data.columns = data.columns.str.lower().str.replace(' ', '_').str.strip()
    return data

def ensure_unique_column_names(data):
    """Ensuring each column name is unique by adding suffixes if there are duplicates."""
    cols = pd.Series(data.columns)
    for dup in cols[cols.duplicated()].unique():  # Finding duplicate column names
        dup_indices = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dup_indices):
            if i == 0:
                continue
            cols[idx] = f"{dup}_{i}"
    data.columns = cols
    return data

def preprocess_data(data, required_columns):
    """Converting specific columns to numeric and filling any missing values with the column's mean."""
    data[required_columns] = data[required_columns].apply(pd.to_numeric, errors='coerce').fillna(data[required_columns].mean())
    return data

def scale_data(data, columns):
    """Standardizing selected columns to have a mean of 0 and a standard deviation of 1 (using StandardScaler)."""
    data[columns] = StandardScaler().fit_transform(data[columns])
    return data

# Scoring Methods
def create_combined_scores(data, col1, col2, weight1=0.4, weight2=0.6):
    """Generating different scores by combining two columns (like 'playduration' and 'matchshare') in multiple ways."""
    # Applying safety checks to avoid dividing by zero
    data[col1] = data[col1].apply(lambda x: max(x, 1e-5))
    data[col2] = data[col2].apply(lambda x: max(x, 1e-5))
    
    # Creating simple sum of both columns
    data['simple_sum_score'] = data[col1] + data[col2]
    
    # Creating a weighted combination of the two columns
    data['weighted_score'] = (data[col1] * weight1) + (data[col2] * weight2)
    
    # Creating geometric mean (an alternative way to combine the two columns)
    data['geometric_mean_score'] = np.sqrt(data[col1] * data[col2])
    
    # Creating Z-score-based combination to standardize and combine the two columns
    data['z_score_combined'] = (
        (data[col1] - data[col1].mean()) / data[col1].std() + 
        (data[col2] - data[col2].mean()) / data[col2].std()
    )
    
    # Creating PCA (Principal Component Analysis) score, which reduces both columns into a single score
    data = create_pca_combined_score(data, [col1, col2])
    
    # Creating harmonic mean score, which is more sensitive to low values
    data['harmonic_mean_score'] = 2 / ((1 / data[col1]) + (1 / data[col2]))
    
    return data

def create_pca_combined_score(data, cols):
    """Using PCA to combine two columns into a single score (capturing the main variance)."""
    pca = PCA(n_components=1)
    data['pca_score'] = pca.fit_transform(data[cols])
    return data

def apply_all_scores(data: pd.DataFrame, col1: str, col2: str, weight1: float = 0.6, weight2: float = 0.4):
    """Applying all the different scoring methods (sum, weighted, geometric, z-score, PCA, harmonic mean) to the data."""
    data = create_combined_scores(data, col1, col2, weight1, weight2)
    data = create_ai_based_score(data, col1, col2)
    return data

def create_ai_based_score(data, col1, col2, model_type='random_forest'):
    """Using an AI model (either RandomForest or KMeans) to generate a new score."""
    if model_type == 'kmeans':
        # Using KMeans clustering to assign players into 3 groups (clusters) based on the two columns
        kmeans = KMeans(n_clusters=3, random_state=42)
        data['ai_score'] = kmeans.fit_predict(data[[col1, col2]])
    elif model_type == 'random_forest':
        # Using RandomForest to predict a new AI-based score using the two columns
        X = data[[col1, col2]]
        model = RandomForestRegressor(random_state=42)
        model.fit(X, X.mean(axis=1))  # Training on the average of the two columns
        data['ai_score'] = model.predict(X)
    return data

def prepare_data_for_model(data):
    """Preparing the features (X) and target (y) for training the AI model."""
    X = data[['simple_sum_score', 'weighted_score', 'geometric_mean_score',
              'z_score_combined', 'pca_score', 'harmonic_mean_score', 'ai_score']]
    y = X.mean(axis=1)  # The target is the average of all the scores
    return X, y

def train_ai_model(data):
    """Training a RandomForest model to determine the optimal weights for the final score."""
    X, y = prepare_data_for_model(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training a RandomForest model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Predicting and evaluating the model using Mean Squared Error
    y_pred = model.predict(X_test)
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

    # Extracting feature importance from the trained model (this tells us how important each score is)
    importance = model.feature_importances_
    optimal_weights = {col: importance[i] for i, col in enumerate(X.columns)}
    
    # Normalizing the weights so they sum to 1
    total_importance = sum(optimal_weights.values())
    return {key: val / total_importance for key, val in optimal_weights.items()}

def create_ultimate_score(data, weights):
    """Combining all the scores into a final 'ultimate score' using the optimal weights."""
    required_columns = ['simple_sum_score', 'weighted_score', 'geometric_mean_score',
                        'z_score_combined', 'pca_score', 'harmonic_mean_score', 'ai_score']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' is missing from the data")

    # Calculating the ultimate score by summing the weighted scores
    data['ultimate_score'] = sum(data[col] * weights.get(col, 0) for col in required_columns)
    return data
