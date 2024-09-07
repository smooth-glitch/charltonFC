from scripts import (
    load_data, standardize_column_names, ensure_unique_column_names,
    preprocess_data, scale_data, apply_all_scores, create_ultimate_score, create_ai_based_score, get_top_players,
    plot_play_duration_vs_match_share, plot_distribution_of_score
)

# Load the dataset
file_path = 'data/DataScientistInternTask.csv'
data = load_data(file_path)

# Standardize column names and ensure uniqueness
data = standardize_column_names(data)
data = ensure_unique_column_names(data)

# List of required columns
required_columns = ['playduration', 'matchshare']

# Preprocess data
data = preprocess_data(data, required_columns)
data = scale_data(data, required_columns)

# Define weights for each score type
weights = {
    'simple_sum_score': 0.2,
    'weighted_score': 0.2,
    'geometric_mean_score': 0.15,
    'z_score_combined': 0.15,
    'pca_score': 0.1,
    'harmonic_mean_score': 0.1,
    'custom_score': 0.1,
    'ai_score': 0.2
}

# Apply all scoring methods to the dataset
data = apply_all_scores(data, 'playduration', 'matchshare')

# Create the ultimate score
data = create_ultimate_score(data, weights)

# Print the top 3 players based on each scoring method
def print_top_players(data, score_column):
    """Helper function to print the top 3 players based on the provided score column."""
    top_players = get_top_players(data, 'playername', score_column)
    print(f"Top 3 Players Based on {score_column.replace('_', ' ').capitalize()}:")
    print(top_players)
    print("\n")

print_top_players(data, 'simple_sum_score')
print_top_players(data, 'weighted_score')
print_top_players(data, 'geometric_mean_score')
print_top_players(data, 'z_score_combined')
print_top_players(data, 'pca_score')
print_top_players(data, 'harmonic_mean_score')
print_top_players(data, 'custom_score')
print_top_players(data, 'ultimate_score')
print_top_players(data, 'ai_score')

# Visualizations
plot_play_duration_vs_match_share(data)
plot_distribution_of_score(data, 'ai_score')
