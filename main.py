from scripts import (
    load_data, standardize_column_names, ensure_unique_column_names,
    preprocess_data, scale_data, apply_all_scores, create_ultimate_score,
    plot_play_duration_vs_match_share, plot_distribution_of_score, get_best_players_by_position, 
    plot_score_distribution_by_position, plot_top_players_by_position, plot_ultimate_score_by_position  
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
    'ai_score': 0.25,
    'weighted_score': 0.2,
    'z_score_combined': 0.15,
    'pca_score': 0.1,
    'geometric_mean_score': 0.1,
    'harmonic_mean_score': 0.1,
    'simple_sum_score': 0.1
}

# Apply all scoring methods to the dataset
data = apply_all_scores(data, 'playduration', 'matchshare')

# Create the ultimate score
data = create_ultimate_score(data, weights)

# Get the best players by position category
best_players_by_position = get_best_players_by_position(data)

# Print the best players from each position category
print("Top 3 Players by Position:")
for position, players in best_players_by_position.items():
    print(f"\nPosition: {position}")
    print(players)

# Visualizations
plot_play_duration_vs_match_share(data)
plot_distribution_of_score(data, 'ultimate_score')
plot_score_distribution_by_position(data, 'ultimate_score')
plot_top_players_by_position(data, 'ultimate_score')
plot_ultimate_score_by_position(data)