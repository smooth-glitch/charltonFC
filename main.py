# main.py

from scripts import (
    load_data, standardize_column_names, ensure_unique_column_names,
    preprocess_data, scale_data, create_combined_score, get_top_players,
    plot_play_duration_vs_match_share, plot_distribution_of_combined_score
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

# Create combined score using the specified columns
data = create_combined_score(data, 'playduration', 'matchshare')

# Get top players
top_players = get_top_players(data, 'playername', 'combined_score')
print("Top 3 Players:")
print(top_players)

# Visualizations
plot_play_duration_vs_match_share(data)
plot_distribution_of_combined_score(data)
