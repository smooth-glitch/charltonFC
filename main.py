from scripts import (
    load_data, standardize_column_names, ensure_unique_column_names,
    preprocess_data, scale_data, apply_all_scores, create_ultimate_score,
    plot_play_duration_vs_match_share, plot_distribution_of_score, get_best_players_by_position, 
    plot_score_distribution_by_position, plot_top_players_by_position, plot_ultimate_score_by_position  
)

# Function to get the top 3 players from each position category and add a new column
def add_top_3_players_flag(data, score_column='ultimate_score'):
    """
    Mark the top 3 players from each position with a flag in a new column.
    
    Args:
        data (pd.DataFrame): The dataset containing player information.
        score_column (str): The column used to determine the top players (default is 'ultimate_score').
        
    Returns:
        pd.DataFrame: The dataset with an additional column indicating the top 3 players per position.
    """
    # Create a new column to indicate top players and initialize it with False
    data['is_top_3_in_position'] = False

    # Group by 'position'
    grouped = data.groupby('position')

    # Iterate over each group (each position)
    for position, group in grouped:
        # Sort the group by the score column and get the top 3 players
        top_players = group.sort_values(by=score_column, ascending=False).head(3)

        # Mark these top players in the original dataframe
        data.loc[top_players.index, 'is_top_3_in_position'] = True
    
    return data


# Load the dataset
file_path = 'data/DataScientistInternTask.csv'
data = load_data(file_path)

# Standardize column names and ensure uniqueness
data = standardize_column_names(data)
data = ensure_unique_column_names(data)

# Preprocess data
required_columns = ['playduration', 'matchshare']
data = preprocess_data(data, required_columns)
data = scale_data(data, required_columns)

# Apply all scoring methods and create the ultimate score
weights = {
    'ai_score': 0.25,
    'weighted_score': 0.2,
    'z_score_combined': 0.15,
    'pca_score': 0.1,
    'geometric_mean_score': 0.1,
    'harmonic_mean_score': 0.1,
    'simple_sum_score': 0.1
}
data = apply_all_scores(data, 'playduration', 'matchshare')
data = create_ultimate_score(data, weights)

# Add top 3 players flag
data = add_top_3_players_flag(data, 'ultimate_score')

# Verify DataFrame before saving
print(data.head())
print(data.columns)

# Save the updated DataFrame to a CSV file
csv_file_path = 'processed_data_for_tableau.csv'
data.to_csv(csv_file_path, index=False)
print(f"Data saved to {csv_file_path}")


best_players_by_position = get_best_players_by_position(data)

# Print the best players from each position category
print("Top 3 Players by Position:")
for position, players in best_players_by_position.items():
    print(f"\nPosition: {position}")
    print(players)

# Visualizations (optional)
plot_play_duration_vs_match_share(data)
plot_distribution_of_score(data, 'ultimate_score')
plot_score_distribution_by_position(data, 'ultimate_score')
plot_top_players_by_position(data, 'ultimate_score')
plot_ultimate_score_by_position(data)
