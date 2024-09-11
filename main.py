from scripts import (
    load_data, standardize_column_names, ensure_unique_column_names,
    preprocess_data, scale_data, apply_all_scores, create_ultimate_score,
    plot_play_duration_vs_match_share, plot_distribution_of_score, get_best_players_by_position, 
    plot_score_distribution_by_position, plot_top_players_by_position, plot_ultimate_score_by_position  
)

# Function to flag the top 3 players in each position based on their score
def add_top_3_players_flag(data, score_column='ultimate_score'):
    """
    Marking the top 3 players for each position in the dataset.
    
    Args:
        data (pd.DataFrame): The dataset with player stats.
        score_column (str): The score column used to rank players (default is 'ultimate_score').
        
    Returns:
        pd.DataFrame: The updated dataset with a new column indicating the top 3 players in each position.
    """
    # Sorting the dataset by score and position, so the top players appear at the top
    data = data.sort_values(by=[score_column, 'position'], ascending=[False, True])

    # Ranking players within each position and flagging the top 3
    data['rank_in_position'] = data.groupby('position')[score_column].rank(method='first', ascending=False)
    data['is_top_3_in_position'] = data['rank_in_position'] <= 3

    # Dropping the 'rank_in_position' column since it’s no longer needed
    data.drop(columns=['rank_in_position'], inplace=True)
    
    return data

# Loading the dataset
file_path = 'data/DataScientistInternTask.csv'
data = load_data(file_path)

# Standardizing column names (e.g., making them lowercase, replacing spaces with underscores) and ensuring there are no duplicate names
data = ensure_unique_column_names(standardize_column_names(data))

# Preprocessing the data: filling missing values, converting columns to numeric, and scaling the selected columns for analysis
required_columns = ['playduration', 'matchshare']
data = scale_data(preprocess_data(data, required_columns), required_columns)

# Applying different scoring methods to the data and calculating the ultimate score based on a set of weights
weights = {
    'ai_score': 0.25,  # 25% weight to AI-generated score
    'weighted_score': 0.2,  # 20% weight to a weighted score
    'z_score_combined': 0.15,  # 15% weight to Z-score based combined score
    'pca_score': 0.1,  # 10% weight to PCA-based score
    'geometric_mean_score': 0.1,  # 10% weight to geometric mean score
    'harmonic_mean_score': 0.1,  # 10% weight to harmonic mean score
    'simple_sum_score': 0.1  # 10% weight to simple sum of scores
}

# Applying the scoring methods and calculating the ultimate score
data = create_ultimate_score(apply_all_scores(data, 'playduration', 'matchshare'), weights)

# Adding a flag to identify the top 3 players in each position based on the ultimate score
data = add_top_3_players_flag(data, 'ultimate_score')

# Saving the processed data into a CSV file for use in Tableau or further analysis
csv_file_path = 'processed_data_for_tableau.csv'
data.to_csv(csv_file_path, index=False)
print(f"Data saved to {csv_file_path}")

# Getting the top players in each position
best_players_by_position = get_best_players_by_position(data)

# Printing out the top 3 players for each position
print("Top 3 Players by Position:")
for position, players in best_players_by_position.items():
    print(f"\nPosition: {position}")
    print(players)

# Generating visualizations to help understand the data better
plot_play_duration_vs_match_share(data)  # Plotting play duration vs match share
plot_distribution_of_score(data, 'ultimate_score')  # Showing the distribution of the ultimate score
plot_score_distribution_by_position(data, 'ultimate_score')  # Comparing score distributions by player position
plot_top_players_by_position(data, 'ultimate_score')  # Highlighting top players in each position
plot_ultimate_score_by_position(data)  # Visualizing the ultimate scores for players in different positions
