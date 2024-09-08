# scripts/analysis.py

def get_top_players(data, player_column, score_column, top_n=3):
    """Sort players by combined score and return the top N players."""
    return data[[player_column, score_column]].sort_values(by=score_column, ascending=False).head(top_n)

def get_best_players_by_position(data, score_column='ultimate_score'):
    """
    Get the top 3 players from each specific position category.
    
    Args:
        data (pd.DataFrame): The dataset containing player information.
        score_column (str): The column used to determine the best players (default is 'ultimate_score').
        
    Returns:
        dict: A dictionary where keys are positions and values are DataFrames of the top 3 players.
    """
    # Define the specific positions
    position_categories = [
        'CENTRAL_MIDFIELD', 'RIGHT_WINGBACK_DEFENDER', 'LEFT_WINGBACK_DEFENDER',
        'GOALKEEPER', 'DEFENSE_MIDFIELD', 'CENTER_FORWARD', 'ATTACKING_MIDFIELD',
        'CENTRAL_DEFENDER', 'LEFT_WINGER', 'RIGHT_WINGER'
    ]
    
    # Dictionary to store the top players for each position
    top_players_by_position = {}
    
    # Loop over each position and get the top 3 players based on the score column
    for position in position_categories:
        # Filter the players based on the position
        position_players = data[data['position'] == position]
        
        # Sort the players by the given score column and get the top 3
        top_players = position_players.sort_values(by=score_column, ascending=False).head(3)
        
        # Store the top 3 players in the dictionary
        top_players_by_position[position] = top_players[['playername', score_column]]
    
    return top_players_by_position
