# scripts/analysis.py

def get_top_players(data, player_column, score_column, top_n=3):
    """Sort players by combined score and return the top N players."""
    return data[[player_column, score_column]].sort_values(by=score_column, ascending=False).head(top_n)
