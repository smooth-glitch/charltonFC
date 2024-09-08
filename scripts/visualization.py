import matplotlib.pyplot as plt
import seaborn as sns

def plot_play_duration_vs_match_share(data):
    """Plot Play Duration vs Match Share (standardized)."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='playduration', y='matchshare', data=data)
    plt.title('Play Duration vs Match Share (Standardized)')
    plt.xlabel('Play Duration (standardized)')
    plt.ylabel('Match Share (standardized)')
    plt.show()

def plot_distribution_of_score(data, score_column):
    """Plot the distribution of any score column with a KDE."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data[score_column], bins=50, kde=True)
    plt.title(f'Distribution of {score_column.capitalize()}')
    plt.xlabel(score_column.capitalize())
    plt.ylabel('Frequency')
    plt.show()

def plot_score_distribution_by_position(data, score_column):
    """Plot the distribution of scores by player position."""
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='position', y=score_column, data=data)
    plt.title(f'Distribution of {score_column.capitalize()} by Position')
    plt.xlabel('Position')
    plt.ylabel(score_column.capitalize())
    plt.xticks(rotation=45)
    plt.show()

def plot_top_players_by_position(data, score_column, top_n=3):
    """Plot the top N players in each position based on the given score column."""
    top_players = data.groupby('position').apply(lambda x: x.nlargest(top_n, score_column)).reset_index(drop=True)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='playername', y=score_column, hue='position', data=top_players)
    plt.title(f'Top {top_n} Players in Each Position Based on {score_column.capitalize()}')
    plt.xlabel('Player')
    plt.ylabel(score_column.capitalize())
    plt.xticks(rotation=90)
    plt.legend(title='Position')
    plt.show()

def plot_ultimate_score_by_position(data):
    """Plot the ultimate score by position."""
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='position', y='ultimate_score', data=data)
    plt.title('Ultimate Score by Position')
    plt.xlabel('Position')
    plt.ylabel('Ultimate Score')
    plt.xticks(rotation=45)
    plt.show()
