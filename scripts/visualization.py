# scripts/visualization.py

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
    sns.histplot(data[score_column], bins=50, kde=True)
    plt.title(f'Distribution of {score_column.capitalize()}')
    plt.xlabel(score_column.capitalize())
    plt.ylabel('Frequency')
    plt.show()