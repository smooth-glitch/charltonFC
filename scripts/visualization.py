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

def plot_distribution_of_combined_score(data):
    """Plot the distribution of Combined Score."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data['combined_score'], bins=50, kde=True)
    plt.title('Distribution of Combined Score')
    plt.xlabel('Combined Score')
    plt.ylabel('Frequency')
    plt.show()
