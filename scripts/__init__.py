# scripts/__init__.py

"""
This is the data science task package.

Modules:
- data_preprocessing: Functions for loading and preprocessing data.
- analysis: Functions for analyzing data.
- visualization: Functions for plotting and visualizing data.
"""

from .data_preprocessing import (
    load_data, standardize_column_names, ensure_unique_column_names,
    preprocess_data, scale_data, apply_all_scores, create_ultimate_score, create_ai_based_score
)
from .analysis import get_top_players
from .visualization import plot_play_duration_vs_match_share, plot_distribution_of_score
