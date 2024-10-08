# scripts/__init__.py

"""
This is the package for the main function.

Modules:
- data_preprocessing: Functions for loading and preprocessing data.
- analysis: Functions for analyzing data.
- visualization: Functions for plotting and visualizing data.
"""

from .data_preprocessing import (
    load_data, standardize_column_names, ensure_unique_column_names,
    preprocess_data, scale_data, apply_all_scores, create_ultimate_score, create_ai_based_score,
)
from .analysis import *
from .visualization import *
