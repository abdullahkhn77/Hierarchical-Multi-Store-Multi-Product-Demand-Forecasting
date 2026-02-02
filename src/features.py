"""
Feature engineering module for Favorita Hierarchical Forecast.

This module will be populated in Step 2: Feature Engineering.

Planned features based on EDA findings:
- Temporal features (day of week, month, Fourier terms)
- Lag features (1, 7, 14, 28 day lags)
- Rolling statistics (mean, std, min, max)
- Promotion features (current, lagged, frequency)
- Oil price features (current, lagged, changes)
- Holiday features (flags, distance to holiday)
- Store/family embeddings and interactions
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path


def add_lag_features(df: pd.DataFrame,
                    target_col: str = 'sales',
                    group_cols: List[str] = ['store_nbr', 'family'],
                    lags: List[int] = [1, 7, 14, 28]) -> pd.DataFrame:
    """
    Add lagged features for the target column.

    Parameters:
    -----------
    df : DataFrame with date column sorted
    target_col : Column to create lags for
    group_cols : Columns to group by before creating lags
    lags : List of lag periods

    Returns:
    --------
    DataFrame with new lag columns
    """
    # TODO: Implement in Step 2
    raise NotImplementedError("Feature engineering - Step 2")


def add_rolling_features(df: pd.DataFrame,
                        target_col: str = 'sales',
                        group_cols: List[str] = ['store_nbr', 'family'],
                        windows: List[int] = [7, 14, 28],
                        agg_funcs: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
    """
    Add rolling window statistics.

    Parameters:
    -----------
    df : DataFrame with date column sorted
    target_col : Column to compute rolling stats for
    group_cols : Columns to group by
    windows : Window sizes
    agg_funcs : Aggregation functions

    Returns:
    --------
    DataFrame with rolling feature columns
    """
    # TODO: Implement in Step 2
    raise NotImplementedError("Feature engineering - Step 2")


def add_fourier_features(df: pd.DataFrame,
                        periods: List[Tuple[int, int]] = [(7, 3), (365, 5)]) -> pd.DataFrame:
    """
    Add Fourier terms for seasonality.

    Parameters:
    -----------
    df : DataFrame with date column
    periods : List of (period, n_terms) tuples
             e.g., (7, 3) = weekly with 3 harmonics

    Returns:
    --------
    DataFrame with Fourier feature columns
    """
    # TODO: Implement in Step 2
    raise NotImplementedError("Feature engineering - Step 2")


def add_holiday_features(df: pd.DataFrame,
                        look_ahead: int = 7,
                        look_back: int = 7) -> pd.DataFrame:
    """
    Add holiday-related features.

    - Days until next holiday
    - Days since last holiday
    - Pre/post holiday flags
    - Holiday type encoding

    Parameters:
    -----------
    df : DataFrame with holiday columns
    look_ahead : Days to look ahead for next holiday
    look_back : Days to look back for last holiday

    Returns:
    --------
    DataFrame with holiday feature columns
    """
    # TODO: Implement in Step 2
    raise NotImplementedError("Feature engineering - Step 2")


def add_oil_features(df: pd.DataFrame,
                    lags: List[int] = [1, 7, 14, 28],
                    windows: List[int] = [7, 14, 28]) -> pd.DataFrame:
    """
    Add oil price features.

    - Lagged oil prices
    - Oil price changes (%)
    - Rolling oil statistics

    Parameters:
    -----------
    df : DataFrame with dcoilwtico column
    lags : Lag periods for oil price
    windows : Rolling windows

    Returns:
    --------
    DataFrame with oil feature columns
    """
    # TODO: Implement in Step 2
    raise NotImplementedError("Feature engineering - Step 2")


def add_promotion_features(df: pd.DataFrame,
                          group_cols: List[str] = ['store_nbr', 'family']) -> pd.DataFrame:
    """
    Add promotion-related features.

    - Days since last promotion
    - Days until next promotion
    - Promotion frequency (rolling)
    - Cumulative promotions

    Parameters:
    -----------
    df : DataFrame with onpromotion column
    group_cols : Columns to group by

    Returns:
    --------
    DataFrame with promotion feature columns
    """
    # TODO: Implement in Step 2
    raise NotImplementedError("Feature engineering - Step 2")


def add_cyclical_encoding(df: pd.DataFrame,
                         col: str,
                         max_val: int) -> pd.DataFrame:
    """
    Add cyclical (sin/cos) encoding for a periodic feature.

    Parameters:
    -----------
    df : DataFrame
    col : Column to encode (e.g., 'dayofweek', 'month')
    max_val : Maximum value of the cycle (7 for dow, 12 for month)

    Returns:
    --------
    DataFrame with sin/cos encoded columns
    """
    # TODO: Implement in Step 2
    raise NotImplementedError("Feature engineering - Step 2")


def create_training_features(df: pd.DataFrame,
                            target_col: str = 'sales') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create full feature set for training.

    Combines all feature engineering functions.

    Parameters:
    -----------
    df : Master DataFrame with all data
    target_col : Target column name

    Returns:
    --------
    X : Feature DataFrame
    y : Target Series
    """
    # TODO: Implement in Step 2
    raise NotImplementedError("Feature engineering - Step 2")


if __name__ == "__main__":
    print("Feature engineering module - placeholder for Step 2")
    print("\nPlanned features:")
    print("- Lag features (1, 7, 14, 28 days)")
    print("- Rolling statistics (mean, std, min, max)")
    print("- Fourier terms (weekly, yearly)")
    print("- Holiday distance features")
    print("- Oil price features")
    print("- Promotion features")
    print("- Cyclical encodings")
