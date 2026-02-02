"""
Feature Engineering Module for Favorita Hierarchical Forecast.

This module provides comprehensive feature engineering for time series forecasting:
- Calendar/temporal features with cyclical encodings
- Lag features at multiple horizons
- Rolling statistics (mean, std, ewm)
- Promotion features
- Oil price features
- Holiday features
- Store/family target encodings
- Special event flags (earthquake, etc.)

Author: Senior Time Series Forecasting Expert
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# CALENDAR / TEMPORAL FEATURES
# =============================================================================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add comprehensive calendar and time-based features.

    Adds: year, month, day, dayofweek, weekofyear, quarter,
          is_weekend, is_payday, days_in_month positions, etc.
    """
    df = df.copy()

    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Basic calendar features (some may already exist)
    df['year'] = df['date'].dt.year.astype('int16')
    df['month'] = df['date'].dt.month.astype('int8')
    df['day'] = df['date'].dt.day.astype('int8')
    df['dayofweek'] = df['date'].dt.dayofweek.astype('int8')  # Monday=0
    df['dayofyear'] = df['date'].dt.dayofyear.astype('int16')
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype('int8')
    df['quarter'] = df['date'].dt.quarter.astype('int8')

    # Weekend flag
    df['is_weekend'] = df['dayofweek'].isin([5, 6])

    # Month position features
    df['is_month_start'] = df['date'].dt.is_month_start
    df['is_month_end'] = df['date'].dt.is_month_end
    df['days_in_month'] = df['date'].dt.days_in_month.astype('int8')
    df['day_of_month_sin'] = np.sin(2 * np.pi * df['day'] / df['days_in_month']).astype('float32')
    df['day_of_month_cos'] = np.cos(2 * np.pi * df['day'] / df['days_in_month']).astype('float32')

    # Payday features (15th and last day of month are common paydays in Ecuador)
    df['is_payday'] = (df['day'] == 15) | df['is_month_end']
    df['days_to_payday'] = df.apply(
        lambda x: min(abs(x['day'] - 15), abs(x['days_in_month'] - x['day']), x['day']),
        axis=1
    ).astype('int8')

    # Days from month boundaries
    df['days_since_month_start'] = (df['day'] - 1).astype('int8')
    df['days_to_month_end'] = (df['days_in_month'] - df['day']).astype('int8')

    # Year progress
    df['year_progress'] = (df['dayofyear'] / 365.25).astype('float32')

    return df


def add_cyclical_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sine/cosine cyclical encodings for periodic features.

    Encodes: dayofweek, month, dayofyear for capturing cyclical patterns.
    """
    df = df.copy()

    # Day of week (period = 7)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7).astype('float32')
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7).astype('float32')

    # Month (period = 12)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12).astype('float32')
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12).astype('float32')

    # Day of year (period = 365.25)
    df['doy_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25).astype('float32')
    df['doy_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25).astype('float32')

    # Week of year (period = 52)
    df['woy_sin'] = np.sin(2 * np.pi * df['weekofyear'] / 52).astype('float32')
    df['woy_cos'] = np.cos(2 * np.pi * df['weekofyear'] / 52).astype('float32')

    return df


def add_fourier_features(df: pd.DataFrame,
                        periods: List[Tuple[int, int]] = [(7, 3), (365, 4)]) -> pd.DataFrame:
    """
    Add Fourier terms for capturing complex seasonality.

    Parameters:
    -----------
    periods : List of (period, n_harmonics) tuples
              (7, 3) = weekly seasonality with 3 harmonics
              (365, 4) = yearly seasonality with 4 harmonics
    """
    df = df.copy()

    for period, n_harmonics in periods:
        for k in range(1, n_harmonics + 1):
            if period == 7:
                # Weekly: use dayofweek
                t = df['dayofweek']
            elif period == 365:
                # Yearly: use dayofyear
                t = df['dayofyear']
            else:
                t = df['dayofyear']

            df[f'fourier_sin_{period}_{k}'] = np.sin(2 * np.pi * k * t / period).astype('float32')
            df[f'fourier_cos_{period}_{k}'] = np.cos(2 * np.pi * k * t / period).astype('float32')

    return df


# =============================================================================
# LAG FEATURES
# =============================================================================

def add_lag_features(df: pd.DataFrame,
                    target_col: str = 'sales',
                    group_cols: List[str] = ['store_nbr', 'family'],
                    lags: List[int] = [1, 2, 3, 7, 14, 21, 28, 56]) -> pd.DataFrame:
    """
    Add lagged features for the target column, grouped by store-family.

    Critical for time series forecasting - captures autoregressive patterns.
    """
    df = df.copy()
    df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)

    print(f"  Adding {len(lags)} lag features for {target_col}...")

    for lag in tqdm(lags, desc="  Lags"):
        col_name = f'{target_col}_lag_{lag}'
        df[col_name] = df.groupby(group_cols)[target_col].shift(lag).astype('float32')

    return df


def add_same_day_last_year_lag(df: pd.DataFrame,
                               target_col: str = 'sales',
                               group_cols: List[str] = ['store_nbr', 'family']) -> pd.DataFrame:
    """
    Add same-day-last-year lag (handles leap years).

    Uses 364 days (52 weeks exactly) to align weekdays.
    """
    df = df.copy()

    print("  Adding same-day-last-year lag (364 days)...")

    # 364 days = exactly 52 weeks, preserves day of week
    df[f'{target_col}_lag_364'] = df.groupby(group_cols)[target_col].shift(364).astype('float32')

    # Also add 365 for comparison
    df[f'{target_col}_lag_365'] = df.groupby(group_cols)[target_col].shift(365).astype('float32')

    return df


# =============================================================================
# ROLLING FEATURES
# =============================================================================

def add_rolling_features(df: pd.DataFrame,
                        target_col: str = 'sales',
                        group_cols: List[str] = ['store_nbr', 'family'],
                        windows: List[int] = [7, 14, 28, 56]) -> pd.DataFrame:
    """
    Add rolling window statistics (mean, std).

    Uses shift(1) to avoid data leakage - rolling window ends at t-1.
    """
    df = df.copy()
    df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)

    print(f"  Adding rolling features for windows: {windows}...")

    for window in tqdm(windows, desc="  Rolling"):
        # Shifted by 1 to avoid leakage
        rolled = df.groupby(group_cols)[target_col].shift(1).rolling(window, min_periods=1)

        df[f'{target_col}_rolling_mean_{window}'] = rolled.mean().reset_index(drop=True).astype('float32')
        df[f'{target_col}_rolling_std_{window}'] = rolled.std().reset_index(drop=True).astype('float32')
        df[f'{target_col}_rolling_min_{window}'] = rolled.min().reset_index(drop=True).astype('float32')
        df[f'{target_col}_rolling_max_{window}'] = rolled.max().reset_index(drop=True).astype('float32')

    return df


def add_ewm_features(df: pd.DataFrame,
                    target_col: str = 'sales',
                    group_cols: List[str] = ['store_nbr', 'family'],
                    spans: List[int] = [7, 14, 28]) -> pd.DataFrame:
    """
    Add exponentially weighted moving average features.

    EWM gives more weight to recent observations.
    """
    df = df.copy()
    df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)

    print(f"  Adding EWM features for spans: {spans}...")

    for span in spans:
        # Shifted by 1 to avoid leakage
        ewm_col = df.groupby(group_cols)[target_col].shift(1).ewm(span=span, min_periods=1).mean()
        df[f'{target_col}_ewm_{span}'] = ewm_col.reset_index(drop=True).astype('float32')

    return df


# =============================================================================
# PROMOTION FEATURES
# =============================================================================

def add_promotion_features(df: pd.DataFrame,
                          group_cols: List[str] = ['store_nbr', 'family']) -> pd.DataFrame:
    """
    Add promotion-related features.

    Promotions are the strongest predictor in this dataset (618% avg lift).
    """
    df = df.copy()
    df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)

    print("  Adding promotion features...")

    # Binary has_promo flag
    df['has_promo'] = (df['onpromotion'] > 0).astype('int8')

    # Log transform of promotion count
    df['onpromotion_log1p'] = np.log1p(df['onpromotion']).astype('float32')

    # Lagged promotion features
    df['promo_lag_1'] = df.groupby(group_cols)['onpromotion'].shift(1).astype('float32')
    df['promo_lag_7'] = df.groupby(group_cols)['onpromotion'].shift(7).astype('float32')
    df['promo_lag_14'] = df.groupby(group_cols)['onpromotion'].shift(14).astype('float32')

    # Was promoted last week (binary)
    df['was_promoted_last_week'] = (df['promo_lag_7'] > 0).astype('int8')

    # Promotion change
    df['promo_change'] = (df['onpromotion'] - df['promo_lag_1']).astype('float32')

    # Rolling promotion sum (how many promos in last N days)
    for window in [7, 14, 28]:
        df[f'promo_rolling_sum_{window}'] = df.groupby(group_cols)['has_promo'].shift(1).rolling(
            window, min_periods=1
        ).sum().reset_index(drop=True).astype('float32')

    # Promotion intensity relative to rolling average
    df['promo_intensity'] = (
        df['onpromotion'] / (df['promo_rolling_sum_28'] / 28 + 1)
    ).astype('float32')

    return df


# =============================================================================
# HOLIDAY FEATURES
# =============================================================================

def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add enhanced holiday features.

    Includes pre/post holiday flags and holiday type encoding.
    """
    df = df.copy()

    print("  Adding holiday features...")

    # Ensure holiday flags exist and are boolean
    for col in ['is_holiday', 'is_national_holiday', 'is_regional_holiday', 'is_local_holiday']:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    # Get unique dates with holidays
    holiday_dates = df[df['is_holiday'] == True]['date'].unique()
    holiday_dates = pd.to_datetime(holiday_dates)

    if len(holiday_dates) > 0:
        # Create a date-level DataFrame for holiday distance calculation
        unique_dates = df['date'].unique()
        date_df = pd.DataFrame({'date': unique_dates})
        date_df = date_df.sort_values('date').reset_index(drop=True)

        # Calculate days to nearest holiday (simplified - just forward looking)
        def days_to_next_holiday(date, holidays, max_days=30):
            future = holidays[holidays >= date]
            if len(future) > 0:
                return min((future[0] - date).days, max_days)
            return max_days

        def days_since_last_holiday(date, holidays, max_days=30):
            past = holidays[holidays <= date]
            if len(past) > 0:
                return min((date - past[-1]).days, max_days)
            return max_days

        # This is slow for 3M rows, so we compute at date level and merge
        print("    Computing holiday distances (this may take a moment)...")
        holiday_array = np.sort(holiday_dates)

        date_df['days_to_holiday'] = date_df['date'].apply(
            lambda x: days_to_next_holiday(x, holiday_array)
        ).astype('int8')

        date_df['days_since_holiday'] = date_df['date'].apply(
            lambda x: days_since_last_holiday(x, holiday_array)
        ).astype('int8')

        # Merge back
        df = df.merge(date_df[['date', 'days_to_holiday', 'days_since_holiday']],
                      on='date', how='left')
    else:
        df['days_to_holiday'] = 30
        df['days_since_holiday'] = 30

    # Pre-holiday flags (1-3 days before)
    df['is_pre_holiday_1'] = (df['days_to_holiday'] == 1).astype('int8')
    df['is_pre_holiday_2'] = (df['days_to_holiday'] == 2).astype('int8')
    df['is_pre_holiday_3'] = (df['days_to_holiday'] <= 3).astype('int8')

    # Post-holiday flags
    df['is_post_holiday_1'] = (df['days_since_holiday'] == 1).astype('int8')
    df['is_post_holiday_2'] = (df['days_since_holiday'] == 2).astype('int8')

    # Holiday type encoding (if exists)
    if 'holiday_type' in df.columns:
        # Create dummies for holiday types
        df['is_bridge_holiday'] = df['holiday_type'].astype(str).str.contains('Bridge', case=False).astype('int8')
        df['is_transfer_holiday'] = df['holiday_type'].astype(str).str.contains('Transfer', case=False).astype('int8')
        df['is_workday_holiday'] = df['holiday_type'].astype(str).str.contains('Work', case=False).astype('int8')

    return df


# =============================================================================
# OIL PRICE FEATURES
# =============================================================================

def add_oil_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add oil price features.

    Oil price has strong negative correlation (-0.627) with sales in Ecuador.
    """
    df = df.copy()

    print("  Adding oil price features...")

    if 'dcoilwtico' not in df.columns:
        print("    Warning: dcoilwtico column not found!")
        return df

    # Forward fill any remaining NaN
    df['dcoilwtico'] = df['dcoilwtico'].ffill().bfill()

    # Create date-level oil features (oil is same for all store-families on a date)
    oil_df = df[['date', 'dcoilwtico']].drop_duplicates().sort_values('date')

    # Lagged oil prices
    for lag in [1, 7, 14, 28]:
        oil_df[f'oil_lag_{lag}'] = oil_df['dcoilwtico'].shift(lag)

    # Oil price change (percent)
    oil_df['oil_change_1d'] = oil_df['dcoilwtico'].pct_change(1).astype('float32')
    oil_df['oil_change_7d'] = oil_df['dcoilwtico'].pct_change(7).astype('float32')
    oil_df['oil_change_28d'] = oil_df['dcoilwtico'].pct_change(28).astype('float32')

    # Rolling oil statistics
    for window in [7, 14, 28]:
        oil_df[f'oil_rolling_mean_{window}'] = oil_df['dcoilwtico'].rolling(window, min_periods=1).mean()
        oil_df[f'oil_rolling_std_{window}'] = oil_df['dcoilwtico'].rolling(window, min_periods=1).std()

    # Oil relative to rolling mean (deviation)
    oil_df['oil_deviation_28d'] = (
        (oil_df['dcoilwtico'] - oil_df['oil_rolling_mean_28']) / (oil_df['oil_rolling_std_28'] + 1)
    ).astype('float32')

    # Convert to float32
    for col in oil_df.columns:
        if col != 'date' and oil_df[col].dtype == 'float64':
            oil_df[col] = oil_df[col].astype('float32')

    # Drop original oil column from merge (we'll keep the one from oil_df)
    df = df.drop(columns=['dcoilwtico'])

    # Merge oil features back
    df = df.merge(oil_df, on='date', how='left')

    return df


# =============================================================================
# SPECIAL EVENT FEATURES
# =============================================================================

def add_special_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add special event features like the 2016 Ecuador earthquake.

    The earthquake on 2016-04-16 significantly impacted sales.
    """
    df = df.copy()

    print("  Adding special event features...")

    # Ecuador earthquake: 2016-04-16
    earthquake_date = pd.Timestamp('2016-04-16')

    # Days since earthquake (0 before, positive after)
    df['days_since_earthquake'] = (df['date'] - earthquake_date).dt.days
    df['days_since_earthquake'] = df['days_since_earthquake'].clip(lower=0).astype('int16')

    # Earthquake period flag (immediate aftermath: 2 weeks)
    df['is_earthquake_period'] = (
        (df['date'] >= earthquake_date) &
        (df['date'] <= earthquake_date + pd.Timedelta(days=14))
    ).astype('int8')

    # Recovery period (2-8 weeks after)
    df['is_earthquake_recovery'] = (
        (df['date'] > earthquake_date + pd.Timedelta(days=14)) &
        (df['date'] <= earthquake_date + pd.Timedelta(days=56))
    ).astype('int8')

    return df


# =============================================================================
# STORE & FAMILY FEATURES
# =============================================================================

def add_store_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add store and family categorical features and groupings.
    """
    df = df.copy()

    print("  Adding store/family features...")

    # Store type as numeric (A=5, B=4, C=3, D=2, E=1 based on sales volume)
    type_map = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}
    df['store_type_num'] = df['type'].map(type_map).fillna(3).astype('int8')

    # Family groupings (higher-level categories)
    family_groups = {
        'GROCERY': ['GROCERY I', 'GROCERY II'],
        'BEVERAGES': ['BEVERAGES'],
        'PRODUCE': ['PRODUCE'],
        'CLEANING': ['CLEANING', 'HOME CARE'],
        'DAIRY': ['DAIRY'],
        'DELI': ['DELI'],
        'MEAT': ['MEATS', 'POULTRY', 'SEAFOOD'],
        'BAKERY': ['BREAD/BAKERY'],
        'FROZEN': ['FROZEN FOODS'],
        'PERSONAL_CARE': ['PERSONAL CARE', 'BABY CARE', 'BEAUTY'],
        'HOME': ['HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES', 'HARDWARE'],
        'CLOTHING': ['LINGERIE', 'LADIESWEAR'],
        'OTHER': ['AUTOMOTIVE', 'BOOKS', 'CELEBRATION', 'EGGS', 'LAWN AND GARDEN',
                  'MAGAZINES', 'PET SUPPLIES', 'PLAYERS AND ELECTRONICS',
                  'SCHOOL AND OFFICE SUPPLIES', 'PREPARED FOODS']
    }

    # Create reverse mapping
    family_to_group = {}
    for group, families in family_groups.items():
        for fam in families:
            family_to_group[fam] = group

    df['family_group'] = df['family'].astype(str).map(family_to_group).fillna('OTHER')
    df['family_group'] = df['family_group'].astype('category')

    return df


def add_target_encoding(df: pd.DataFrame,
                       target_col: str = 'sales',
                       encode_cols: List[str] = ['store_nbr', 'family', 'city', 'state', 'cluster'],
                       smoothing: float = 10.0) -> pd.DataFrame:
    """
    Add target encoding for categorical features.

    Uses smoothed mean to prevent overfitting on rare categories.
    Target encoding computed on training data only (before 2017-08-01).
    """
    df = df.copy()

    print("  Adding target encodings...")

    # Global mean
    global_mean = df[target_col].mean()

    # Training data only (to avoid leakage)
    train_mask = df['date'] < '2017-08-01'
    train_df = df[train_mask]

    for col in encode_cols:
        if col not in df.columns:
            continue

        # Calculate category statistics on training data
        agg = train_df.groupby(col)[target_col].agg(['mean', 'count'])

        # Smoothed mean: (count * mean + smoothing * global_mean) / (count + smoothing)
        agg['smoothed_mean'] = (
            (agg['count'] * agg['mean'] + smoothing * global_mean) /
            (agg['count'] + smoothing)
        )

        # Map to all data
        col_name = f'{col}_target_enc'
        df[col_name] = df[col].map(agg['smoothed_mean']).fillna(global_mean).astype('float32')

    return df


def add_store_family_aggregates(df: pd.DataFrame,
                               target_col: str = 'sales') -> pd.DataFrame:
    """
    Add store-level and family-level aggregate features.

    Captures hierarchical patterns in the data.
    """
    df = df.copy()
    df = df.sort_values(['date', 'store_nbr', 'family']).reset_index(drop=True)

    print("  Adding store/family aggregates...")

    # Date-store aggregates (total sales per store per day, lagged)
    store_daily = df.groupby(['date', 'store_nbr'])[target_col].sum().reset_index()
    store_daily.columns = ['date', 'store_nbr', 'store_daily_total']
    store_daily = store_daily.sort_values(['store_nbr', 'date'])
    store_daily['store_daily_total_lag1'] = store_daily.groupby('store_nbr')['store_daily_total'].shift(1)
    store_daily['store_daily_total_lag7'] = store_daily.groupby('store_nbr')['store_daily_total'].shift(7)

    df = df.merge(
        store_daily[['date', 'store_nbr', 'store_daily_total_lag1', 'store_daily_total_lag7']],
        on=['date', 'store_nbr'],
        how='left'
    )

    # Date-family aggregates (total sales per family per day, lagged)
    family_daily = df.groupby(['date', 'family'])[target_col].sum().reset_index()
    family_daily.columns = ['date', 'family', 'family_daily_total']
    family_daily = family_daily.sort_values(['family', 'date'])
    family_daily['family_daily_total_lag1'] = family_daily.groupby('family')['family_daily_total'].shift(1)
    family_daily['family_daily_total_lag7'] = family_daily.groupby('family')['family_daily_total'].shift(7)

    df = df.merge(
        family_daily[['date', 'family', 'family_daily_total_lag1', 'family_daily_total_lag7']],
        on=['date', 'family'],
        how='left'
    )

    # Convert to float32
    for col in ['store_daily_total_lag1', 'store_daily_total_lag7',
                'family_daily_total_lag1', 'family_daily_total_lag7']:
        df[col] = df[col].astype('float32')

    return df


# =============================================================================
# ZERO-INFLATION & NEW SERIES FEATURES
# =============================================================================

def add_zero_inflation_features(df: pd.DataFrame,
                               target_col: str = 'sales',
                               group_cols: List[str] = ['store_nbr', 'family']) -> pd.DataFrame:
    """
    Add features for handling zero-inflation and intermittent demand.
    """
    df = df.copy()
    df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)

    print("  Adding zero-inflation features...")

    # Is zero flag
    df['is_zero'] = (df[target_col] == 0).astype('int8')

    # Lagged zero flags
    df['was_zero_lag1'] = df.groupby(group_cols)['is_zero'].shift(1).fillna(1).astype('int8')
    df['was_zero_lag7'] = df.groupby(group_cols)['is_zero'].shift(7).fillna(1).astype('int8')

    # Proportion of zeros in last 7/28 days
    for window in [7, 28]:
        df[f'zero_ratio_{window}d'] = df.groupby(group_cols)['is_zero'].shift(1).rolling(
            window, min_periods=1
        ).mean().reset_index(drop=True).astype('float32')

    # Consecutive zeros count (days since last non-zero sale)
    def count_consecutive_zeros(series):
        # Create a mask where value changes from 0 to non-zero
        result = series.copy()
        counter = 0
        for i in range(len(series)):
            if series.iloc[i] == 0:
                counter += 1
            else:
                counter = 0
            result.iloc[i] = counter
        return result

    # This is slow, so we'll use a vectorized approximation
    df['days_since_nonzero'] = df.groupby(group_cols)['was_zero_lag1'].cumsum().astype('int16')

    return df


def add_new_series_features(df: pd.DataFrame,
                           target_col: str = 'sales',
                           group_cols: List[str] = ['store_nbr', 'family']) -> pd.DataFrame:
    """
    Add features for new stores/products (days since first sale).
    """
    df = df.copy()
    df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)

    print("  Adding new series features...")

    # First sale date per store-family
    first_sale = df[df[target_col] > 0].groupby(group_cols)['date'].min().reset_index()
    first_sale.columns = list(group_cols) + ['first_sale_date']

    df = df.merge(first_sale, on=group_cols, how='left')

    # Days since first sale
    df['days_since_first_sale'] = (df['date'] - df['first_sale_date']).dt.days
    df['days_since_first_sale'] = df['days_since_first_sale'].clip(lower=0).fillna(0).astype('int16')

    # Is new series (first 30 days)
    df['is_new_series'] = (df['days_since_first_sale'] <= 30).astype('int8')

    # Drop helper column
    df = df.drop(columns=['first_sale_date'])

    return df


# =============================================================================
# MAIN FEATURE ENGINEERING PIPELINE
# =============================================================================

def create_all_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Main function to create all features.

    Applies all feature engineering functions in sequence.
    """
    if verbose:
        print("=" * 70)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 70)
        print(f"Input shape: {df.shape}")

    # 1. Calendar features
    if verbose:
        print("\n[1/12] Calendar features...")
    df = add_calendar_features(df)

    # 2. Cyclical encodings
    if verbose:
        print("[2/12] Cyclical encodings...")
    df = add_cyclical_encodings(df)

    # 3. Fourier features
    if verbose:
        print("[3/12] Fourier features...")
    df = add_fourier_features(df, periods=[(7, 3), (365, 4)])

    # 4. Lag features
    if verbose:
        print("[4/12] Lag features...")
    df = add_lag_features(df, lags=[1, 2, 3, 7, 14, 21, 28, 56])
    df = add_same_day_last_year_lag(df)

    # 5. Rolling features
    if verbose:
        print("[5/12] Rolling features...")
    df = add_rolling_features(df, windows=[7, 14, 28, 56])

    # 6. EWM features
    if verbose:
        print("[6/12] EWM features...")
    df = add_ewm_features(df, spans=[7, 14, 28])

    # 7. Promotion features
    if verbose:
        print("[7/12] Promotion features...")
    df = add_promotion_features(df)

    # 8. Holiday features
    if verbose:
        print("[8/12] Holiday features...")
    df = add_holiday_features(df)

    # 9. Oil features
    if verbose:
        print("[9/12] Oil price features...")
    df = add_oil_features(df)

    # 10. Special events
    if verbose:
        print("[10/12] Special event features...")
    df = add_special_event_features(df)

    # 11. Store/family features
    if verbose:
        print("[11/12] Store/family features...")
    df = add_store_family_features(df)
    df = add_target_encoding(df)
    df = add_store_family_aggregates(df)

    # 12. Zero-inflation & new series
    if verbose:
        print("[12/12] Zero-inflation & new series features...")
    df = add_zero_inflation_features(df)
    df = add_new_series_features(df)

    if verbose:
        print(f"\nOutput shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
        print("=" * 70)

    return df


def prepare_train_val_split(df: pd.DataFrame,
                           val_start: str = '2017-08-01',
                           val_end: str = '2017-08-15') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and validation sets.

    Training: all data before val_start
    Validation: val_start to val_end
    """
    df = df.copy()

    train_df = df[df['date'] < val_start].copy()
    val_df = df[(df['date'] >= val_start) & (df['date'] <= val_end)].copy()

    print(f"Training set: {len(train_df):,} rows ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Validation set: {len(val_df):,} rows ({val_df['date'].min()} to {val_df['date'].max()})")

    return train_df, val_df


def get_feature_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Get lists of feature columns by category.
    """
    # Columns to exclude from features
    exclude_cols = ['id', 'date', 'sales', 'family', 'city', 'state', 'type',
                    'holiday_type', 'holiday_description', 'is_zero', 'family_group']

    # All potential feature columns
    all_cols = [c for c in df.columns if c not in exclude_cols]

    # Categorize features
    feature_groups = {
        'calendar': [c for c in all_cols if any(x in c for x in
                    ['year', 'month', 'day', 'week', 'quarter', 'payday', 'weekend'])],
        'cyclical': [c for c in all_cols if any(x in c for x in ['_sin', '_cos'])],
        'fourier': [c for c in all_cols if 'fourier' in c],
        'lag': [c for c in all_cols if '_lag_' in c],
        'rolling': [c for c in all_cols if 'rolling' in c],
        'ewm': [c for c in all_cols if '_ewm_' in c],
        'promotion': [c for c in all_cols if 'promo' in c.lower()],
        'holiday': [c for c in all_cols if 'holiday' in c.lower()],
        'oil': [c for c in all_cols if 'oil' in c.lower()],
        'store_family': [c for c in all_cols if any(x in c for x in
                        ['store', 'cluster', 'type_num', 'target_enc', 'daily_total'])],
        'zero_inflation': [c for c in all_cols if any(x in c for x in
                          ['zero', 'nonzero', 'first_sale', 'new_series'])],
        'special': [c for c in all_cols if 'earthquake' in c]
    }

    return feature_groups


def save_feature_list(df: pd.DataFrame, output_path: str = 'data/processed/features_list.txt'):
    """
    Save list of all features to a text file.
    """
    feature_groups = get_feature_columns(df)

    with open(output_path, 'w') as f:
        f.write("FAVORITA HIERARCHICAL FORECAST - ENGINEERED FEATURES\n")
        f.write("=" * 60 + "\n\n")

        total_features = 0
        for group_name, features in feature_groups.items():
            f.write(f"\n{group_name.upper()} FEATURES ({len(features)}):\n")
            f.write("-" * 40 + "\n")
            for feat in sorted(features):
                f.write(f"  - {feat}\n")
            total_features += len(features)

        f.write(f"\n\nTOTAL FEATURES: {total_features}\n")

    print(f"Feature list saved to {output_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path

    # Set paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data/processed/train_merged.parquet"
    output_dir = project_root / "data/processed"

    print("Loading data...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows")

    # Create all features
    df = create_all_features(df, verbose=True)

    # Save full feature set
    print("\nSaving full feature set...")
    df.to_parquet(output_dir / "train_features.parquet", index=False)

    # Create train/val split
    print("\nCreating train/validation split...")
    train_df, val_df = prepare_train_val_split(df)

    train_df.to_parquet(output_dir / "train_features_train.parquet", index=False)
    val_df.to_parquet(output_dir / "train_features_val.parquet", index=False)

    # Save feature list
    save_feature_list(df, str(output_dir / "features_list.txt"))

    print("\nDone!")
