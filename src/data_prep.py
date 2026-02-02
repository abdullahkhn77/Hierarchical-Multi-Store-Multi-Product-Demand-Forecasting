"""
Data preparation module for Favorita Hierarchical Forecast.
Handles loading, cleaning, merging, and preprocessing of raw data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm

from .utils import get_data_dir, print_memory_usage, optimize_dtypes


def load_train_data(data_dir: Optional[Path] = None,
                    optimize_memory: bool = True) -> pd.DataFrame:
    """
    Load train.csv with optimized dtypes.

    Returns DataFrame with columns:
    - id, date, store_nbr, family, sales, onpromotion
    """
    if data_dir is None:
        data_dir = get_data_dir("raw")

    filepath = Path(data_dir) / "train.csv"

    print(f"Loading {filepath}...")

    # Define dtypes for memory efficiency
    dtypes = {
        'id': 'int32',
        'store_nbr': 'int8',
        'family': 'category',
        'sales': 'float32',
        'onpromotion': 'int16'
    }

    df = pd.read_csv(
        filepath,
        dtype=dtypes,
        parse_dates=['date']
    )

    print_memory_usage(df, "train.csv")
    return df


def load_stores_data(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load stores.csv with store metadata."""
    if data_dir is None:
        data_dir = get_data_dir("raw")

    filepath = Path(data_dir) / "stores.csv"

    df = pd.read_csv(filepath)

    # Convert to categories for memory efficiency
    for col in ['city', 'state', 'type']:
        df[col] = df[col].astype('category')

    df['cluster'] = df['cluster'].astype('int8')
    df['store_nbr'] = df['store_nbr'].astype('int8')

    print_memory_usage(df, "stores.csv")
    return df


def load_oil_data(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load oil.csv with daily oil prices."""
    if data_dir is None:
        data_dir = get_data_dir("raw")

    filepath = Path(data_dir) / "oil.csv"

    df = pd.read_csv(filepath, parse_dates=['date'])

    # Forward fill missing oil prices (common for weekends)
    df = df.set_index('date').sort_index()
    df['dcoilwtico'] = df['dcoilwtico'].ffill().bfill()
    df = df.reset_index()

    df['dcoilwtico'] = df['dcoilwtico'].astype('float32')

    print_memory_usage(df, "oil.csv")
    return df


def load_holidays_data(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load holidays_events.csv."""
    if data_dir is None:
        data_dir = get_data_dir("raw")

    filepath = Path(data_dir) / "holidays_events.csv"

    df = pd.read_csv(filepath, parse_dates=['date'])

    # Convert string columns to categories
    for col in ['type', 'locale', 'locale_name', 'description']:
        if col in df.columns:
            df[col] = df[col].astype('category')

    print_memory_usage(df, "holidays_events.csv")
    return df


def load_transactions_data(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load transactions.csv (optional file)."""
    if data_dir is None:
        data_dir = get_data_dir("raw")

    filepath = Path(data_dir) / "transactions.csv"

    if not filepath.exists():
        print("transactions.csv not found (optional file)")
        return None

    df = pd.read_csv(filepath, parse_dates=['date'])
    df['store_nbr'] = df['store_nbr'].astype('int8')
    df['transactions'] = df['transactions'].astype('int32')

    print_memory_usage(df, "transactions.csv")
    return df


def process_holidays(holidays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process holidays into a clean format with flags.

    Creates columns for:
    - is_holiday (any holiday)
    - is_national_holiday
    - is_regional_holiday
    - is_local_holiday
    - holiday_type (Work Day, Holiday, Event, etc.)
    - is_transferred (holiday was moved to another date)
    """
    df = holidays_df.copy()

    # Handle transferred holidays
    df['is_transferred'] = df['transferred'].fillna(False).astype(bool)

    # Create locale-based flags
    df['is_national'] = df['locale'] == 'National'
    df['is_regional'] = df['locale'] == 'Regional'
    df['is_local'] = df['locale'] == 'Local'

    # Aggregate by date (a date can have multiple holidays)
    holiday_agg = df.groupby('date').agg({
        'type': lambda x: ','.join(x.astype(str).unique()),
        'is_national': 'any',
        'is_regional': 'any',
        'is_local': 'any',
        'is_transferred': 'any',
        'description': lambda x: ','.join(x.astype(str).unique())
    }).reset_index()

    holiday_agg.columns = ['date', 'holiday_type', 'is_national_holiday',
                          'is_regional_holiday', 'is_local_holiday',
                          'is_transferred', 'holiday_description']

    holiday_agg['is_holiday'] = True

    return holiday_agg


def create_master_dataframe(data_dir: Optional[Path] = None,
                           save_parquet: bool = True) -> pd.DataFrame:
    """
    Create merged master DataFrame with all features.

    Merges:
    - train.csv (main data)
    - stores.csv (store metadata)
    - oil.csv (daily oil prices, forward filled)
    - holidays_events.csv (holiday flags)

    Saves to data/processed/train_merged.parquet
    """
    if data_dir is None:
        data_dir = get_data_dir("raw")

    processed_dir = get_data_dir("processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Creating Master DataFrame")
    print("=" * 60)

    # Load all data
    train_df = load_train_data(data_dir)
    stores_df = load_stores_data(data_dir)
    oil_df = load_oil_data(data_dir)
    holidays_df = load_holidays_data(data_dir)

    # Process holidays
    print("\nProcessing holidays...")
    holidays_processed = process_holidays(holidays_df)

    # Merge train with stores
    print("\nMerging train with stores...")
    master_df = train_df.merge(stores_df, on='store_nbr', how='left')
    print_memory_usage(master_df, "After stores merge")

    # Merge with oil
    print("Merging with oil prices...")
    master_df = master_df.merge(oil_df, on='date', how='left')

    # Forward fill any remaining missing oil prices
    master_df['dcoilwtico'] = master_df['dcoilwtico'].ffill().bfill()
    print_memory_usage(master_df, "After oil merge")

    # Merge with holidays
    print("Merging with holidays...")
    master_df = master_df.merge(holidays_processed, on='date', how='left')

    # Fill missing holiday flags with False
    holiday_cols = ['is_holiday', 'is_national_holiday', 'is_regional_holiday',
                   'is_local_holiday', 'is_transferred']
    for col in holiday_cols:
        if col in master_df.columns:
            master_df[col] = master_df[col].fillna(False).astype(bool)

    # Fill missing holiday descriptions
    master_df['holiday_type'] = master_df['holiday_type'].fillna('None').astype('category')
    master_df['holiday_description'] = master_df['holiday_description'].fillna('None').astype('category')

    print_memory_usage(master_df, "Final master DataFrame")

    # Add basic time features
    print("\nAdding time features...")
    master_df['year'] = master_df['date'].dt.year.astype('int16')
    master_df['month'] = master_df['date'].dt.month.astype('int8')
    master_df['day'] = master_df['date'].dt.day.astype('int8')
    master_df['dayofweek'] = master_df['date'].dt.dayofweek.astype('int8')
    master_df['dayofyear'] = master_df['date'].dt.dayofyear.astype('int16')
    master_df['weekofyear'] = master_df['date'].dt.isocalendar().week.astype('int8')
    master_df['is_weekend'] = master_df['dayofweek'].isin([5, 6])
    master_df['is_month_start'] = master_df['date'].dt.is_month_start
    master_df['is_month_end'] = master_df['date'].dt.is_month_end

    # Sort by date and store
    master_df = master_df.sort_values(['date', 'store_nbr', 'family']).reset_index(drop=True)

    print_memory_usage(master_df, "With time features")

    if save_parquet:
        output_path = processed_dir / "train_merged.parquet"
        print(f"\nSaving to {output_path}...")
        master_df.to_parquet(output_path, index=False)
        print(f"Saved! File size: {output_path.stat().st_size / (1024*1024):.1f} MB")

    print("\n" + "=" * 60)
    print("Master DataFrame created successfully!")
    print("=" * 60)

    return master_df


def load_master_dataframe(processed_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load the pre-processed master DataFrame from parquet."""
    if processed_dir is None:
        processed_dir = get_data_dir("processed")

    filepath = Path(processed_dir) / "train_merged.parquet"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Master DataFrame not found at {filepath}. "
            "Run create_master_dataframe() first."
        )

    print(f"Loading {filepath}...")
    df = pd.read_parquet(filepath)
    print_memory_usage(df, "Master DataFrame")

    return df


def generate_synthetic_data(n_days: int = 365,
                           n_stores: int = 10,
                           n_families: int = 10,
                           save: bool = True) -> dict:
    """
    Generate synthetic data matching Favorita schema for testing.

    Useful when real data is not available.
    """
    np.random.seed(42)

    data_dir = get_data_dir("raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic Favorita-like data...")

    # Date range
    dates = pd.date_range(start='2013-01-01', periods=n_days, freq='D')

    # Store metadata
    cities = ['Quito', 'Guayaquil', 'Cuenca', 'Machala', 'Santo Domingo']
    states = ['Pichincha', 'Guayas', 'Azuay', 'El Oro', 'Santo Domingo']
    store_types = ['A', 'B', 'C', 'D', 'E']

    stores_df = pd.DataFrame({
        'store_nbr': range(1, n_stores + 1),
        'city': np.random.choice(cities, n_stores),
        'state': np.random.choice(states, n_stores),
        'type': np.random.choice(store_types, n_stores),
        'cluster': np.random.randint(1, 18, n_stores)
    })

    # Families (product categories)
    families = ['GROCERY I', 'BEVERAGES', 'PRODUCE', 'CLEANING', 'DAIRY',
                'BREAD/BAKERY', 'POULTRY', 'MEATS', 'PERSONAL CARE', 'AUTOMOTIVE'][:n_families]

    # Generate train data
    train_rows = []
    id_counter = 0

    for date in tqdm(dates, desc="Generating train data"):
        for store in range(1, n_stores + 1):
            for family in families:
                # Base sales with seasonality
                base_sales = np.random.exponential(50)

                # Day of week effect (higher on weekends)
                dow = date.dayofweek
                dow_mult = 1.3 if dow >= 5 else 1.0

                # Monthly seasonality
                month_mult = 1 + 0.2 * np.sin(2 * np.pi * date.month / 12)

                # Family-specific scaling
                family_scale = {'GROCERY I': 2.0, 'BEVERAGES': 1.5, 'PRODUCE': 1.3}.get(family, 1.0)

                # Random promotion
                onpromotion = np.random.binomial(1, 0.15)
                promo_lift = 1.4 if onpromotion else 1.0

                sales = max(0, base_sales * dow_mult * month_mult * family_scale * promo_lift)

                # Add some zero sales (intermittent demand)
                if np.random.random() < 0.1:
                    sales = 0

                train_rows.append({
                    'id': id_counter,
                    'date': date,
                    'store_nbr': store,
                    'family': family,
                    'sales': round(sales, 2),
                    'onpromotion': onpromotion
                })
                id_counter += 1

    train_df = pd.DataFrame(train_rows)

    # Oil prices (random walk)
    oil_prices = [80]
    for _ in range(len(dates) - 1):
        change = np.random.normal(0, 2)
        oil_prices.append(max(30, min(150, oil_prices[-1] + change)))

    oil_df = pd.DataFrame({
        'date': dates,
        'dcoilwtico': oil_prices
    })

    # Holidays
    holiday_dates = dates[::30][:12]  # ~12 holidays
    holidays_df = pd.DataFrame({
        'date': holiday_dates,
        'type': np.random.choice(['Holiday', 'Event', 'Transfer'], len(holiday_dates)),
        'locale': np.random.choice(['National', 'Regional', 'Local'], len(holiday_dates)),
        'locale_name': 'Ecuador',
        'description': [f'Holiday {i}' for i in range(len(holiday_dates))],
        'transferred': [False] * len(holiday_dates)
    })

    datasets = {
        'train': train_df,
        'stores': stores_df,
        'oil': oil_df,
        'holidays_events': holidays_df
    }

    if save:
        for name, df in datasets.items():
            filepath = data_dir / f"{name}.csv"
            df.to_csv(filepath, index=False)
            print(f"Saved {filepath} ({len(df):,} rows)")

    return datasets


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--synthetic':
        print("Generating synthetic data for testing...")
        generate_synthetic_data(n_days=365*2, n_stores=20, n_families=15)
    else:
        # Try to create master dataframe from real data
        try:
            master_df = create_master_dataframe()
            print("\nSample of master DataFrame:")
            print(master_df.head())
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("\nTo generate synthetic data for testing, run:")
            print("  python -m src.data_prep --synthetic")
