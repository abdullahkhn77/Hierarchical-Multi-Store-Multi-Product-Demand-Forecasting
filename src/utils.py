"""
Utility functions for Favorita Hierarchical Forecast project.
"""

import os
import zipfile
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def get_data_dir(subdir: str = "raw") -> Path:
    """Get data directory path."""
    return get_project_root() / "data" / subdir


def get_plots_dir() -> Path:
    """Get plots directory path."""
    return get_project_root() / "plots"


def setup_kaggle_credentials(kaggle_json_path: Optional[str] = None):
    """
    Setup Kaggle API credentials.

    Instructions to get kaggle.json:
    1. Go to https://www.kaggle.com/account
    2. Click "Create New Token" under API section
    3. Download kaggle.json
    4. Move it to ~/.kaggle/kaggle.json
    5. chmod 600 ~/.kaggle/kaggle.json
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)

    kaggle_creds = kaggle_dir / "kaggle.json"

    if kaggle_creds.exists():
        print(f"Kaggle credentials found at {kaggle_creds}")
        return True

    if kaggle_json_path and Path(kaggle_json_path).exists():
        import shutil
        shutil.copy(kaggle_json_path, kaggle_creds)
        os.chmod(kaggle_creds, 0o600)
        print(f"Copied credentials from {kaggle_json_path} to {kaggle_creds}")
        return True

    print("""
    Kaggle credentials not found. To set up:

    1. Go to https://www.kaggle.com/account
    2. Scroll to "API" section and click "Create New Token"
    3. Download kaggle.json
    4. Run these commands:
       mkdir -p ~/.kaggle
       mv ~/Downloads/kaggle.json ~/.kaggle/
       chmod 600 ~/.kaggle/kaggle.json

    Then run this function again or use download_favorita_data()
    """)
    return False


def download_favorita_data(data_dir: Optional[Path] = None):
    """
    Download Favorita store sales dataset from Kaggle.

    Requires kaggle API credentials to be set up.
    """
    if data_dir is None:
        data_dir = get_data_dir("raw")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    required_files = ["train.csv", "test.csv", "stores.csv", "oil.csv", "holidays_events.csv"]
    existing = [f for f in required_files if (data_dir / f).exists()]

    if len(existing) == len(required_files):
        print("All required data files already exist!")
        return True

    print(f"Found {len(existing)}/{len(required_files)} files. Downloading...")

    try:
        import kaggle
        kaggle.api.authenticate()

        # Download competition data
        kaggle.api.competition_download_files(
            competition="store-sales-time-series-forecasting",
            path=str(data_dir),
            quiet=False
        )

        # Unzip if needed
        zip_file = data_dir / "store-sales-time-series-forecasting.zip"
        if zip_file.exists():
            print("Extracting zip file...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            zip_file.unlink()  # Remove zip after extraction

        print("Download complete!")
        return True

    except Exception as e:
        print(f"Error downloading data: {e}")
        print("\nManual download instructions:")
        print("1. Go to https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data")
        print("2. Click 'Download All' (you may need to accept competition rules)")
        print(f"3. Extract the zip file contents to: {data_dir}")
        return False


def check_data_files(data_dir: Optional[Path] = None) -> dict:
    """Check which data files are present."""
    if data_dir is None:
        data_dir = get_data_dir("raw")

    data_dir = Path(data_dir)

    expected_files = {
        "train.csv": "Main training data (~3M rows)",
        "test.csv": "Test data for submission",
        "stores.csv": "Store metadata (city, state, type, cluster)",
        "oil.csv": "Daily oil prices",
        "holidays_events.csv": "Holiday and event information",
        "transactions.csv": "Daily transaction counts per store (optional)",
        "sample_submission.csv": "Submission format example"
    }

    status = {}
    for filename, description in expected_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            status[filename] = f"EXISTS ({size_mb:.1f} MB) - {description}"
        else:
            status[filename] = f"MISSING - {description}"

    return status


def print_memory_usage(df, name: str = "DataFrame"):
    """Print memory usage of a DataFrame."""
    mem_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"{name}: {len(df):,} rows, {len(df.columns)} columns, {mem_usage:.1f} MB")


def optimize_dtypes(df):
    """Optimize DataFrame memory usage by downcasting dtypes."""
    import pandas as pd
    import numpy as np

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == 'object':
            # Check if it should be categorical
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')

        elif col_type in ['int64', 'int32']:
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= 0:
                if c_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif c_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif c_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
            else:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

        elif col_type == 'float64':
            df[col] = df[col].astype(np.float32)

    return df


if __name__ == "__main__":
    # Check data files status
    print("Checking data files...")
    status = check_data_files()
    for filename, info in status.items():
        print(f"  {filename}: {info}")

    print("\nTo download data, run:")
    print("  from src.utils import download_favorita_data")
    print("  download_favorita_data()")
