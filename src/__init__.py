"""
Favorita Hierarchical Forecast - Source Package
"""

from .utils import (
    get_project_root,
    get_data_dir,
    get_plots_dir,
    download_favorita_data,
    check_data_files,
)

from .data_prep import (
    load_train_data,
    load_stores_data,
    load_oil_data,
    load_holidays_data,
    create_master_dataframe,
    load_master_dataframe,
    generate_synthetic_data,
)

from .eda import (
    run_full_eda,
    generate_eda_report,
)

from .features import (
    create_all_features,
    add_calendar_features,
    add_cyclical_encodings,
    add_fourier_features,
    add_lag_features,
    add_rolling_features,
    add_ewm_features,
    add_promotion_features,
    add_holiday_features,
    add_oil_features,
    add_store_family_features,
    add_target_encoding,
    prepare_train_val_split,
    get_feature_columns,
)

__version__ = "0.2.0"
__all__ = [
    # Utils
    "get_project_root",
    "get_data_dir",
    "get_plots_dir",
    "download_favorita_data",
    "check_data_files",
    # Data prep
    "load_train_data",
    "load_stores_data",
    "load_oil_data",
    "load_holidays_data",
    "create_master_dataframe",
    "load_master_dataframe",
    "generate_synthetic_data",
    # EDA
    "run_full_eda",
    "generate_eda_report",
    # Features
    "create_all_features",
    "add_calendar_features",
    "add_cyclical_encodings",
    "add_fourier_features",
    "add_lag_features",
    "add_rolling_features",
    "add_ewm_features",
    "add_promotion_features",
    "add_holiday_features",
    "add_oil_features",
    "add_store_family_features",
    "add_target_encoding",
    "prepare_train_val_split",
    "get_feature_columns",
]
