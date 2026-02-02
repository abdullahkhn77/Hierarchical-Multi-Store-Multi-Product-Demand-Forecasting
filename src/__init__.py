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

__version__ = "0.1.0"
__all__ = [
    "get_project_root",
    "get_data_dir",
    "get_plots_dir",
    "download_favorita_data",
    "check_data_files",
    "load_train_data",
    "load_stores_data",
    "load_oil_data",
    "load_holidays_data",
    "create_master_dataframe",
    "load_master_dataframe",
    "generate_synthetic_data",
    "run_full_eda",
    "generate_eda_report",
]
