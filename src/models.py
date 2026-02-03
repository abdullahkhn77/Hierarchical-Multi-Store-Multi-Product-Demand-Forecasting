"""
Modeling Module for Favorita Hierarchical Forecast - Step 3.

This module provides:
- RMSLE and other metrics for evaluation
- Time-series cross-validation splitters
- Gradient boosting training and prediction (sklearn HistGradientBoosting)
- Error analysis utilities
- Feature importance extraction

Author: Senior Retail Forecasting Specialist
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple, Optional, Any, Generator
from pathlib import Path
import pickle
import warnings
from datetime import timedelta
import joblib

warnings.filterwarnings('ignore')


# =============================================================================
# METRICS
# =============================================================================

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Logarithmic Error.

    Handles zeros/negatives by clipping predictions to 0.
    RMSLE = sqrt(mean((log1p(pred) - log1p(actual))^2))
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Clip predictions to non-negative
    y_pred = np.maximum(y_pred, 0)

    # Compute RMSLE
    log_diff = np.log1p(y_pred) - np.log1p(y_true)
    return np.sqrt(np.mean(log_diff ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_pred - y_true))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """
    Mean Absolute Percentage Error.

    Adds epsilon to denominator to handle zeros.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean bias (mean(pred - actual)). Positive = over-predicting."""
    return np.mean(y_pred - y_true)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all standard metrics."""
    return {
        'rmsle': rmsle(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'mape': mape(y_true, y_pred),
        'bias': bias(y_true, y_pred),
        'rmse': np.sqrt(np.mean((y_pred - y_true) ** 2))
    }


# =============================================================================
# TIME-SERIES CROSS-VALIDATION
# =============================================================================

class TimeSeriesFoldGenerator:
    """
    Time-series cross-validation fold generator.

    Creates expanding or sliding window splits for time series data.
    Mimics the 16-day test period structure.
    """

    def __init__(
        self,
        n_folds: int = 3,
        val_days: int = 16,
        gap_days: int = 0,
        expanding: bool = True,
        min_train_days: int = 365
    ):
        """
        Parameters:
        -----------
        n_folds : Number of CV folds
        val_days : Days in each validation window (16 = competition test period)
        gap_days : Gap between train and val (for embargo if needed)
        expanding : If True, use expanding window; if False, sliding window
        min_train_days : Minimum training days required
        """
        self.n_folds = n_folds
        self.val_days = val_days
        self.gap_days = gap_days
        self.expanding = expanding
        self.min_train_days = min_train_days

    def split(self, df: pd.DataFrame, date_col: str = 'date') -> Generator:
        """
        Generate train/validation indices for each fold.

        Yields: (fold_num, train_idx, val_idx, train_dates, val_dates)
        """
        dates = df[date_col].sort_values().unique()
        n_dates = len(dates)

        # Calculate fold boundaries (work backwards from end)
        # Last fold should end near end of data
        last_val_end = dates[-1]

        for fold in range(self.n_folds - 1, -1, -1):
            # Validation period for this fold
            fold_offset = (self.n_folds - 1 - fold) * self.val_days
            val_end_idx = n_dates - 1 - fold_offset
            val_start_idx = val_end_idx - self.val_days + 1

            if val_start_idx < 0:
                continue

            val_end = dates[val_end_idx]
            val_start = dates[val_start_idx]

            # Training period
            train_end_idx = val_start_idx - 1 - self.gap_days
            if train_end_idx < self.min_train_days:
                continue

            train_end = dates[train_end_idx]

            if self.expanding:
                train_start = dates[0]
            else:
                # Sliding window: use same size as largest expanding window
                train_start = dates[max(0, train_end_idx - 365 * 3)]  # ~3 years

            # Get indices
            train_mask = (df[date_col] >= train_start) & (df[date_col] <= train_end)
            val_mask = (df[date_col] >= val_start) & (df[date_col] <= val_end)

            train_idx = df[train_mask].index.tolist()
            val_idx = df[val_mask].index.tolist()

            yield (
                self.n_folds - fold,  # fold number (1-indexed)
                train_idx,
                val_idx,
                (train_start, train_end),
                (val_start, val_end)
            )


def get_cv_folds(
    df: pd.DataFrame,
    date_col: str = 'date',
    folds: Optional[List[Dict]] = None
) -> List[Dict]:
    """
    Get predefined CV folds matching competition timeline.

    Default folds (adjusted for available data):
    - Fold 1: train until 2017-06-30 -> val 2017-07-01 to 2017-07-15
    - Fold 2: train until 2017-07-15 -> val 2017-07-16 to 2017-07-31
    - Fold 3: train until 2017-07-31 -> val 2017-08-01 to 2017-08-15
    """
    max_date = df[date_col].max()

    if folds is None:
        # Use folds that fit within available data
        folds = [
            {
                'name': 'Fold 1',
                'train_end': '2017-06-30',
                'val_start': '2017-07-01',
                'val_end': '2017-07-15'
            },
            {
                'name': 'Fold 2',
                'train_end': '2017-07-15',
                'val_start': '2017-07-16',
                'val_end': '2017-07-31'
            },
            {
                'name': 'Fold 3',
                'train_end': '2017-07-31',
                'val_start': '2017-08-01',
                'val_end': '2017-08-15'
            }
        ]

    result = []
    for fold in folds:
        train_mask = df[date_col] <= fold['train_end']
        val_mask = (df[date_col] >= fold['val_start']) & (df[date_col] <= fold['val_end'])

        # Skip folds with no validation data
        if val_mask.sum() == 0:
            print(f"  Warning: Skipping {fold['name']} - no validation data available")
            continue

        result.append({
            'name': fold['name'],
            'train_idx': df[train_mask].index.tolist(),
            'val_idx': df[val_mask].index.tolist(),
            'train_dates': (df[train_mask][date_col].min(), fold['train_end']),
            'val_dates': (fold['val_start'], fold['val_end']),
            'train_size': train_mask.sum(),
            'val_size': val_mask.sum()
        })

    return result


# =============================================================================
# FEATURE PREPARATION
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns for modeling.

    Excludes: target, date, IDs, raw categoricals (use encoded versions)
    """
    exclude_cols = {
        # Target and identifiers
        'sales', 'id', 'date',
        # Raw categoricals (will use encoded or as lgb categorical)
        'family', 'city', 'state', 'type', 'family_group',
        'holiday_type', 'holiday_description',
        # Leakage
        'is_zero',
        # Internal
        'log_sales'
    }

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Get categorical columns for encoding."""
    cat_cols = ['store_nbr', 'cluster']
    return [c for c in cat_cols if c in df.columns]


def prepare_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    label_encoders: Optional[Dict] = None
) -> Tuple[pd.DataFrame, List[str], List[str], Dict]:
    """
    Prepare features for modeling.

    Returns: (X, feature_cols, categorical_cols, label_encoders)
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    if cat_cols is None:
        cat_cols = get_categorical_columns(df)

    X = df[feature_cols].copy()

    # Encode categoricals for sklearn
    if label_encoders is None:
        label_encoders = {}

    for col in cat_cols:
        if col in X.columns:
            if col not in label_encoders:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
            else:
                # Handle unseen categories
                le = label_encoders[col]
                X[col] = X[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

    # Fill any remaining NaN with 0 (should be minimal after feature engineering)
    X = X.fillna(0)

    # Convert boolean columns to int
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)

    return X, feature_cols, cat_cols, label_encoders


# =============================================================================
# GRADIENT BOOSTING TRAINING (sklearn HistGradientBoosting)
# =============================================================================

def get_hgb_params() -> Dict[str, Any]:
    """
    Get sensible default HistGradientBoostingRegressor parameters.

    These are production-ready defaults, not fully tuned but robust.
    """
    params = {
        'loss': 'squared_error',
        'learning_rate': 0.05,
        'max_iter': 1500,
        'max_depth': 10,
        'max_leaf_nodes': 63,
        'min_samples_leaf': 50,
        'l2_regularization': 1.0,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 50,
        'random_state': 42,
        'verbose': 0
    }

    return params


def train_hgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
    verbose: bool = True
) -> Tuple[HistGradientBoostingRegressor, Dict]:
    """
    Train a HistGradientBoostingRegressor model.

    Returns: (model, training_info)
    """
    if params is None:
        params = get_hgb_params()

    # If validation data provided, use it for early stopping
    if X_val is not None and y_val is not None:
        params = params.copy()
        params['early_stopping'] = True
        params['validation_fraction'] = None  # Will use provided val set

    model = HistGradientBoostingRegressor(**params)

    if verbose:
        print("    Training HistGradientBoostingRegressor...")

    model.fit(X_train, y_train)

    training_info = {
        'n_iter': model.n_iter_,
        'feature_names': list(X_train.columns)
    }

    if hasattr(model, 'validation_score_'):
        training_info['best_score'] = model.validation_score_[-1] if len(model.validation_score_) > 0 else None

    return model, training_info


def predict_hgb(
    model: HistGradientBoostingRegressor,
    X: pd.DataFrame,
    clip_negative: bool = True
) -> np.ndarray:
    """
    Generate predictions from HistGradientBoosting model.

    Clips negative predictions to 0.
    """
    preds = model.predict(X)

    if clip_negative:
        preds = np.maximum(preds, 0)

    return preds


# =============================================================================
# CROSS-VALIDATION RUNNER
# =============================================================================

def run_cv(
    df: pd.DataFrame,
    target_col: str = 'sales',
    use_log_target: bool = False,
    params: Optional[Dict] = None,
    verbose: bool = True
) -> Tuple[Dict, List[HistGradientBoostingRegressor], pd.DataFrame]:
    """
    Run time-series cross-validation with HistGradientBoosting.

    Returns: (cv_results, models, oof_predictions)
    """
    # Get CV folds
    folds = get_cv_folds(df)

    # Prepare features once to get column lists
    feature_cols = get_feature_columns(df)
    cat_cols = get_categorical_columns(df)

    if verbose:
        print(f"Features: {len(feature_cols)} columns")
        print(f"Categorical: {cat_cols}")
        print(f"Target: {target_col}" + (" (log-transformed)" if use_log_target else ""))
        print("-" * 60)

    models = []
    fold_results = []
    oof_df = pd.DataFrame()
    label_encoders = None  # Will be fitted on first fold and reused

    for fold_info in folds:
        fold_name = fold_info['name']

        if verbose:
            print(f"\n{fold_name}:")
            print(f"  Train: {fold_info['train_dates'][0]} to {fold_info['train_dates'][1]} ({fold_info['train_size']:,} rows)")
            print(f"  Val:   {fold_info['val_dates'][0]} to {fold_info['val_dates'][1]} ({fold_info['val_size']:,} rows)")

        # Get train/val data
        train_df = df.iloc[fold_info['train_idx']].copy()
        val_df = df.iloc[fold_info['val_idx']].copy()

        # Prepare features (fit encoders on first fold only)
        X_train, _, _, label_encoders = prepare_features(
            train_df, feature_cols, cat_cols,
            label_encoders=None if label_encoders is None else label_encoders
        )
        X_val, _, _, _ = prepare_features(val_df, feature_cols, cat_cols, label_encoders)

        if use_log_target:
            y_train = np.log1p(train_df[target_col].values)
            y_val_raw = val_df[target_col].values
        else:
            y_train = train_df[target_col].values
            y_val_raw = val_df[target_col].values

        # Train model
        model, train_info = train_hgb(
            X_train, y_train,
            params=params,
            verbose=verbose
        )

        models.append(model)

        # Predict on validation
        preds = predict_hgb(model, X_val)

        if use_log_target:
            preds = np.expm1(preds)  # Transform back
            preds = np.maximum(preds, 0)

        # Compute metrics
        metrics = compute_all_metrics(y_val_raw, preds)
        metrics['fold'] = fold_name
        metrics['n_iter'] = train_info['n_iter']
        fold_results.append(metrics)

        if verbose:
            print(f"  -> RMSLE: {metrics['rmsle']:.4f} | MAE: {metrics['mae']:.2f} | Iterations: {train_info['n_iter']}")

        # Store OOF predictions
        oof_fold = val_df[['date', 'store_nbr', 'family', target_col]].copy()
        oof_fold['pred'] = preds
        oof_fold['fold'] = fold_name
        oof_df = pd.concat([oof_df, oof_fold], ignore_index=True)

    # Aggregate results
    results_df = pd.DataFrame(fold_results)
    cv_results = {
        'folds': fold_results,
        'mean_rmsle': results_df['rmsle'].mean(),
        'std_rmsle': results_df['rmsle'].std(),
        'mean_mae': results_df['mae'].mean(),
        'mean_mape': results_df['mape'].mean(),
        'mean_bias': results_df['bias'].mean(),
        'label_encoders': label_encoders
    }

    if verbose:
        print("\n" + "=" * 60)
        print(f"CV Results: RMSLE = {cv_results['mean_rmsle']:.4f} +/- {cv_results['std_rmsle']:.4f}")
        print(f"            MAE = {cv_results['mean_mae']:.2f}")
        print(f"            MAPE = {cv_results['mean_mape']:.2f}%")
        print(f"            Bias = {cv_results['mean_bias']:.2f}")
        print("=" * 60)

    return cv_results, models, oof_df


# =============================================================================
# ERROR ANALYSIS
# =============================================================================

def analyze_errors_by_group(
    oof_df: pd.DataFrame,
    group_col: str,
    actual_col: str = 'sales',
    pred_col: str = 'pred'
) -> pd.DataFrame:
    """
    Analyze prediction errors grouped by a column.

    Returns DataFrame with RMSLE, MAE, MAPE, count per group.
    """
    results = []

    for group_val, group_df in oof_df.groupby(group_col):
        y_true = group_df[actual_col].values
        y_pred = group_df[pred_col].values

        metrics = compute_all_metrics(y_true, y_pred)
        metrics[group_col] = group_val
        metrics['count'] = len(group_df)
        metrics['mean_actual'] = y_true.mean()
        metrics['zero_pct'] = (y_true == 0).mean() * 100
        results.append(metrics)

    return pd.DataFrame(results).sort_values('rmsle', ascending=True)


def analyze_promotion_effect(
    oof_df: pd.DataFrame,
    df_full: pd.DataFrame,
    actual_col: str = 'sales',
    pred_col: str = 'pred'
) -> Dict:
    """
    Analyze errors on promo vs non-promo days.
    """
    # Merge promo info
    promo_cols = ['date', 'store_nbr', 'family', 'onpromotion', 'has_promo']
    promo_cols = [c for c in promo_cols if c in df_full.columns]

    merged = oof_df.merge(
        df_full[promo_cols].drop_duplicates(),
        on=['date', 'store_nbr', 'family'],
        how='left'
    )

    if 'has_promo' not in merged.columns and 'onpromotion' in merged.columns:
        merged['has_promo'] = (merged['onpromotion'] > 0).astype(int)

    results = {}
    for promo_val in [0, 1]:
        mask = merged['has_promo'] == promo_val
        subset = merged[mask]
        if len(subset) > 0:
            metrics = compute_all_metrics(subset[actual_col].values, subset[pred_col].values)
            metrics['count'] = len(subset)
            results[f"promo={promo_val}"] = metrics

    return results


def analyze_zero_vs_nonzero(
    oof_df: pd.DataFrame,
    actual_col: str = 'sales',
    pred_col: str = 'pred'
) -> Dict:
    """
    Analyze errors on zero vs non-zero actual sales.
    """
    results = {}

    # Zero actual
    zero_mask = oof_df[actual_col] == 0
    zero_df = oof_df[zero_mask]
    if len(zero_df) > 0:
        results['zero_actual'] = {
            'count': len(zero_df),
            'mean_pred': zero_df[pred_col].mean(),
            'mae': mae(zero_df[actual_col].values, zero_df[pred_col].values),
            'over_predict_pct': (zero_df[pred_col] > 0).mean() * 100
        }

    # Non-zero actual
    nonzero_df = oof_df[~zero_mask]
    if len(nonzero_df) > 0:
        metrics = compute_all_metrics(nonzero_df[actual_col].values, nonzero_df[pred_col].values)
        metrics['count'] = len(nonzero_df)
        results['nonzero_actual'] = metrics

    return results


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def get_feature_importance(
    model: HistGradientBoostingRegressor,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract permutation-based feature importance from sklearn model.

    Uses the built-in feature_importances_ if available.
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        # Fallback to zeros if not available
        importance = np.zeros(len(feature_names))

    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    imp_df['importance_pct'] = imp_df['importance'] / (imp_df['importance'].sum() + 1e-10) * 100

    return imp_df


def aggregate_feature_importance(
    models: List[HistGradientBoostingRegressor],
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Aggregate feature importance across multiple models (CV folds).
    """
    all_imp = []

    for i, model in enumerate(models):
        imp = get_feature_importance(model, feature_names)
        imp['fold'] = i + 1
        all_imp.append(imp)

    all_imp_df = pd.concat(all_imp, ignore_index=True)

    # Aggregate
    agg_imp = all_imp_df.groupby('feature').agg({
        'importance': ['mean', 'std']
    }).reset_index()
    agg_imp.columns = ['feature', 'importance_mean', 'importance_std']
    agg_imp = agg_imp.sort_values('importance_mean', ascending=False)
    agg_imp['importance_pct'] = agg_imp['importance_mean'] / (agg_imp['importance_mean'].sum() + 1e-10) * 100

    return agg_imp


# =============================================================================
# NAIVE BASELINES
# =============================================================================

def naive_baseline_last_week(df: pd.DataFrame, target_col: str = 'sales') -> pd.Series:
    """
    Naive baseline: same value as 7 days ago.
    """
    return df.groupby(['store_nbr', 'family'])[target_col].shift(7)


def naive_baseline_rolling_mean(
    df: pd.DataFrame,
    target_col: str = 'sales',
    window: int = 7
) -> pd.Series:
    """
    Naive baseline: rolling mean of last N days.
    """
    return df.groupby(['store_nbr', 'family'])[target_col].shift(1).rolling(
        window, min_periods=1
    ).mean().reset_index(drop=True)


def compute_naive_baselines(
    df: pd.DataFrame,
    val_mask: pd.Series,
    target_col: str = 'sales'
) -> Dict:
    """
    Compute naive baseline metrics on validation set.
    """
    results = {}
    y_true = df.loc[val_mask, target_col].values

    # Last week
    pred_last_week = naive_baseline_last_week(df, target_col).loc[val_mask].fillna(0).values
    results['naive_last_week'] = compute_all_metrics(y_true, pred_last_week)

    # Rolling mean 7d
    pred_rolling_7 = naive_baseline_rolling_mean(df, target_col, 7).loc[val_mask].fillna(0).values
    results['naive_rolling_7d'] = compute_all_metrics(y_true, pred_rolling_7)

    # Rolling mean 28d
    pred_rolling_28 = naive_baseline_rolling_mean(df, target_col, 28).loc[val_mask].fillna(0).values
    results['naive_rolling_28d'] = compute_all_metrics(y_true, pred_rolling_28)

    return results


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(model: HistGradientBoostingRegressor, path: str):
    """Save sklearn model to file."""
    joblib.dump(model, path)


def load_model(path: str) -> HistGradientBoostingRegressor:
    """Load sklearn model from file."""
    return joblib.load(path)


def save_models(models: List[HistGradientBoostingRegressor], output_dir: str, prefix: str = 'hgb'):
    """Save multiple models (e.g., from CV)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, model in enumerate(models):
        path = output_dir / f"{prefix}_fold{i+1}.joblib"
        save_model(model, str(path))

    print(f"Saved {len(models)} models to {output_dir}")


# =============================================================================
# MAIN EXECUTION (Step 3 Pipeline)
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data/processed"
    plots_dir = project_root / "plots"
    predictions_dir = project_root / "data/predictions"
    models_dir = project_root / "models"

    # Create directories
    predictions_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 3: BASELINE MODELING & EVALUATION")
    print("=" * 70)

    # 1. Load data
    print("\n[1] Loading data...")
    df = pd.read_parquet(data_dir / "train_features.parquet")
    df = df.sort_values(['date', 'store_nbr', 'family']).reset_index(drop=True)
    print(f"    Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"    Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"    Sales: min={df['sales'].min():.0f}, max={df['sales'].max():.0f}, mean={df['sales'].mean():.1f}")

    # Check for extreme outliers
    extreme_threshold = 10000
    extreme_count = (df['sales'] > extreme_threshold).sum()
    print(f"    Extreme sales (>{extreme_threshold}): {extreme_count:,} ({100*extreme_count/len(df):.2f}%)")

    # 2. Run cross-validation
    print("\n[2] Running Time-Series Cross-Validation with HistGradientBoosting...")
    cv_results, models, oof_df = run_cv(
        df,
        target_col='sales',
        use_log_target=False,  # Direct regression on sales
        verbose=True
    )

    # Get feature names for importance
    feature_cols = get_feature_columns(df)

    # 3. Naive baselines for comparison
    print("\n[3] Computing naive baselines...")
    val_mask = df['date'] >= '2017-08-01'
    naive_results = compute_naive_baselines(df, val_mask)

    print("\n    Naive Baseline Comparison:")
    print("    " + "-" * 50)
    for name, metrics in naive_results.items():
        print(f"    {name:20s}: RMSLE = {metrics['rmsle']:.4f}")
    print(f"    {'HistGradientBoosting':20s}: RMSLE = {cv_results['mean_rmsle']:.4f}")

    improvement = (naive_results['naive_last_week']['rmsle'] - cv_results['mean_rmsle']) / naive_results['naive_last_week']['rmsle'] * 100
    print(f"\n    -> Model improvement over naive: {improvement:.1f}%")

    # 4. Error Analysis
    print("\n[4] Error Analysis...")

    # By family
    family_errors = analyze_errors_by_group(oof_df, 'family')
    print("\n    Top 5 BEST predicted families (lowest RMSLE):")
    print(family_errors[['family', 'rmsle', 'mae', 'count', 'zero_pct']].head().to_string(index=False))
    print("\n    Top 5 WORST predicted families (highest RMSLE):")
    print(family_errors[['family', 'rmsle', 'mae', 'count', 'zero_pct']].tail().to_string(index=False))

    # Promo effect
    promo_analysis = analyze_promotion_effect(oof_df, df)
    print("\n    Promotion Effect on Errors:")
    for key, metrics in promo_analysis.items():
        print(f"    {key}: RMSLE={metrics['rmsle']:.4f}, MAE={metrics['mae']:.2f}, n={metrics['count']:,}")

    # Zero vs non-zero
    zero_analysis = analyze_zero_vs_nonzero(oof_df)
    print("\n    Zero vs Non-Zero Analysis:")
    for key, info in zero_analysis.items():
        if 'rmsle' in info:
            print(f"    {key}: RMSLE={info['rmsle']:.4f}, n={info['count']:,}")
        else:
            print(f"    {key}: mean_pred={info['mean_pred']:.2f}, over-predict%={info['over_predict_pct']:.1f}%, n={info['count']:,}")

    # 5. Feature Importance
    print("\n[5] Feature Importance...")
    feature_imp = aggregate_feature_importance(models, feature_cols)

    print("\n    Top 20 Features (by importance):")
    print(feature_imp[['feature', 'importance_mean', 'importance_pct']].head(20).to_string(index=False))

    # Plot feature importance
    plt.figure(figsize=(12, 10))
    top_30 = feature_imp.head(30)
    plt.barh(range(len(top_30)), top_30['importance_mean'].values, color='steelblue')
    plt.yticks(range(len(top_30)), top_30['feature'].values)
    plt.xlabel('Importance')
    plt.title('Top 30 Feature Importance - Gradient Boosting Baseline')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importance.png', dpi=150)
    plt.close()
    print(f"    Saved: plots/feature_importance.png")

    # 6. Validation Predictions Plot
    print("\n[6] Generating validation prediction plots...")

    # Time series plot for top families
    top_families = df.groupby('family')['sales'].sum().nlargest(5).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, family in enumerate(top_families):
        ax = axes[idx]
        fam_data = oof_df[oof_df['family'] == family].groupby('date').agg({
            'sales': 'sum',
            'pred': 'sum'
        }).reset_index()

        ax.plot(fam_data['date'], fam_data['sales'], 'b-', label='Actual', alpha=0.8)
        ax.plot(fam_data['date'], fam_data['pred'], 'r--', label='Predicted', alpha=0.8)
        ax.set_title(f'{family}')
        ax.legend(fontsize=8)
        ax.tick_params(axis='x', rotation=45)

    # Overall
    ax = axes[5]
    overall = oof_df.groupby('date').agg({'sales': 'sum', 'pred': 'sum'}).reset_index()
    ax.plot(overall['date'], overall['sales'], 'b-', label='Actual', alpha=0.8)
    ax.plot(overall['date'], overall['pred'], 'r--', label='Predicted', alpha=0.8)
    ax.set_title('TOTAL (all families)')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Validation Predictions vs Actual (Aug 2017)', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / 'validation_predictions.png', dpi=150)
    plt.close()
    print(f"    Saved: plots/validation_predictions.png")

    # Scatter plot: actual vs predicted
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # All data
    ax = axes[0]
    sample = oof_df.sample(min(50000, len(oof_df)), random_state=42)
    ax.scatter(sample['sales'], sample['pred'], alpha=0.1, s=1)
    max_val = max(sample['sales'].max(), sample['pred'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect')
    ax.set_xlabel('Actual Sales')
    ax.set_ylabel('Predicted Sales')
    ax.set_title('Actual vs Predicted (all)')
    ax.legend()

    # Log scale
    ax = axes[1]
    ax.scatter(np.log1p(sample['sales']), np.log1p(sample['pred']), alpha=0.1, s=1)
    max_log = max(np.log1p(sample['sales']).max(), np.log1p(sample['pred']).max())
    ax.plot([0, max_log], [0, max_log], 'r--', label='Perfect')
    ax.set_xlabel('log1p(Actual)')
    ax.set_ylabel('log1p(Predicted)')
    ax.set_title('Actual vs Predicted (log scale)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(plots_dir / 'actual_vs_predicted_scatter.png', dpi=150)
    plt.close()
    print(f"    Saved: plots/actual_vs_predicted_scatter.png")

    # Error by family bar chart
    plt.figure(figsize=(14, 6))
    family_errors_sorted = family_errors.sort_values('rmsle')
    colors = ['green' if x < cv_results['mean_rmsle'] else 'red' for x in family_errors_sorted['rmsle']]
    plt.bar(range(len(family_errors_sorted)), family_errors_sorted['rmsle'], color=colors)
    plt.xticks(range(len(family_errors_sorted)), family_errors_sorted['family'], rotation=90)
    plt.axhline(cv_results['mean_rmsle'], color='blue', linestyle='--', label=f"Mean RMSLE: {cv_results['mean_rmsle']:.4f}")
    plt.xlabel('Product Family')
    plt.ylabel('RMSLE')
    plt.title('RMSLE by Product Family')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'error_by_family.png', dpi=150)
    plt.close()
    print(f"    Saved: plots/error_by_family.png")

    # 7. Save models and OOF predictions
    print("\n[7] Saving models and predictions...")
    save_models(models, str(models_dir), prefix='hgb_baseline')
    oof_df.to_parquet(predictions_dir / 'oof_predictions.parquet', index=False)

    # Save label encoders
    joblib.dump(cv_results['label_encoders'], models_dir / 'label_encoders.joblib')

    print(f"    Saved: models/hgb_baseline_fold*.joblib (3 CV models)")
    print(f"    Saved: models/label_encoders.joblib")
    print(f"    Saved: data/predictions/oof_predictions.parquet")

    # 8. Final model for pseudo-test predictions
    print("\n[8] Training final model on all data through 2017-08-15...")
    final_train_mask = df['date'] <= '2017-08-15'
    final_train_df = df[final_train_mask].copy()

    cat_cols = get_categorical_columns(df)
    X_final, _, _, final_encoders = prepare_features(
        final_train_df, feature_cols, cat_cols,
        label_encoders=cv_results['label_encoders']
    )
    y_final = final_train_df['sales'].values

    # Use more iterations for final model
    final_params = get_hgb_params()
    final_params['max_iter'] = 2000
    final_params['n_iter_no_change'] = 100

    final_model, final_info = train_hgb(
        X_final, y_final,
        params=final_params,
        verbose=True
    )
    save_model(final_model, str(models_dir / 'hgb_final.joblib'))
    print(f"    Final model trained with {final_info['n_iter']} iterations")

    # 9. Save error analysis results
    family_errors.to_csv(predictions_dir / 'error_by_family.csv', index=False)
    print(f"    Saved: data/predictions/error_by_family.csv")

    # 10. Summary
    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE - BASELINE MODELING SUMMARY")
    print("=" * 70)
    print(f"\nHistGradientBoosting Cross-Validation Results:")
    print(f"  RMSLE: {cv_results['mean_rmsle']:.4f} +/- {cv_results['std_rmsle']:.4f}")
    print(f"  MAE:   {cv_results['mean_mae']:.2f}")
    print(f"  MAPE:  {cv_results['mean_mape']:.2f}%")
    print(f"  Bias:  {cv_results['mean_bias']:.2f}")

    print(f"\nNaive Baseline Comparison:")
    print(f"  Last Week:     RMSLE = {naive_results['naive_last_week']['rmsle']:.4f}")
    print(f"  Rolling 7d:    RMSLE = {naive_results['naive_rolling_7d']['rmsle']:.4f}")
    print(f"  GradientBoost: RMSLE = {cv_results['mean_rmsle']:.4f} ({improvement:.1f}% improvement)")

    print(f"\nTop 5 Most Important Features:")
    for idx, row in feature_imp.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance_pct']:.2f}%")

    print(f"\nArtifacts saved:")
    print(f"  - models/hgb_baseline_fold*.joblib (3 CV models)")
    print(f"  - models/hgb_final.joblib (final model)")
    print(f"  - models/label_encoders.joblib")
    print(f"  - data/predictions/oof_predictions.parquet")
    print(f"  - data/predictions/error_by_family.csv")
    print(f"  - plots/feature_importance.png")
    print(f"  - plots/validation_predictions.png")
    print(f"  - plots/actual_vs_predicted_scatter.png")
    print(f"  - plots/error_by_family.png")

    print("\n" + "=" * 70)
    print(f"Step 3 Baseline Modeling complete. RMSLE = {cv_results['mean_rmsle']:.4f}")
    print("Ready for Step 4: Advanced Models + Hierarchical Reconciliation?")
    print("=" * 70)
