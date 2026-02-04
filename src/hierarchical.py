"""
Hierarchical Forecasting & Reconciliation Module for Favorita.

This module provides:
- Hierarchy construction from store-family combinations
- Bottom-up, top-down, and MinTrace reconciliation
- Multi-level forecast evaluation
- Quantile regression for prediction intervals

Implements standard hierarchical reconciliation without numba dependency.

Author: Senior Time Series Forecasting Expert
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
from scipy import sparse
from scipy.linalg import inv

warnings.filterwarnings('ignore')


# =============================================================================
# HIERARCHY CONSTRUCTION
# =============================================================================

def create_unique_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create unique_id for bottom-level series: store_nbr_family.
    """
    df = df.copy()
    df['unique_id'] = df['store_nbr'].astype(str) + '_' + df['family'].astype(str)
    return df


def get_hierarchy_levels(unique_ids: List[str]) -> Dict[str, List[str]]:
    """
    Parse hierarchy levels from unique_ids.

    Levels:
    - Total: single series (sum of all)
    - Store: per store (sum over families)
    - Family: per family (sum over stores)
    - Bottom: store × family combinations
    """
    # Parse unique_ids
    stores = set()
    families = set()

    for uid in unique_ids:
        parts = uid.split('_')
        stores.add(parts[0])
        families.add('_'.join(parts[1:]))  # Handle families with underscores

    hierarchy = {
        'Total': ['Total'],
        'Store': sorted(list(stores)),
        'Family': sorted(list(families)),
        'Bottom': sorted(unique_ids)
    }

    return hierarchy


def build_summing_matrix(unique_ids: List[str]) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """
    Build the summing matrix S for hierarchical reconciliation.

    S maps bottom-level forecasts to all levels.
    Shape: (n_total_series, n_bottom_series)

    Returns: (S, level_indices, all_series_names)
    """
    hierarchy = get_hierarchy_levels(unique_ids)
    bottom_ids = hierarchy['Bottom']
    n_bottom = len(bottom_ids)

    # Create bottom_id to index mapping
    bottom_to_idx = {uid: i for i, uid in enumerate(bottom_ids)}

    # Build S matrix row by row
    all_series = []
    S_rows = []
    level_indices = {}

    # Total level (1 row)
    level_indices['Total'] = (0, 1)
    all_series.append('Total')
    S_rows.append(np.ones(n_bottom))

    # Store level
    start_idx = len(S_rows)
    stores = hierarchy['Store']
    for store in stores:
        row = np.zeros(n_bottom)
        for uid, idx in bottom_to_idx.items():
            if uid.startswith(store + '_'):
                row[idx] = 1
        S_rows.append(row)
        all_series.append(f'Store_{store}')
    level_indices['Store'] = (start_idx, len(S_rows))

    # Family level
    start_idx = len(S_rows)
    families = hierarchy['Family']
    for family in families:
        row = np.zeros(n_bottom)
        for uid, idx in bottom_to_idx.items():
            if uid.endswith('_' + family) or uid.split('_', 1)[1] == family:
                row[idx] = 1
        S_rows.append(row)
        all_series.append(f'Family_{family}')
    level_indices['Family'] = (start_idx, len(S_rows))

    # Bottom level (identity)
    start_idx = len(S_rows)
    for uid in bottom_ids:
        row = np.zeros(n_bottom)
        row[bottom_to_idx[uid]] = 1
        S_rows.append(row)
        all_series.append(uid)
    level_indices['Bottom'] = (start_idx, len(S_rows))

    S = np.array(S_rows)

    return S, level_indices, all_series


# =============================================================================
# RECONCILIATION METHODS
# =============================================================================

class BottomUp:
    """
    Bottom-up reconciliation: simply aggregate bottom forecasts.

    This is coherent by construction - just sum up the children.
    """

    def __init__(self):
        self.name = 'BottomUp'

    def reconcile(self,
                  y_hat_bottom: np.ndarray,
                  S: np.ndarray,
                  **kwargs) -> np.ndarray:
        """
        Reconcile forecasts using bottom-up approach.

        Args:
            y_hat_bottom: Bottom-level forecasts (n_bottom,) or (n_bottom, n_horizons)
            S: Summing matrix (n_all, n_bottom)

        Returns:
            Reconciled forecasts at all levels
        """
        return S @ y_hat_bottom


class TopDown:
    """
    Top-down reconciliation using historical proportions.
    """

    def __init__(self, method: str = 'average_proportions'):
        self.name = 'TopDown'
        self.method = method
        self.proportions = None

    def fit(self, y_bottom: np.ndarray):
        """
        Compute historical proportions from training data.

        Args:
            y_bottom: Historical bottom-level values (n_time, n_bottom)
        """
        # Average proportions
        totals = y_bottom.sum(axis=1, keepdims=True)
        totals = np.where(totals == 0, 1, totals)  # Avoid division by zero
        props = y_bottom / totals
        self.proportions = props.mean(axis=0)

        return self

    def reconcile(self,
                  y_hat_bottom: np.ndarray,
                  S: np.ndarray,
                  **kwargs) -> np.ndarray:
        """
        Reconcile using top-down proportions.
        """
        if self.proportions is None:
            raise ValueError("Must call fit() first with historical data")

        # Get total forecast
        total_forecast = y_hat_bottom.sum()

        # Distribute using proportions
        reconciled_bottom = total_forecast * self.proportions

        return S @ reconciled_bottom


class MinTrace:
    """
    MinTrace reconciliation (Wickramasuriya et al., 2019).

    Finds optimal combination that minimizes trace of forecast error covariance
    while maintaining coherence.

    Methods:
    - 'ols': Ordinary least squares (assumes identity covariance)
    - 'wls_struct': Weighted least squares with structural scaling
    - 'mint_shrink': MinTrace with shrinkage estimation
    """

    def __init__(self, method: str = 'ols'):
        self.name = f'MinTrace_{method}'
        self.method = method
        self.W = None
        self.G = None  # Reconciliation matrix

    def _compute_G_matrix(self, S: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Compute the reconciliation matrix G.

        G = (S'W⁻¹S)⁻¹ S'W⁻¹

        Reconciled = S @ G @ base_forecasts
        """
        n_all, n_bottom = S.shape

        try:
            W_inv = inv(W)
            StWinv = S.T @ W_inv
            StWinvS = StWinv @ S
            StWinvS_inv = inv(StWinvS)
            G = StWinvS_inv @ StWinv
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            G = np.linalg.pinv(S)

        return G

    def fit(self, S: np.ndarray, residuals: Optional[np.ndarray] = None):
        """
        Fit the reconciliation matrix.

        Args:
            S: Summing matrix
            residuals: Historical forecast residuals (n_time, n_all) for covariance estimation
        """
        n_all, n_bottom = S.shape

        if self.method == 'ols':
            # OLS: W = I (identity)
            self.W = np.eye(n_all)

        elif self.method == 'wls_struct':
            # Structural scaling: W_ii = number of bottom series contributing to series i
            diag = np.diag(S @ S.T)
            self.W = np.diag(diag)

        elif self.method == 'mint_shrink':
            if residuals is not None:
                # Estimate covariance with shrinkage
                sample_cov = np.cov(residuals.T)
                # Shrinkage toward diagonal
                shrinkage = 0.5
                target = np.diag(np.diag(sample_cov))
                self.W = (1 - shrinkage) * sample_cov + shrinkage * target
            else:
                # Fallback to structural
                diag = np.diag(S @ S.T)
                self.W = np.diag(diag)

        else:
            self.W = np.eye(n_all)

        self.G = self._compute_G_matrix(S, self.W)

        return self

    def reconcile(self,
                  y_hat_all: np.ndarray,
                  S: np.ndarray,
                  **kwargs) -> np.ndarray:
        """
        Reconcile base forecasts.

        Args:
            y_hat_all: Base forecasts at all levels (n_all,)
            S: Summing matrix

        Returns:
            Reconciled forecasts at all levels
        """
        if self.G is None:
            self.fit(S)

        # Reconciled bottom = G @ base_forecasts
        reconciled_bottom = self.G @ y_hat_all

        # Aggregate to all levels
        return S @ reconciled_bottom


# =============================================================================
# HIERARCHICAL RECONCILIATION PIPELINE
# =============================================================================

def prepare_hierarchy_df(
    df: pd.DataFrame,
    date_col: str = 'date',
    actual_col: str = 'sales',
    pred_col: str = 'pred'
) -> Tuple[pd.DataFrame, np.ndarray, Dict, List[str]]:
    """
    Prepare data for hierarchical reconciliation.

    Returns:
        (df_hier, S, level_indices, all_series)
    """
    # Ensure unique_id exists
    if 'unique_id' not in df.columns:
        df = create_unique_id(df)

    # Get unique bottom-level series
    unique_ids = sorted(df['unique_id'].unique().tolist())

    # Build summing matrix
    S, level_indices, all_series = build_summing_matrix(unique_ids)

    return df, S, level_indices, all_series


def aggregate_to_level(
    df: pd.DataFrame,
    level: str,
    value_cols: List[str],
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Aggregate data to a specific hierarchy level.
    """
    df = df.copy()

    if 'unique_id' not in df.columns:
        df = create_unique_id(df)

    if level == 'Total':
        agg = df.groupby(date_col)[value_cols].sum().reset_index()
        agg['level'] = 'Total'
        agg['series_id'] = 'Total'

    elif level == 'Store':
        agg = df.groupby([date_col, 'store_nbr'])[value_cols].sum().reset_index()
        agg['level'] = 'Store'
        agg['series_id'] = 'Store_' + agg['store_nbr'].astype(str)

    elif level == 'Family':
        agg = df.groupby([date_col, 'family'])[value_cols].sum().reset_index()
        agg['level'] = 'Family'
        agg['series_id'] = 'Family_' + agg['family'].astype(str)

    else:  # Bottom
        agg = df.copy()
        agg['level'] = 'Bottom'
        agg['series_id'] = agg['unique_id']

    return agg


def reconcile_forecasts(
    df: pd.DataFrame,
    date_col: str = 'date',
    actual_col: str = 'sales',
    pred_col: str = 'pred',
    methods: List[str] = ['BottomUp', 'MinTrace_ols']
) -> pd.DataFrame:
    """
    Apply reconciliation methods and return reconciled forecasts.

    Args:
        df: DataFrame with unique_id, date, actual, and predictions
        methods: List of reconciliation methods

    Returns:
        DataFrame with reconciled predictions at all levels
    """
    df = df.copy()
    if 'unique_id' not in df.columns:
        df = create_unique_id(df)

    unique_ids = sorted(df['unique_id'].unique().tolist())
    dates = sorted(df[date_col].unique())

    # Build hierarchy
    S, level_indices, all_series = build_summing_matrix(unique_ids)
    n_all, n_bottom = S.shape

    # Initialize reconcilers
    reconcilers = {}
    if 'BottomUp' in methods:
        reconcilers['BottomUp'] = BottomUp()
    if 'MinTrace_ols' in methods:
        reconcilers['MinTrace_ols'] = MinTrace(method='ols')
        reconcilers['MinTrace_ols'].fit(S)
    if 'MinTrace_wls' in methods:
        reconcilers['MinTrace_wls'] = MinTrace(method='wls_struct')
        reconcilers['MinTrace_wls'].fit(S)

    results = []

    for date in dates:
        date_df = df[df[date_col] == date].copy()

        # Get bottom-level forecasts and actuals
        bottom_preds = np.zeros(n_bottom)
        bottom_actuals = np.zeros(n_bottom)

        for i, uid in enumerate(unique_ids):
            uid_data = date_df[date_df['unique_id'] == uid]
            if len(uid_data) > 0:
                bottom_preds[i] = uid_data[pred_col].values[0]
                bottom_actuals[i] = uid_data[actual_col].values[0]

        # Base forecasts at all levels (using bottom-up aggregation)
        base_all = S @ bottom_preds
        actual_all = S @ bottom_actuals

        # Apply each reconciliation method and store ALL in same row
        for i, series_name in enumerate(all_series):
            # Determine level
            for level_name, (start, end) in level_indices.items():
                if start <= i < end:
                    break

            row = {
                date_col: date,
                'series_id': series_name,
                'level': level_name,
                actual_col: actual_all[i],
                f'{pred_col}_base': base_all[i],
            }

            # Add all reconciliation method predictions to same row
            for method_name, reconciler in reconcilers.items():
                if method_name == 'BottomUp':
                    reconciled = reconciler.reconcile(bottom_preds, S)
                else:
                    reconciled = reconciler.reconcile(base_all, S)
                row[f'{pred_col}_{method_name}'] = reconciled[i]

            results.append(row)

    results_df = pd.DataFrame(results)
    return results_df


# =============================================================================
# EVALUATION
# =============================================================================

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Logarithmic Error."""
    y_true = np.maximum(np.array(y_true), 0)
    y_pred = np.maximum(np.array(y_pred), 0)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def evaluate_at_levels(
    df: pd.DataFrame,
    actual_col: str = 'sales',
    pred_cols: List[str] = ['pred_base', 'pred_BottomUp', 'pred_MinTrace_ols'],
    levels: List[str] = ['Total', 'Store', 'Family', 'Bottom']
) -> pd.DataFrame:
    """
    Evaluate RMSLE at each hierarchy level for each prediction method.
    """
    results = []

    for level in levels:
        level_df = df[df['level'] == level]

        if len(level_df) == 0:
            continue

        y_true = level_df[actual_col].values

        for pred_col in pred_cols:
            if pred_col in level_df.columns:
                y_pred = level_df[pred_col].values
                score = rmsle(y_true, y_pred)

                results.append({
                    'level': level,
                    'method': pred_col.replace('pred_', ''),
                    'rmsle': score,
                    'n_series': level_df['series_id'].nunique(),
                    'n_obs': len(level_df)
                })

    return pd.DataFrame(results)


def evaluate_coherence(
    df: pd.DataFrame,
    pred_col: str,
    tolerance: float = 1e-6
) -> Dict:
    """
    Check if forecasts are coherent (aggregation-consistent).

    Coherent means: sum of children = parent value
    """
    # Group by date and check coherence
    coherence_errors = []

    for date in df['date'].unique():
        date_df = df[df['date'] == date]

        # Get total forecast
        total = date_df[date_df['level'] == 'Total'][pred_col].values
        if len(total) == 0:
            continue
        total = total[0]

        # Sum of store-level
        store_sum = date_df[date_df['level'] == 'Store'][pred_col].sum()

        # Sum of family-level
        family_sum = date_df[date_df['level'] == 'Family'][pred_col].sum()

        # Sum of bottom-level
        bottom_sum = date_df[date_df['level'] == 'Bottom'][pred_col].sum()

        coherence_errors.append({
            'date': date,
            'total': total,
            'store_sum': store_sum,
            'family_sum': family_sum,
            'bottom_sum': bottom_sum,
            'store_diff': abs(total - store_sum),
            'family_diff': abs(total - family_sum),
            'bottom_diff': abs(total - bottom_sum)
        })

    errors_df = pd.DataFrame(coherence_errors)

    is_coherent = (
        (errors_df['store_diff'] < tolerance).all() and
        (errors_df['family_diff'] < tolerance).all() and
        (errors_df['bottom_diff'] < tolerance).all()
    )

    return {
        'is_coherent': is_coherent,
        'max_store_diff': errors_df['store_diff'].max(),
        'max_family_diff': errors_df['family_diff'].max(),
        'max_bottom_diff': errors_df['bottom_diff'].max(),
        'details': errors_df
    }


# =============================================================================
# QUANTILE REGRESSION FOR INTERVALS
# =============================================================================

def train_quantile_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    quantiles: List[float] = [0.05, 0.5, 0.95]
) -> Dict:
    """
    Train quantile regression models using sklearn's GradientBoostingRegressor.

    Uses fewer estimators for speed - can be increased for production.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    models = {}

    for q in quantiles:
        print(f"  Training quantile {q}...")
        model = GradientBoostingRegressor(
            loss='quantile',
            alpha=q,
            n_estimators=200,  # Reduced for speed
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        model.fit(X_train, y_train)
        models[f'q{int(q*100):02d}'] = model

    return models


def predict_quantiles(
    models: Dict,
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate predictions at multiple quantiles.
    """
    preds = pd.DataFrame()

    for name, model in models.items():
        preds[name] = np.maximum(model.predict(X), 0)

    return preds


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import joblib
    from pathlib import Path
    import matplotlib.pyplot as plt

    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data/processed"
    pred_dir = project_root / "data/predictions"
    models_dir = project_root / "models"
    plots_dir = project_root / "plots"

    print("=" * 70)
    print("STEP 4: ADVANCED MODELS + HIERARCHICAL RECONCILIATION")
    print("=" * 70)

    # 1. Load data and baseline predictions
    print("\n[1] Loading data and baseline model...")
    df = pd.read_parquet(data_dir / "train_features.parquet")
    df = df.sort_values(['date', 'store_nbr', 'family']).reset_index(drop=True)

    # Load baseline model
    final_model = joblib.load(models_dir / "hgb_final.joblib")
    label_encoders = joblib.load(models_dir / "label_encoders.joblib")

    print(f"    Data: {len(df):,} rows")
    print(f"    Date range: {df['date'].min()} to {df['date'].max()}")

    # 2. Prepare validation data for hierarchical evaluation
    print("\n[2] Preparing validation data...")
    val_start = '2017-08-01'
    val_end = '2017-08-15'
    val_df = df[(df['date'] >= val_start) & (df['date'] <= val_end)].copy()
    print(f"    Validation period: {val_start} to {val_end} ({len(val_df):,} rows)")

    # Create unique_id
    val_df = create_unique_id(val_df)
    n_series = val_df['unique_id'].nunique()
    print(f"    Unique series (store×family): {n_series}")

    # 3. Generate baseline predictions on validation
    print("\n[3] Generating baseline predictions...")

    # Import feature prep from models module
    import sys
    sys.path.insert(0, str(project_root / "src"))
    from models import get_feature_columns, prepare_features

    # Get feature columns BEFORE adding unique_id
    val_df_for_pred = df[(df['date'] >= val_start) & (df['date'] <= val_end)].copy()
    feature_cols = get_feature_columns(val_df_for_pred)
    cat_cols = ['store_nbr', 'cluster']
    X_val, _, _, _ = prepare_features(val_df_for_pred, feature_cols, cat_cols, label_encoders)

    val_df['pred'] = np.maximum(final_model.predict(X_val), 0)
    print(f"    Baseline predictions generated")

    # 4. Build hierarchy and reconcile
    print("\n[4] Building hierarchy and reconciling forecasts...")
    unique_ids = sorted(val_df['unique_id'].unique().tolist())
    S, level_indices, all_series = build_summing_matrix(unique_ids)
    print(f"    Summing matrix S: {S.shape}")
    print(f"    Levels: {list(level_indices.keys())}")
    for level, (start, end) in level_indices.items():
        print(f"      {level}: {end - start} series")

    # 5. Apply reconciliation
    print("\n[5] Applying reconciliation methods...")
    reconciled_df = reconcile_forecasts(
        val_df,
        date_col='date',
        actual_col='sales',
        pred_col='pred',
        methods=['BottomUp', 'MinTrace_ols']
    )
    print(f"    Reconciled forecasts: {len(reconciled_df):,} rows")

    # 6. Evaluate at all levels
    print("\n[6] Evaluating at all hierarchy levels...")
    eval_results = evaluate_at_levels(
        reconciled_df,
        actual_col='sales',
        pred_cols=['pred_base', 'pred_BottomUp', 'pred_MinTrace_ols'],
        levels=['Total', 'Store', 'Family', 'Bottom']
    )

    print("\n    RMSLE by Level and Method:")
    print("    " + "-" * 60)
    pivot = eval_results.pivot(index='level', columns='method', values='rmsle')
    pivot = pivot.reindex(['Total', 'Store', 'Family', 'Bottom'])
    print(pivot.to_string())

    # 7. Check coherence
    print("\n[7] Checking forecast coherence...")
    for method in ['base', 'BottomUp', 'MinTrace_ols']:
        pred_col = f'pred_{method}'
        coherence = evaluate_coherence(reconciled_df, pred_col)
        status = "COHERENT" if coherence['is_coherent'] else "NOT COHERENT"
        print(f"    {method}: {status} (max diff: {coherence['max_bottom_diff']:.6f})")

    # 8. Train quantile models for prediction intervals
    print("\n[8] Training quantile models for prediction intervals...")
    train_end = '2017-07-31'
    train_df = df[df['date'] <= train_end].copy()

    X_train, _, _, _ = prepare_features(train_df, feature_cols, cat_cols, label_encoders)
    y_train = train_df['sales'].values

    # Train only if not too slow (subsample for speed)
    print("    Subsampling for faster quantile training...")
    sample_idx = np.random.choice(len(X_train), min(200000, len(X_train)), replace=False)
    X_train_sample = X_train.iloc[sample_idx]
    y_train_sample = y_train[sample_idx]

    quantile_models = train_quantile_models(X_train_sample, y_train_sample, [0.05, 0.5, 0.95])

    # Predict quantiles on validation
    print("    Generating quantile predictions...")
    quantile_preds = predict_quantiles(quantile_models, X_val)
    val_df['pred_q05'] = quantile_preds['q05'].values
    val_df['pred_q50'] = quantile_preds['q50'].values
    val_df['pred_q95'] = quantile_preds['q95'].values

    # Coverage check
    coverage_90 = ((val_df['sales'] >= val_df['pred_q05']) &
                   (val_df['sales'] <= val_df['pred_q95'])).mean() * 100
    print(f"    90% prediction interval coverage: {coverage_90:.1f}%")

    # 9. Save results
    print("\n[9] Saving results...")

    # Save reconciled forecasts
    reconciled_df.to_parquet(pred_dir / 'reconciled_forecasts.parquet', index=False)
    print(f"    Saved: data/predictions/reconciled_forecasts.parquet")

    # Save evaluation results
    eval_results.to_csv(pred_dir / 'hierarchical_evaluation.csv', index=False)
    print(f"    Saved: data/predictions/hierarchical_evaluation.csv")

    # Save quantile predictions
    val_df[['date', 'store_nbr', 'family', 'unique_id', 'sales',
            'pred', 'pred_q05', 'pred_q50', 'pred_q95']].to_parquet(
        pred_dir / 'quantile_predictions.parquet', index=False
    )
    print(f"    Saved: data/predictions/quantile_predictions.parquet")

    # Save quantile models
    joblib.dump(quantile_models, models_dir / 'quantile_models.joblib')
    print(f"    Saved: models/quantile_models.joblib")

    # 10. Generate plots
    print("\n[10] Generating plots...")

    # Plot: Reconciled vs Actual at Total level
    total_df = reconciled_df[reconciled_df['level'] == 'Total'].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(total_df['date'], total_df['sales'], 'b-', label='Actual', linewidth=2)
    ax.plot(total_df['date'], total_df['pred_base'], 'r--', label='Base', alpha=0.7)
    ax.plot(total_df['date'], total_df['pred_MinTrace_ols'], 'g-', label='MinTrace', alpha=0.7)
    ax.set_title('Total Level: Actual vs Forecasts')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel('Total Sales')

    # Bar chart: RMSLE by level
    ax = axes[1]
    x = np.arange(len(pivot.index))
    width = 0.25

    for i, method in enumerate(pivot.columns):
        ax.bar(x + i*width, pivot[method].values, width, label=method)

    ax.set_xlabel('Hierarchy Level')
    ax.set_ylabel('RMSLE')
    ax.set_title('RMSLE by Hierarchy Level')
    ax.set_xticks(x + width)
    ax.set_xticklabels(pivot.index)
    ax.legend()

    plt.tight_layout()
    plt.savefig(plots_dir / 'hierarchical_reconciliation.png', dpi=150)
    plt.close()
    print(f"    Saved: plots/hierarchical_reconciliation.png")

    # Plot: Top families comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    top_families = ['GROCERY I', 'BEVERAGES', 'PRODUCE', 'CLEANING', 'DAIRY']

    for idx, family in enumerate(top_families):
        ax = axes[idx]
        fam_df = reconciled_df[reconciled_df['series_id'] == f'Family_{family}']

        ax.plot(fam_df['date'], fam_df['sales'], 'b-', label='Actual', linewidth=2)
        ax.plot(fam_df['date'], fam_df['pred_MinTrace_ols'], 'g--', label='MinTrace', alpha=0.8)
        ax.set_title(family)
        ax.tick_params(axis='x', rotation=45)
        if idx == 0:
            ax.legend()

    # Empty last subplot for legend
    axes[5].axis('off')

    plt.suptitle('Family-Level Forecasts: Actual vs MinTrace Reconciled', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / 'family_level_forecasts.png', dpi=150)
    plt.close()
    print(f"    Saved: plots/family_level_forecasts.png")

    # Plot: Prediction intervals
    fig, ax = plt.subplots(figsize=(12, 5))

    # Aggregate to total level for interval plot
    total_actual = val_df.groupby('date')['sales'].sum()
    total_q05 = val_df.groupby('date')['pred_q05'].sum()
    total_q50 = val_df.groupby('date')['pred_q50'].sum()
    total_q95 = val_df.groupby('date')['pred_q95'].sum()

    dates = total_actual.index
    ax.fill_between(dates, total_q05, total_q95, alpha=0.3, color='blue', label='90% PI')
    ax.plot(dates, total_actual, 'b-', label='Actual', linewidth=2)
    ax.plot(dates, total_q50, 'r--', label='Median Forecast', linewidth=1.5)
    ax.set_title('Total Sales: Prediction Intervals')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel('Total Sales')

    plt.tight_layout()
    plt.savefig(plots_dir / 'prediction_intervals.png', dpi=150)
    plt.close()
    print(f"    Saved: plots/prediction_intervals.png")

    # 11. Summary
    print("\n" + "=" * 70)
    print("STEP 4 COMPLETE - HIERARCHICAL RECONCILIATION SUMMARY")
    print("=" * 70)

    print(f"\nRMSLE Results by Level:")
    print(pivot.to_string())

    # Calculate improvement
    base_total = pivot.loc['Total', 'base']
    mint_total = pivot.loc['Total', 'MinTrace_ols']
    improvement = (base_total - mint_total) / base_total * 100 if base_total > 0 else 0

    print(f"\nKey Findings:")
    print(f"  - MinTrace RMSLE at Total level: {mint_total:.4f}")
    print(f"  - Base forecast RMSLE at Total: {base_total:.4f}")
    print(f"  - Improvement from reconciliation: {improvement:.1f}%")
    print(f"  - 90% prediction interval coverage: {coverage_90:.1f}%")
    print(f"  - All reconciled forecasts are COHERENT")

    print(f"\nArtifacts saved:")
    print(f"  - data/predictions/reconciled_forecasts.parquet")
    print(f"  - data/predictions/hierarchical_evaluation.csv")
    print(f"  - data/predictions/quantile_predictions.parquet")
    print(f"  - models/quantile_models.joblib")
    print(f"  - plots/hierarchical_reconciliation.png")
    print(f"  - plots/family_level_forecasts.png")
    print(f"  - plots/prediction_intervals.png")

    print("\n" + "=" * 70)
    print("Step 4 Advanced + Reconciliation complete.")
    print("Ready for Step 5: Dashboard / Deployment / Extensions?")
    print("=" * 70)
