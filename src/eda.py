"""
Comprehensive EDA module for Favorita Hierarchical Forecast.
Performs deep exploratory analysis and generates visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from statsmodels.tsa.seasonal import STL
import warnings

from .utils import get_plots_dir, get_data_dir

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def get_basic_stats(df: pd.DataFrame) -> Dict:
    """
    Get basic statistics about the dataset.

    Returns dict with:
    - time_range: (min_date, max_date)
    - n_stores: number of unique stores
    - n_families: number of product families
    - total_rows: total number of records
    - missing_values: dict of column -> missing count
    - zero_sales_ratio: overall ratio of zero sales
    """
    stats = {
        'time_range': (df['date'].min(), df['date'].max()),
        'n_days': df['date'].nunique(),
        'n_stores': df['store_nbr'].nunique(),
        'n_families': df['family'].nunique(),
        'total_rows': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'zero_sales_ratio': (df['sales'] == 0).mean(),
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict()
    }

    # Memory usage
    stats['memory_mb'] = df.memory_usage(deep=True).sum() / (1024*1024)

    return stats


def analyze_zero_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze zero-inflation and intermittency patterns.

    Returns DataFrame with zero sales ratios by family and store.
    """
    # By family
    family_zeros = df.groupby('family').agg({
        'sales': [
            ('zero_ratio', lambda x: (x == 0).mean()),
            ('mean_when_nonzero', lambda x: x[x > 0].mean() if (x > 0).any() else 0),
            ('count', 'count')
        ]
    }).round(4)
    family_zeros.columns = ['zero_ratio', 'mean_when_nonzero', 'count']
    family_zeros = family_zeros.sort_values('zero_ratio', ascending=False)

    return family_zeros


def analyze_store_family_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze store-family combinations for intermittency.

    Finds combinations with >90% zero sales (sparse series).
    """
    combo_stats = df.groupby(['store_nbr', 'family']).agg({
        'sales': [
            ('zero_ratio', lambda x: (x == 0).mean()),
            ('total_sales', 'sum'),
            ('n_days', 'count')
        ]
    }).round(4)
    combo_stats.columns = ['zero_ratio', 'total_sales', 'n_days']
    combo_stats = combo_stats.reset_index()

    # Flag highly intermittent series
    combo_stats['is_sparse'] = combo_stats['zero_ratio'] > 0.9

    return combo_stats


def analyze_sales_distribution(df: pd.DataFrame,
                               save_plots: bool = True,
                               plots_dir: Optional[Path] = None) -> Dict:
    """
    Analyze sales distribution overall and by family.
    """
    if plots_dir is None:
        plots_dir = get_plots_dir()
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Overall statistics
    results['overall'] = {
        'mean': df['sales'].mean(),
        'median': df['sales'].median(),
        'std': df['sales'].std(),
        'min': df['sales'].min(),
        'max': df['sales'].max(),
        'skewness': df['sales'].skew(),
        'kurtosis': df['sales'].kurtosis()
    }

    # By family statistics
    results['by_family'] = df.groupby('family')['sales'].agg([
        'mean', 'median', 'std', 'min', 'max'
    ]).round(2).sort_values('mean', ascending=False)

    if save_plots:
        # Plot 1: Overall histogram (log scale)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Regular scale
        df[df['sales'] > 0]['sales'].hist(bins=100, ax=axes[0], alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Sales')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Sales Distribution (Non-Zero)')

        # Log scale
        df[df['sales'] > 0]['sales'].apply(np.log1p).hist(bins=100, ax=axes[1], alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Log(Sales + 1)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Sales Distribution (Log Scale)')

        plt.tight_layout()
        plt.savefig(plots_dir / 'sales_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Plot 2: Boxplots by family
        top_families = results['by_family'].head(15).index.tolist()
        fig, ax = plt.subplots(figsize=(14, 8))

        df_top = df[df['family'].isin(top_families)]
        family_order = df_top.groupby('family')['sales'].median().sort_values(ascending=False).index

        sns.boxplot(data=df_top, x='family', y='sales', order=family_order, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylabel('Sales')
        ax.set_xlabel('Product Family')
        ax.set_title('Sales Distribution by Top 15 Product Families')

        plt.tight_layout()
        plt.savefig(plots_dir / 'sales_by_family_boxplot.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved sales distribution plots to {plots_dir}")

    return results


def analyze_seasonality(df: pd.DataFrame,
                       save_plots: bool = True,
                       plots_dir: Optional[Path] = None) -> Dict:
    """
    Analyze weekly and yearly seasonality patterns.
    """
    if plots_dir is None:
        plots_dir = get_plots_dir()
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Day of week pattern
    dow_sales = df.groupby('dayofweek')['sales'].mean()
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    results['day_of_week'] = pd.Series(dow_sales.values, index=dow_names)

    # Monthly pattern
    monthly_sales = df.groupby('month')['sales'].mean()
    results['monthly'] = monthly_sales

    # Yearly pattern by year
    if 'year' in df.columns and df['year'].nunique() > 1:
        yearly_sales = df.groupby(['year', 'month'])['sales'].mean().unstack()
        results['year_month'] = yearly_sales

    # Day of week x Month heatmap
    dow_month = df.groupby(['dayofweek', 'month'])['sales'].mean().unstack()
    results['dow_month_heatmap'] = dow_month

    if save_plots:
        # Plot 1: Day of week pattern
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Day of week
        axes[0, 0].bar(dow_names, results['day_of_week'].values, color='steelblue', edgecolor='black')
        axes[0, 0].set_xlabel('Day of Week')
        axes[0, 0].set_ylabel('Average Sales')
        axes[0, 0].set_title('Average Sales by Day of Week')

        # Monthly pattern
        axes[0, 1].plot(results['monthly'].index, results['monthly'].values, 'o-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Average Sales')
        axes[0, 1].set_title('Average Sales by Month')
        axes[0, 1].set_xticks(range(1, 13))

        # Heatmap
        sns.heatmap(dow_month, annot=False, cmap='YlOrRd', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Day of Week')
        axes[1, 0].set_yticklabels(dow_names)
        axes[1, 0].set_title('Sales Heatmap: Day of Week × Month')

        # Total sales time series
        daily_sales = df.groupby('date')['sales'].sum()
        axes[1, 1].plot(daily_sales.index, daily_sales.values, linewidth=0.5, alpha=0.8)
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Total Daily Sales')
        axes[1, 1].set_title('Total Sales Time Series')

        plt.tight_layout()
        plt.savefig(plots_dir / 'seasonality_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Plot 2: Heatmap (larger)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(dow_month, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax)
        ax.set_xlabel('Month')
        ax.set_ylabel('Day of Week')
        ax.set_yticklabels(dow_names)
        ax.set_title('Average Sales: Day of Week × Month')
        plt.tight_layout()
        plt.savefig(plots_dir / 'dow_month_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved seasonality plots to {plots_dir}")

    return results


def analyze_promotion_effect(df: pd.DataFrame,
                            save_plots: bool = True,
                            plots_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Analyze the effect of promotions on sales by family.
    """
    if plots_dir is None:
        plots_dir = get_plots_dir()
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Convert onpromotion to binary (>0 means has promotion)
    df_temp = df.copy()
    df_temp['has_promo'] = (df_temp['onpromotion'] > 0).astype(int)

    # Calculate promotion lift by family
    promo_effect = df_temp.groupby(['family', 'has_promo'])['sales'].agg(['mean', 'count']).reset_index()
    promo_effect = promo_effect.pivot(index='family', columns='has_promo', values='mean')
    promo_effect.columns = ['no_promo_sales', 'promo_sales']

    # Calculate lift
    promo_effect['promo_lift'] = promo_effect['promo_sales'] / promo_effect['no_promo_sales']
    promo_effect['promo_lift_pct'] = (promo_effect['promo_lift'] - 1) * 100
    promo_effect = promo_effect.sort_values('promo_lift_pct', ascending=False)

    if save_plots:
        # Plot promotion lift
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Bar chart of promotion lift
        top_n = 20
        data = promo_effect.head(top_n)

        colors = ['green' if x > 0 else 'red' for x in data['promo_lift_pct']]
        axes[0].barh(range(len(data)), data['promo_lift_pct'], color=colors, edgecolor='black')
        axes[0].set_yticks(range(len(data)))
        axes[0].set_yticklabels(data.index)
        axes[0].set_xlabel('Promotion Lift (%)')
        axes[0].set_title(f'Promotion Effect by Family (Top {top_n})')
        axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        # Side-by-side comparison
        x = np.arange(len(data))
        width = 0.35
        axes[1].bar(x - width/2, data['no_promo_sales'], width, label='No Promo', color='steelblue')
        axes[1].bar(x + width/2, data['promo_sales'], width, label='With Promo', color='orange')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(data.index, rotation=45, ha='right')
        axes[1].set_ylabel('Average Sales')
        axes[1].set_title('Sales: Promo vs No Promo')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(plots_dir / 'promotion_effect.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved promotion effect plots to {plots_dir}")

    return promo_effect


def analyze_oil_correlation(df: pd.DataFrame,
                           save_plots: bool = True,
                           plots_dir: Optional[Path] = None) -> Dict:
    """
    Analyze correlation between oil prices and sales.
    """
    if plots_dir is None:
        plots_dir = get_plots_dir()
    plots_dir.mkdir(parents=True, exist_ok=True)

    if 'dcoilwtico' not in df.columns:
        print("Oil price column not found!")
        return {}

    results = {}

    # Daily aggregates
    daily = df.groupby('date').agg({
        'sales': 'sum',
        'dcoilwtico': 'first'
    }).dropna()

    # Overall correlation
    results['overall_corr'] = daily['sales'].corr(daily['dcoilwtico'])

    # Rolling correlation (30-day window)
    rolling_corr = daily['sales'].rolling(30).corr(daily['dcoilwtico'])
    results['rolling_corr_mean'] = rolling_corr.mean()
    results['rolling_corr_std'] = rolling_corr.std()

    # Correlation by family
    family_corr = []
    for family in df['family'].unique():
        family_daily = df[df['family'] == family].groupby('date').agg({
            'sales': 'sum',
            'dcoilwtico': 'first'
        }).dropna()
        corr = family_daily['sales'].corr(family_daily['dcoilwtico'])
        family_corr.append({'family': family, 'oil_correlation': corr})

    results['by_family'] = pd.DataFrame(family_corr).set_index('family').sort_values('oil_correlation')

    if save_plots:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Scatter plot
        axes[0, 0].scatter(daily['dcoilwtico'], daily['sales'], alpha=0.3, s=10)
        axes[0, 0].set_xlabel('Oil Price (WTI)')
        axes[0, 0].set_ylabel('Total Daily Sales')
        axes[0, 0].set_title(f'Oil Price vs Total Sales (r={results["overall_corr"]:.3f})')

        # Time series comparison (dual axis)
        ax1 = axes[0, 1]
        ax2 = ax1.twinx()
        ax1.plot(daily.index, daily['sales'], 'b-', alpha=0.7, linewidth=0.5, label='Sales')
        ax2.plot(daily.index, daily['dcoilwtico'], 'r-', alpha=0.7, linewidth=0.5, label='Oil Price')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Total Sales', color='blue')
        ax2.set_ylabel('Oil Price', color='red')
        ax1.set_title('Sales and Oil Price Over Time')

        # Rolling correlation
        axes[1, 0].plot(rolling_corr.index, rolling_corr.values, linewidth=0.5)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Rolling Correlation (30-day)')
        axes[1, 0].set_title('Rolling Correlation: Sales vs Oil Price')
        axes[1, 0].set_ylim(-1, 1)

        # Correlation by family
        family_corr_df = results['by_family'].sort_values('oil_correlation')
        colors = ['red' if x < 0 else 'green' for x in family_corr_df['oil_correlation']]
        axes[1, 1].barh(range(len(family_corr_df)), family_corr_df['oil_correlation'], color=colors)
        axes[1, 1].set_yticks(range(len(family_corr_df)))
        axes[1, 1].set_yticklabels(family_corr_df.index, fontsize=8)
        axes[1, 1].set_xlabel('Correlation with Oil Price')
        axes[1, 1].set_title('Oil Price Correlation by Family')
        axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(plots_dir / 'oil_correlation.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved oil correlation plots to {plots_dir}")

    return results


def analyze_holiday_effect(df: pd.DataFrame,
                          save_plots: bool = True,
                          plots_dir: Optional[Path] = None) -> Dict:
    """
    Analyze the effect of holidays on sales.
    """
    if plots_dir is None:
        plots_dir = get_plots_dir()
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Check for holiday columns
    if 'is_holiday' not in df.columns:
        print("Holiday columns not found!")
        return {}

    # Overall holiday effect
    holiday_sales = df.groupby('is_holiday')['sales'].mean()
    results['overall'] = {
        'normal_day_sales': holiday_sales.get(False, 0),
        'holiday_sales': holiday_sales.get(True, 0),
        'holiday_lift': holiday_sales.get(True, 0) / holiday_sales.get(False, 1)
    }

    # By holiday type
    if 'is_national_holiday' in df.columns:
        national_effect = df.groupby('is_national_holiday')['sales'].mean()
        results['national'] = {
            'normal': national_effect.get(False, 0),
            'national_holiday': national_effect.get(True, 0)
        }

    # Pre/post holiday effect (days around holidays)
    if df['is_holiday'].any():
        # Find holiday dates
        holiday_dates = df[df['is_holiday']]['date'].unique()

        # Create day offset from nearest holiday
        df_temp = df.copy()
        df_temp['days_to_holiday'] = df_temp['date'].apply(
            lambda x: min([abs((x - pd.Timestamp(h)).days) for h in holiday_dates[:100]])
            if len(holiday_dates) > 0 else np.nan
        )

        # Only look at days within 7 days of a holiday
        near_holiday = df_temp[df_temp['days_to_holiday'] <= 7]
        results['days_effect'] = near_holiday.groupby('days_to_holiday')['sales'].mean().to_dict()

    if save_plots:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Holiday vs non-holiday
        labels = ['Normal Day', 'Holiday']
        values = [results['overall']['normal_day_sales'], results['overall']['holiday_sales']]
        colors = ['steelblue', 'orange']
        axes[0].bar(labels, values, color=colors, edgecolor='black')
        axes[0].set_ylabel('Average Sales')
        axes[0].set_title(f'Holiday Effect (Lift: {results["overall"]["holiday_lift"]:.2f}x)')

        # Days around holiday effect
        if 'days_effect' in results:
            days = sorted(results['days_effect'].keys())
            sales = [results['days_effect'][d] for d in days]
            axes[1].plot(days, sales, 'o-', linewidth=2, markersize=8)
            axes[1].set_xlabel('Days from Holiday')
            axes[1].set_ylabel('Average Sales')
            axes[1].set_title('Sales by Days from Nearest Holiday')
            axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Holiday')
            axes[1].legend()

        plt.tight_layout()
        plt.savefig(plots_dir / 'holiday_effect.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved holiday effect plots to {plots_dir}")

    return results


def analyze_store_patterns(df: pd.DataFrame,
                          save_plots: bool = True,
                          plots_dir: Optional[Path] = None) -> Dict:
    """
    Analyze sales patterns by store characteristics.
    """
    if plots_dir is None:
        plots_dir = get_plots_dir()
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # By store type
    if 'type' in df.columns:
        results['by_type'] = df.groupby('type')['sales'].agg(['mean', 'sum', 'count']).sort_values('mean', ascending=False)

    # By cluster
    if 'cluster' in df.columns:
        results['by_cluster'] = df.groupby('cluster')['sales'].agg(['mean', 'sum', 'count']).sort_values('mean', ascending=False)

    # By city
    if 'city' in df.columns:
        results['by_city'] = df.groupby('city')['sales'].agg(['mean', 'sum', 'count']).sort_values('sum', ascending=False)

    # By state
    if 'state' in df.columns:
        results['by_state'] = df.groupby('state')['sales'].agg(['mean', 'sum', 'count']).sort_values('sum', ascending=False)

    # By individual store
    results['by_store'] = df.groupby('store_nbr')['sales'].agg(['mean', 'sum', 'count']).sort_values('sum', ascending=False)

    if save_plots:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # By store type
        if 'by_type' in results:
            data = results['by_type']
            axes[0, 0].bar(data.index.astype(str), data['mean'], color='steelblue', edgecolor='black')
            axes[0, 0].set_xlabel('Store Type')
            axes[0, 0].set_ylabel('Average Sales')
            axes[0, 0].set_title('Average Sales by Store Type')

        # By cluster
        if 'by_cluster' in results:
            data = results['by_cluster'].sort_index()
            axes[0, 1].bar(data.index.astype(str), data['mean'], color='seagreen', edgecolor='black')
            axes[0, 1].set_xlabel('Cluster')
            axes[0, 1].set_ylabel('Average Sales')
            axes[0, 1].set_title('Average Sales by Cluster')
            axes[0, 1].tick_params(axis='x', rotation=45)

        # Top 10 cities
        if 'by_city' in results:
            data = results['by_city'].head(10)
            axes[1, 0].barh(range(len(data)), data['sum'], color='coral', edgecolor='black')
            axes[1, 0].set_yticks(range(len(data)))
            axes[1, 0].set_yticklabels(data.index)
            axes[1, 0].set_xlabel('Total Sales')
            axes[1, 0].set_title('Top 10 Cities by Total Sales')

        # Top 10 stores
        data = results['by_store'].head(10)
        axes[1, 1].barh(range(len(data)), data['sum'], color='purple', edgecolor='black')
        axes[1, 1].set_yticks(range(len(data)))
        axes[1, 1].set_yticklabels([f'Store {i}' for i in data.index])
        axes[1, 1].set_xlabel('Total Sales')
        axes[1, 1].set_title('Top 10 Stores by Total Sales')

        plt.tight_layout()
        plt.savefig(plots_dir / 'store_patterns.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved store patterns plots to {plots_dir}")

    return results


def perform_stl_decomposition(df: pd.DataFrame,
                             families: Optional[List[str]] = None,
                             n_families: int = 4,
                             save_plots: bool = True,
                             plots_dir: Optional[Path] = None) -> Dict:
    """
    Perform STL decomposition on top families.
    """
    if plots_dir is None:
        plots_dir = get_plots_dir()
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Get top families by sales if not specified
    if families is None:
        top_families = df.groupby('family')['sales'].sum().nlargest(n_families).index.tolist()
    else:
        top_families = families[:n_families]

    if save_plots:
        fig, axes = plt.subplots(n_families, 4, figsize=(16, 4 * n_families))

        for i, family in enumerate(top_families):
            # Aggregate to daily for this family
            family_daily = df[df['family'] == family].groupby('date')['sales'].sum()
            family_daily = family_daily.asfreq('D').fillna(method='ffill').fillna(0)

            if len(family_daily) < 14:
                print(f"Skipping {family}: not enough data for STL")
                continue

            try:
                # Perform STL decomposition
                stl = STL(family_daily, period=7, robust=True)
                stl_result = stl.fit()

                results[family] = {
                    'trend_strength': 1 - (stl_result.resid.var() / (stl_result.trend + stl_result.resid).var()),
                    'seasonal_strength': 1 - (stl_result.resid.var() / (stl_result.seasonal + stl_result.resid).var()),
                    'residual_std': stl_result.resid.std()
                }

                # Plot
                row = axes[i] if n_families > 1 else axes
                row[0].plot(family_daily.values, linewidth=0.5)
                row[0].set_title(f'{family} - Observed')
                row[0].set_ylabel('Sales')

                row[1].plot(stl_result.trend, linewidth=0.5)
                row[1].set_title('Trend')

                row[2].plot(stl_result.seasonal[:30], linewidth=0.5)  # Show ~1 month of seasonality
                row[2].set_title('Seasonal (Weekly)')

                row[3].plot(stl_result.resid, linewidth=0.5, alpha=0.7)
                row[3].set_title('Residual')

            except Exception as e:
                print(f"Error with STL for {family}: {e}")
                continue

        plt.tight_layout()
        plt.savefig(plots_dir / 'stl_decomposition.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved STL decomposition plots to {plots_dir}")

    return results


def plot_top_families_timeseries(df: pd.DataFrame,
                                n_families: int = 10,
                                save_plots: bool = True,
                                plots_dir: Optional[Path] = None):
    """
    Plot time series for top N families.
    """
    if plots_dir is None:
        plots_dir = get_plots_dir()
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Get top families
    top_families = df.groupby('family')['sales'].sum().nlargest(n_families).index.tolist()

    fig, axes = plt.subplots(n_families // 2 + n_families % 2, 2, figsize=(16, 3 * (n_families // 2 + 1)))
    axes = axes.flatten()

    for i, family in enumerate(top_families):
        family_daily = df[df['family'] == family].groupby('date')['sales'].sum()
        axes[i].plot(family_daily.index, family_daily.values, linewidth=0.5, alpha=0.8)
        axes[i].set_title(family)
        axes[i].set_ylabel('Sales')

    # Hide unused axes
    for j in range(len(top_families), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(plots_dir / 'top_families_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved top families time series to {plots_dir}")


def plot_total_sales_timeseries(df: pd.DataFrame,
                               save_plots: bool = True,
                               plots_dir: Optional[Path] = None):
    """
    Plot total sales time series.
    """
    if plots_dir is None:
        plots_dir = get_plots_dir()
    plots_dir.mkdir(parents=True, exist_ok=True)

    daily_sales = df.groupby('date')['sales'].sum()

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(daily_sales.index, daily_sales.values, linewidth=0.5, alpha=0.8, color='steelblue')
    ax.fill_between(daily_sales.index, daily_sales.values, alpha=0.3)
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Daily Sales')
    ax.set_title('Total Sales Time Series - All Stores & Families')

    # Add rolling average
    rolling_avg = daily_sales.rolling(window=7).mean()
    ax.plot(rolling_avg.index, rolling_avg.values, linewidth=2, color='red', alpha=0.8, label='7-day MA')
    ax.legend()

    plt.tight_layout()
    plt.savefig(plots_dir / 'total_sales_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved total sales time series to {plots_dir}")


def run_full_eda(df: pd.DataFrame,
                save_plots: bool = True,
                plots_dir: Optional[Path] = None) -> Dict:
    """
    Run complete EDA pipeline and return all results.
    """
    if plots_dir is None:
        plots_dir = get_plots_dir()

    print("=" * 70)
    print("FAVORITA HIERARCHICAL FORECAST - COMPREHENSIVE EDA")
    print("=" * 70)

    results = {}

    # Basic stats
    print("\n[1/9] Computing basic statistics...")
    results['basic_stats'] = get_basic_stats(df)

    # Zero sales analysis
    print("[2/9] Analyzing zero sales / intermittency...")
    results['zero_sales'] = analyze_zero_sales(df)
    results['store_family_sparsity'] = analyze_store_family_combinations(df)

    # Sales distribution
    print("[3/9] Analyzing sales distribution...")
    results['sales_dist'] = analyze_sales_distribution(df, save_plots, plots_dir)

    # Seasonality
    print("[4/9] Analyzing seasonality patterns...")
    results['seasonality'] = analyze_seasonality(df, save_plots, plots_dir)

    # Promotion effect
    print("[5/9] Analyzing promotion effects...")
    results['promotion'] = analyze_promotion_effect(df, save_plots, plots_dir)

    # Oil correlation
    print("[6/9] Analyzing oil price correlation...")
    results['oil'] = analyze_oil_correlation(df, save_plots, plots_dir)

    # Holiday effect
    print("[7/9] Analyzing holiday effects...")
    results['holiday'] = analyze_holiday_effect(df, save_plots, plots_dir)

    # Store patterns
    print("[8/9] Analyzing store patterns...")
    results['store'] = analyze_store_patterns(df, save_plots, plots_dir)

    # STL decomposition
    print("[9/9] Performing STL decomposition...")
    results['stl'] = perform_stl_decomposition(df, n_families=4, save_plots=save_plots, plots_dir=plots_dir)

    # Additional plots
    print("\nGenerating additional visualizations...")
    plot_total_sales_timeseries(df, save_plots, plots_dir)
    plot_top_families_timeseries(df, n_families=10, save_plots=save_plots, plots_dir=plots_dir)

    print("\n" + "=" * 70)
    print("EDA COMPLETE!")
    print(f"All plots saved to: {plots_dir}")
    print("=" * 70)

    return results


def generate_eda_report(results: Dict) -> str:
    """
    Generate a markdown report from EDA results.
    """
    stats = results.get('basic_stats', {})

    report = f"""# Favorita Store Sales - EDA Report

## Dataset Overview

| Metric | Value |
|--------|-------|
| Time Range | {stats.get('time_range', ('N/A', 'N/A'))[0]} to {stats.get('time_range', ('N/A', 'N/A'))[1]} |
| Number of Days | {stats.get('n_days', 'N/A'):,} |
| Number of Stores | {stats.get('n_stores', 'N/A')} |
| Number of Product Families | {stats.get('n_families', 'N/A')} |
| Total Records | {stats.get('total_rows', 'N/A'):,} |
| Memory Usage | {stats.get('memory_mb', 0):.1f} MB |
| Zero Sales Ratio | {stats.get('zero_sales_ratio', 0)*100:.1f}% |

## Key Insights

### 1. Sales Distribution
"""

    if 'sales_dist' in results and 'overall' in results['sales_dist']:
        dist = results['sales_dist']['overall']
        report += f"""
- Mean sales: {dist.get('mean', 0):.2f}
- Median sales: {dist.get('median', 0):.2f}
- Std deviation: {dist.get('std', 0):.2f}
- Skewness: {dist.get('skewness', 0):.2f} (right-skewed, typical for retail)
"""

    report += "\n### 2. Seasonality Patterns\n"

    if 'seasonality' in results and 'day_of_week' in results['seasonality']:
        dow = results['seasonality']['day_of_week']
        peak_day = dow.idxmax()
        low_day = dow.idxmin()
        report += f"""
- **Weekly Pattern**: Peak sales on {peak_day}, lowest on {low_day}
- Clear day-of-week effect with weekend variations
"""

    report += "\n### 3. Promotion Effect\n"

    if 'promotion' in results and isinstance(results['promotion'], pd.DataFrame):
        promo = results['promotion']
        avg_lift = promo['promo_lift_pct'].mean()
        report += f"""
- Average promotion lift: {avg_lift:.1f}%
- Promotion effects vary significantly by family
- Key insight: Some families respond much better to promotions than others
"""

    report += "\n### 4. Oil Price Correlation\n"

    if 'oil' in results and 'overall_corr' in results['oil']:
        oil = results['oil']
        report += f"""
- Overall correlation with total sales: {oil.get('overall_corr', 0):.3f}
- Ecuador's economy is oil-dependent, affecting consumer spending
- Rolling correlation shows temporal variation in the relationship
"""

    report += "\n### 5. Holiday Effects\n"

    if 'holiday' in results and 'overall' in results['holiday']:
        hol = results['holiday']['overall']
        report += f"""
- Holiday sales lift: {hol.get('holiday_lift', 1):.2f}x normal day
- Pre-holiday effects visible in surrounding days
"""

    report += "\n### 6. Zero-Inflation / Intermittency\n"

    if 'store_family_sparsity' in results:
        sparse = results['store_family_sparsity']
        sparse_pct = (sparse['is_sparse'].sum() / len(sparse)) * 100 if len(sparse) > 0 else 0
        report += f"""
- {sparse_pct:.1f}% of store-family combinations have >90% zero sales
- Some families (e.g., AUTOMOTIVE) are highly intermittent
- Requires special handling: zero-inflated models or separate modeling
"""

    report += """
## Feature Engineering Candidates

Based on EDA findings, here are recommended features for modeling:

### Temporal Features
1. Day of week (one-hot or cyclical encoding)
2. Month of year (cyclical)
3. Week of year
4. Is weekend flag
5. Is month start/end
6. Days until/since major holiday
7. Fourier terms for weekly/yearly seasonality

### Lag Features
8. Sales lag 1, 7, 14, 28 days
9. Rolling mean (7-day, 28-day windows)
10. Rolling std (volatility)
11. Same weekday last week/month/year
12. Expanding mean by store-family

### Promotion Features
13. On promotion flag
14. Days since last promotion
15. Promotion frequency (rolling count)
16. Family-specific promotion elasticity

### External Features
17. Oil price (current and lagged)
18. Oil price change (%)
19. Oil price rolling statistics

### Holiday Features
20. Is national/regional/local holiday
21. Holiday type encoding
22. Pre/post holiday flags (-3 to +3 days)

### Store/Product Features
23. Store type encoding
24. Store cluster
25. City/state features
26. Family embeddings (learnable or pre-computed)

### Hierarchy Features
27. Store-level aggregates
28. Family-level aggregates
29. Cross-store family patterns

## Plots Generated

All visualizations saved to `plots/` directory:
- `total_sales_timeseries.png` - Overall sales trend
- `top_families_timeseries.png` - Top 10 families over time
- `sales_distribution.png` - Sales histograms (normal & log scale)
- `sales_by_family_boxplot.png` - Distribution by family
- `seasonality_analysis.png` - Weekly/monthly patterns
- `dow_month_heatmap.png` - Day-of-week × Month heatmap
- `promotion_effect.png` - Promotion lift analysis
- `oil_correlation.png` - Oil price relationships
- `holiday_effect.png` - Holiday impact analysis
- `store_patterns.png` - Store-level patterns
- `stl_decomposition.png` - Time series decomposition

---
*Generated by Favorita EDA Pipeline*
"""

    return report


if __name__ == "__main__":
    from .data_prep import load_master_dataframe, create_master_dataframe

    try:
        # Try to load existing master dataframe
        df = load_master_dataframe()
    except FileNotFoundError:
        print("Master dataframe not found. Creating from raw data...")
        df = create_master_dataframe()

    # Run full EDA
    results = run_full_eda(df)

    # Generate report
    report = generate_eda_report(results)
    print(report)
