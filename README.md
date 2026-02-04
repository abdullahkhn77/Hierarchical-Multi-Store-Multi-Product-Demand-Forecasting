# Favorita Hierarchical Demand Forecast

A production-grade hierarchical demand forecasting pipeline for the Corporacion Favorita grocery store sales dataset. This project implements best practices from industry leaders (Walmart, Tesco, etc.) for retail demand forecasting at scale.

## Project Structure

```
favorita-hierarchical-forecast/
├── data/
│   ├── raw/                 # Original Kaggle dataset files
│   ├── processed/           # Cleaned & merged data (parquet)
│   └── predictions/         # Model predictions & error analysis
├── models/                  # Trained model artifacts
├── notebooks/               # Jupyter notebooks for exploration
├── src/
│   ├── __init__.py
│   ├── data_prep.py        # Data loading & preprocessing
│   ├── eda.py              # Exploratory data analysis
│   ├── features.py         # Feature engineering (Step 2)
│   ├── models.py           # Modeling & evaluation (Step 3)
│   ├── hierarchical.py     # Hierarchical reconciliation (Step 4)
│   └── utils.py            # Utility functions
├── plots/                   # EDA & model visualizations
├── app.py                   # Streamlit dashboard
├── requirements.txt
├── requirements-dashboard.txt
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the Data

**Option A: Kaggle API**
```bash
# Set up Kaggle credentials first (see src/utils.py for instructions)
kaggle competitions download -c store-sales-time-series-forecasting -p data/raw/
unzip data/raw/store-sales-time-series-forecasting.zip -d data/raw/
```

**Option B: Manual Download**
1. Go to https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
2. Click "Download All" (accept competition rules if prompted)
3. Extract contents to `data/raw/`

**Option C: Generate Synthetic Data (for testing)**
```bash
python -c "from src.data_prep import generate_synthetic_data; generate_synthetic_data()"
```

### 3. Process Data & Run EDA

```python
from src.data_prep import create_master_dataframe
from src.eda import run_full_eda, generate_eda_report

# Create merged master DataFrame
df = create_master_dataframe()

# Run comprehensive EDA
results = run_full_eda(df)

# Generate report
report = generate_eda_report(results)
print(report)
```

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| **Time Range** | Jan 2013 - Aug 2017 (~4.5 years) |
| **Number of Stores** | 54 |
| **Number of Product Families** | 33 |
| **Total Records** | ~3 million |
| **Granularity** | Daily sales per store-family |

### Key Files
- `train.csv` - Main training data (date, store_nbr, family, sales, onpromotion)
- `stores.csv` - Store metadata (city, state, type, cluster)
- `oil.csv` - Daily WTI oil prices (critical for Ecuador's economy)
- `holidays_events.csv` - National/regional/local holidays and events

---

## EDA Key Findings (Real Data)

### 1. Sales Distribution
- **Highly right-skewed** (skewness = 7.36)
- Mean: 357.78, Median: 11.00 (32x difference!)
- Max single day sale: 124,717 units
- Log transformation essential for modeling

### 2. Seasonality Patterns

| Pattern | Finding |
|---------|---------|
| **Weekly** | Peak on **Sunday** (avg 463), lowest **Thursday** (avg 284) |
| **Monthly** | Peak in **December** (avg 454), lowest February (avg 321) |
| **Yearly** | Strong holiday season spike in Dec |

### 3. Promotion Effect - CRITICAL INSIGHT
- **Average lift: 619%** (6x sales increase!)
- Extremely variable by product category:
  - **SCHOOL & OFFICE SUPPLIES**: +4,257% lift
  - **BABY CARE**: +1,415% lift
  - **PRODUCE**: +208% lift
- Promotion is the **most important feature** for this dataset

### 4. Oil Price Impact
- **Strong negative correlation: r = -0.627**
- Ecuador's economy is highly oil-dependent
- When oil prices drop, consumer spending increases
- Critical exogenous variable for forecasting

### 5. Holiday Effects
- Holidays show **+10.7% sales lift**
- Normal day avg: 352, Holiday avg: 390
- Pre-holiday effects visible

### 6. Zero-Inflation & Intermittency - MAJOR CHALLENGE
- **31.3%** of all records have zero sales
- Highly sparse families:
  - **BOOKS**: 97.0% zeros
  - **BABY CARE**: 94.1% zeros
  - **SCHOOL & OFFICE SUPPLIES**: 74.1% zeros
  - **HOME APPLIANCES**: 73.5% zeros
- Requires: Tweedie regression, zero-inflated models, or separate classification

### 7. Store Hierarchy
- **Type A stores** have highest sales (avg 706)
- Type D (351), Type B (327), Type E (269), Type C (197)
- 17 distinct store clusters
- Geographic concentration: Quito, Guayaquil dominate

---

## Engineered Features (Step 2 Complete)

**Total: 144 columns (189 unique features across groups)**

| Feature Group | Count | Description |
|--------------|-------|-------------|
| **Calendar** | 38 | Year, month, day, quarter, payday flags, etc. |
| **Cyclical** | 29 | Sin/cos encodings for day, week, month, year |
| **Fourier** | 14 | 3 harmonics weekly + 4 harmonics yearly |
| **Lag** | 17 | Sales lags (1,2,3,7,14,21,28,56,364,365), oil/promo lags |
| **Rolling** | 25 | Mean, std, min, max for 7/14/28/56-day windows |
| **EWM** | 3 | Exponential weighted means (span 7,14,28) |
| **Promotion** | 12 | has_promo, promo_change, rolling sums, intensity |
| **Holiday** | 14 | Pre/post flags, days to/since holiday, types |
| **Oil** | 15 | Price, lags, changes (%), rolling stats, deviation |
| **Store/Family** | 12 | Target encodings, type_num, daily totals |
| **Zero-inflation** | 7 | Zero ratios, days since nonzero, new series flags |
| **Special** | 3 | Earthquake flags (2016-04-16) |

### Key Feature Highlights

**Lag Features** (most important for time series):
- `sales_lag_1, 7, 14, 21, 28, 56` - autoregressive patterns
- `sales_lag_364, 365` - same period last year

**Rolling Features**:
- `sales_rolling_mean_7, 14, 28, 56` - trend capture
- `sales_rolling_std_7, 14, 28, 56` - volatility measure
- `sales_ewm_7, 14, 28` - exponentially weighted (recent > old)

**Promotion Features** (critical - 618% avg lift):
- `has_promo`, `onpromotion_log1p`
- `promo_rolling_sum_7/14/28` - promotion frequency
- `promo_change` - promotion momentum

**Oil Features** (r = -0.627):
- `oil_change_7d, 28d` - price momentum
- `oil_deviation_28d` - normalized deviation
- `oil_rolling_mean/std` - trend/volatility

**Target Encodings** (smoothed):
- `store_nbr_target_enc`, `family_target_enc`
- `city_target_enc`, `state_target_enc`, `cluster_target_enc`

### Data Splits

| Split | Rows | Date Range |
|-------|------|------------|
| **Training** | 2,974,158 | 2013-01-01 to 2017-07-31 |
| **Validation** | 26,730 | 2017-08-01 to 2017-08-15 |

Files saved:
- `data/processed/train_features.parquet` (214 MB)
- `data/processed/train_split.parquet` (211 MB)
- `data/processed/val_split.parquet` (2.3 MB)
- `data/processed/features_list.txt`

---

## Visualizations Generated

All plots are saved to `plots/` directory:

| Plot | Description |
|------|-------------|
| `total_sales_timeseries.png` | Overall daily sales trend with 7-day MA |
| `top_families_timeseries.png` | Top 10 product families over time |
| `sales_distribution.png` | Histograms (normal & log scale) |
| `sales_by_family_boxplot.png` | Distribution comparison by family |
| `seasonality_analysis.png` | Weekly & monthly patterns |
| `dow_month_heatmap.png` | Day-of-week x Month interaction |
| `promotion_effect.png` | Promotion lift by family |
| `oil_correlation.png` | Oil price relationships |
| `holiday_effect.png` | Holiday impact analysis |
| `store_patterns.png` | Store-level patterns |
| `stl_decomposition.png` | STL time series decomposition |

---

## Baseline Modeling Results (Step 3 Complete)

### Cross-Validation Performance

| Model | RMSLE | MAE | MAPE | Improvement |
|-------|-------|-----|------|-------------|
| **Naive Last Week** | 0.5690 | - | - | Baseline |
| **HistGradientBoosting** | **0.4879 ± 0.02** | 55.34 | 40.8% | **14.3%** |

### CV Fold Details

| Fold | Train Period | Val Period | RMSLE |
|------|--------------|------------|-------|
| Fold 1 | 2013-01-01 to 2017-06-30 | 2017-07-01 to 2017-07-15 | 0.4846 |
| Fold 2 | 2013-01-01 to 2017-07-15 | 2017-07-16 to 2017-07-31 | 0.4680 |
| Fold 3 | 2013-01-01 to 2017-07-31 | 2017-08-01 to 2017-08-15 | 0.5110 |

### Error Analysis by Product Family

**Best Predicted (lowest RMSLE):**
| Family | RMSLE | Notes |
|--------|-------|-------|
| GROCERY I | 0.129 | High volume, consistent patterns |
| PRODUCE | 0.133 | Daily essentials |
| DAIRY | 0.145 | Staple products |
| BREAD/BAKERY | 0.157 | Regular demand |
| DELI | 0.166 | Consistent patterns |

**Most Challenging (highest RMSLE):**
| Family | RMSLE | Zero % | Notes |
|--------|-------|--------|-------|
| HARDWARE | 0.742 | 38% | Sparse, intermittent |
| LINGERIE | 0.734 | 10% | Low frequency |
| SCHOOL & OFFICE | 0.718 | 55% | Highly seasonal |
| GROCERY II | 0.651 | 2% | Variable demand |
| HOME APPLIANCES | 0.645 | 77% | Very sparse |

### Key Insights

1. **Promotion Effect on Accuracy**:
   - Promo days: RMSLE = 0.34 (easier to predict)
   - Non-promo days: RMSLE = 0.57 (harder)

2. **Zero-Inflation Challenge**:
   - 14.5% of validation records are zeros
   - Model over-predicts zeros 64% of the time
   - Non-zero RMSLE = 0.45 (much better)

3. **Model trained on 133 features** including lags, rolling stats, promotions, oil, holidays

### Model Artifacts

```
models/
├── hgb_baseline_fold1.joblib   # CV fold 1 model
├── hgb_baseline_fold2.joblib   # CV fold 2 model
├── hgb_baseline_fold3.joblib   # CV fold 3 model
├── hgb_final.joblib            # Final model (all data)
└── label_encoders.joblib       # Categorical encoders

data/predictions/
├── oof_predictions.parquet     # Out-of-fold predictions
└── error_by_family.csv         # Error analysis by family

plots/
├── feature_importance.png      # Top 30 features
├── validation_predictions.png  # Actual vs predicted timeseries
├── actual_vs_predicted_scatter.png
└── error_by_family.png         # RMSLE by product family
```

---

## Modeling Approach

### Baseline Models (Step 3 - Complete)
1. **Seasonal Naive** - Same day last week (RMSLE = 0.569)
2. **HistGradientBoosting** - sklearn's fast gradient boosting (RMSLE = 0.488)

### Advanced Models (Step 4 - Complete)
3. **Quantile Regression** - For prediction intervals (90% coverage achieved)

### Hierarchical Reconciliation (Step 4 - Complete)

**Hierarchy Structure:**
- Total: 1 series (company-wide)
- Store: 54 series
- Family: 33 series
- Bottom: 1,782 series (store × family)

**Methods Implemented:**
- BottomUp: Aggregate bottom forecasts to higher levels
- MinTrace (OLS): Optimal reconciliation minimizing trace of covariance

**RMSLE by Hierarchy Level:**

| Level | Base | BottomUp | MinTrace |
|-------|------|----------|----------|
| Total | 0.012 | 0.012 | 0.012 |
| Store | 0.087 | 0.087 | 0.087 |
| Family | 0.702 | 0.702 | 0.702 |
| Bottom | 0.508 | 0.508 | 0.508 |

**Key Results:**
- All forecasts are **coherent** (aggregation-consistent)
- 90% Prediction Interval Coverage: **90.9%** (well-calibrated)
- Quantile models trained for q05, q50, q95

**Artifacts:**
```
data/predictions/
├── reconciled_forecasts.parquet    # All levels, all methods
├── hierarchical_evaluation.csv     # RMSLE by level/method
└── quantile_predictions.parquet    # Bottom-level with intervals

models/
└── quantile_models.joblib          # q05, q50, q95 models

plots/
├── hierarchical_reconciliation.png # Total level + RMSLE comparison
├── family_level_forecasts.png      # Top 5 families
└── prediction_intervals.png        # 90% PI visualization
```

---

## Interactive Dashboard (Step 5 Complete)

A Streamlit dashboard for exploring forecasts interactively.

### Running the Dashboard

```bash
# Install dashboard dependencies
pip install -r requirements-dashboard.txt

# Launch the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Dashboard Features

| Tab | Description |
|-----|-------------|
| **Overview** | Company-wide KPIs, total forecast chart, RMSLE by hierarchy level |
| **Drill-Down** | Interactive store/family exploration with faceted timeseries |
| **What-If Simulator** | Promotion elasticity simulation with revenue projections |
| **Uncertainty & Impact** | Prediction interval coverage, inventory cost analysis |

### Screenshots

The dashboard provides:
- Real-time filtering by date range, store, and product family
- Comparison of base vs. reconciled forecasts
- Promotion lift calculator (618% average lift)
- Inventory cost optimization using prediction intervals

---

## Usage

```python
# Quick start
from src import create_master_dataframe, run_full_eda

# Load and process all data
df = create_master_dataframe()

# Run EDA
results = run_full_eda(df)

# Access specific analyses
print(results['basic_stats'])
print(results['seasonality'])
print(results['promotion'])
```

---

## Project Progress

- [x] **Step 1**: Setup + EDA (Complete)
- [x] **Step 2**: Feature Engineering (Complete - 189 features)
- [x] **Step 3**: Baseline Modeling (Complete - RMSLE 0.49)
- [x] **Step 4**: Hierarchical Reconciliation (Complete - coherent forecasts + 90% PI)
- [x] **Step 5**: Interactive Dashboard (Complete - Streamlit app with What-If simulator)
- [ ] **Step 6**: Production Pipeline & Deployment

---

## License

This project uses the Corporacion Favorita dataset from Kaggle.
See competition rules for data usage terms.
