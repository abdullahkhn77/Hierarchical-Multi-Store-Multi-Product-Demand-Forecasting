# Favorita Hierarchical Demand Forecast

A production-grade hierarchical demand forecasting pipeline for the Corporacion Favorita grocery store sales dataset. This project implements best practices from industry leaders (Walmart, Tesco, etc.) for retail demand forecasting at scale.

## Project Structure

```
favorita-hierarchical-forecast/
├── data/
│   ├── raw/                 # Original Kaggle dataset files
│   └── processed/           # Cleaned & merged data (parquet)
├── notebooks/               # Jupyter notebooks for exploration
├── src/
│   ├── __init__.py
│   ├── data_prep.py        # Data loading & preprocessing
│   ├── eda.py              # Exploratory data analysis
│   ├── features.py         # Feature engineering (Step 2)
│   └── utils.py            # Utility functions
├── plots/                   # EDA visualizations
├── requirements.txt
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

## Feature Engineering Candidates

Based on EDA findings, recommended features for modeling:

### Temporal Features
1. Day of week (cyclical encoding: sin/cos)
2. Month of year (cyclical)
3. Week of year
4. Is weekend flag
5. Is month start/end
6. Days until/since major holiday
7. Fourier terms for weekly/yearly seasonality

### Lag Features
8. Sales lag 1, 7, 14, 28 days
9. Rolling mean (7-day, 14-day, 28-day windows)
10. Rolling std (volatility measure)
11. Same weekday last week/month/year
12. Expanding mean by store-family

### Promotion Features
13. On promotion flag (current)
14. Days since last promotion
15. Promotion frequency (rolling count)
16. Family-specific promotion elasticity coefficient

### External Features
17. Oil price (current)
18. Oil price lagged (7, 14, 28 days)
19. Oil price change (%)
20. Oil price rolling statistics

### Holiday Features
21. Is national/regional/local holiday
22. Holiday type encoding
23. Pre/post holiday flags (-3 to +3 days)
24. Days to next holiday

### Store/Product Features
25. Store type encoding
26. Store cluster
27. City/state encoding
28. Store average sales
29. Family average sales
30. Store-family interaction features

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

## Modeling Approach (Step 2+)

### Baseline Models
1. **Seasonal Naive** - Same day last week/year
2. **Moving Average** - Rolling 7/28-day averages
3. **Linear Regression** - With calendar features

### Advanced Models
4. **LightGBM/XGBoost** - Gradient boosting with lag features
5. **Prophet** - For capturing multiple seasonalities
6. **Temporal Fusion Transformer** - Deep learning for long-horizon

### Hierarchical Reconciliation
- Bottom-up: Forecast store-family, aggregate up
- Top-down: Forecast total, distribute down
- Optimal reconciliation (MinT, ERM)

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

## Next Steps

- [ ] **Step 2**: Feature Engineering + Baseline Models
- [ ] **Step 3**: Advanced ML Models (LightGBM, XGBoost)
- [ ] **Step 4**: Hierarchical Reconciliation
- [ ] **Step 5**: Model Evaluation & Selection
- [ ] **Step 6**: Production Pipeline & Deployment

---

## License

This project uses the Corporacion Favorita dataset from Kaggle.
See competition rules for data usage terms.
