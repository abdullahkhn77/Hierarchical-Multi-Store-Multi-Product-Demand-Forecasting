# Favorita Store Sales - EDA Report

## Dataset Overview

| Metric | Value |
|--------|-------|
| Time Range | 2013-01-01 to 2017-08-15 |
| Number of Days | 1,684 |
| Number of Stores | 54 |
| Number of Product Families | 33 |
| Total Records | 3,000,888 |
| Memory Usage | 131.7 MB |
| Zero Sales Ratio | **31.3%** |

## Key Insights

### 1. Sales Distribution

| Statistic | Value |
|-----------|-------|
| Mean | 357.78 |
| Median | 11.00 |
| Std Dev | 1,102.00 |
| Max | 124,717.00 |
| Skewness | 7.36 |

- Extremely right-skewed distribution (skewness = 7.36)
- Mean is 32x larger than median - heavy tail with outliers
- Log transformation is essential for modeling

### 2. Seasonality Patterns

**Weekly Pattern:**
- Peak: **Sunday** (avg sales: 463.1)
- Lowest: **Thursday** (avg sales: 283.5)
- Weekend effect clearly visible

**Monthly Pattern:**
- Peak: **December** (avg sales: 453.7) - holiday shopping
- Lowest: **February** (avg sales: 320.9)
- Strong yearly seasonality

### 3. Promotion Effect - CRITICAL FINDING

| Metric | Value |
|--------|-------|
| No Promo Average | 158.25 |
| With Promo Average | 1,137.69 |
| **Overall Lift** | **618.9%** |

**Top 5 Families by Promotion Lift:**
1. SCHOOL AND OFFICE SUPPLIES: +4,257%
2. BABY CARE: +1,415%
3. PET SUPPLIES: +352%
4. HOME AND KITCHEN II: +301%
5. PRODUCE: +208%

**Key Insight:** Promotion is the single most important predictor of sales in this dataset. Some categories see 40x+ sales increases during promotions.

### 4. Oil Price Correlation

| Metric | Value |
|--------|-------|
| Overall Correlation | **r = -0.627** |

- Strong negative correlation between oil prices and sales
- Ecuador's economy is highly oil-dependent
- When oil prices fall, consumer spending typically increases
- This is a critical exogenous variable for forecasting

### 5. Holiday Effects

| Metric | Value |
|--------|-------|
| Normal Day Average | 352.16 |
| Holiday Average | 389.69 |
| **Holiday Lift** | **+10.7%** |

- Moderate positive effect from holidays
- Pre-holiday and post-holiday effects likely present

### 6. Zero-Inflation / Intermittency - MAJOR CHALLENGE

**Overall:** 31.3% of all records have zero sales

**Top 5 Sparse Families:**
| Family | Zero Ratio |
|--------|------------|
| BOOKS | 97.0% |
| BABY CARE | 94.1% |
| SCHOOL AND OFFICE SUPPLIES | 74.1% |
| HOME APPLIANCES | 73.5% |
| LADIESWEAR | 59.8% |

**Modeling Implications:**
- Standard regression will underperform
- Consider: Tweedie regression, zero-inflated models, or two-stage (classification + regression)
- May need separate models for sparse vs dense categories

### 7. Store Analysis

**By Store Type:**
| Type | Avg Sales |
|------|-----------|
| A | 705.88 |
| D | 350.98 |
| B | 326.74 |
| E | 269.12 |
| C | 197.26 |

Type A stores generate 3.5x more sales than Type C stores.

---

## Feature Engineering Candidates

Based on EDA findings, here are recommended features for modeling:

### Temporal Features (High Priority)
1. Day of week (cyclical sin/cos encoding)
2. Month of year (cyclical)
3. Week of year
4. Is weekend flag
5. Is month start/end
6. Days until/since major holiday
7. Fourier terms for weekly/yearly seasonality

### Lag Features (High Priority)
8. Sales lag 1, 7, 14, 28 days
9. Rolling mean (7-day, 28-day windows)
10. Rolling std (volatility)
11. Same weekday last week/month/year
12. Expanding mean by store-family

### Promotion Features (CRITICAL - Highest Priority)
13. On promotion flag (binary: onpromotion > 0)
14. Number of items on promotion
15. Days since last promotion
16. Days until next promotion
17. Promotion frequency (rolling count)
18. Family-specific promotion elasticity coefficient

### External Features (High Priority for Ecuador)
19. Oil price (current)
20. Oil price lagged (7, 14, 28 days)
21. Oil price change (%)
22. Oil price rolling mean/std

### Holiday Features
23. Is national/regional/local holiday
24. Holiday type encoding
25. Pre/post holiday flags (-3 to +3 days)
26. Days to next holiday

### Store/Product Features
27. Store type encoding (A > D > B > E > C)
28. Store cluster (1-17)
29. City/state features
30. Family zero-inflation rate
31. Store-family historical average

### Hierarchy Features
32. Store-level aggregates (total store sales)
33. Family-level aggregates (total family sales)
34. Cross-store family patterns

---

## Plots Generated

All visualizations saved to `plots/` directory:

| Plot | Description |
|------|-------------|
| `total_sales_timeseries.png` | Overall sales trend with 7-day MA |
| `top_families_timeseries.png` | Top 10 families over time |
| `sales_distribution.png` | Histograms (normal & log scale) |
| `sales_by_family_boxplot.png` | Distribution by family |
| `seasonality_analysis.png` | Weekly/monthly patterns |
| `dow_month_heatmap.png` | Day-of-week x Month heatmap |
| `promotion_effect.png` | Promotion lift by family |
| `oil_correlation.png` | Oil price relationships |
| `holiday_effect.png` | Holiday impact analysis |
| `store_patterns.png` | Store-level patterns |
| `stl_decomposition.png` | Time series decomposition |

---

## Modeling Recommendations

Based on EDA findings:

1. **Feature Priority**: Promotion > Oil Price > Temporal > Lags
2. **Target Transform**: Log(sales + 1) for regression
3. **Zero-Inflation**: Consider Tweedie or two-stage models
4. **Hierarchy**: Bottom-up forecasting with reconciliation
5. **Validation**: Time-based split (last 16 days = test per Kaggle)

---
*Generated from real Corporacion Favorita dataset (3M rows)*
