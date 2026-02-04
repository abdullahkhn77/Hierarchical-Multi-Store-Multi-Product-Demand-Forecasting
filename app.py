"""
Favorita Hierarchical Demand Forecast - Interactive Dashboard

A production-grade Streamlit dashboard for exploring hierarchical forecasts,
comparing reconciliation methods, simulating promotion scenarios, and
understanding business impact.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Favorita Demand Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DATA LOADING (CACHED)
# =============================================================================

@st.cache_data
def load_reconciled_forecasts():
    """Load hierarchical reconciled forecasts."""
    path = Path("data/predictions/reconciled_forecasts.parquet")
    if path.exists():
        df = pd.read_parquet(path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None


@st.cache_data
def load_quantile_predictions():
    """Load bottom-level predictions with quantiles."""
    path = Path("data/predictions/quantile_predictions.parquet")
    if path.exists():
        df = pd.read_parquet(path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None


@st.cache_data
def load_evaluation_results():
    """Load RMSLE evaluation by level."""
    path = Path("data/predictions/hierarchical_evaluation.csv")
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_historical_data():
    """Load historical sales data for context."""
    path = Path("data/processed/train_merged.parquet")
    if path.exists():
        df = pd.read_parquet(path)
        df['date'] = pd.to_datetime(df['date'])
        # Filter to recent period for performance
        df = df[df['date'] >= '2017-06-01']
        return df
    return None


@st.cache_data
def load_error_by_family():
    """Load error analysis by family."""
    path = Path("data/predictions/error_by_family.csv")
    if path.exists():
        return pd.read_csv(path)
    return None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_number(n, decimals=0):
    """Format number with thousands separator."""
    if decimals == 0:
        return f"{int(n):,}"
    return f"{n:,.{decimals}f}"


def calculate_metrics(actual, predicted):
    """Calculate forecast metrics."""
    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1))) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    rmsle = np.sqrt(np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2))
    bias = np.mean(predicted - actual)

    return {
        'MAE': mae,
        'MAPE': mape,
        'RMSE': rmse,
        'RMSLE': rmsle,
        'Bias': bias
    }


def create_forecast_chart(df, actual_col, pred_cols, title, date_col='date'):
    """Create interactive forecast comparison chart."""
    fig = go.Figure()

    # Actual line
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[actual_col],
        mode='lines+markers',
        name='Actual',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))

    # Prediction lines
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, col in enumerate(pred_cols):
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[col],
                mode='lines+markers',
                name=col.replace('pred_', '').replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                marker=dict(size=4)
            ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Sales',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=400
    )

    return fig


def create_interval_chart(df, actual_col, pred_col, lower_col, upper_col, title):
    """Create chart with prediction intervals."""
    fig = go.Figure()

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([df['date'], df['date'][::-1]]),
        y=pd.concat([df[upper_col], df[lower_col][::-1]]),
        fill='toself',
        fillcolor='rgba(0, 176, 246, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% Interval',
        showlegend=True
    ))

    # Actual
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[actual_col],
        mode='lines+markers',
        name='Actual',
        line=dict(color='#1f77b4', width=2)
    ))

    # Prediction
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[pred_col],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Sales',
        hovermode='x unified',
        height=400
    )

    return fig


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.title("📊 Favorita Hierarchical Demand Forecast")
    st.markdown("""
    **Interactive dashboard for exploring coherent hierarchical forecasts across 54 stores and 33 product families.**

    Navigate using the tabs below to explore forecasts, drill down to specific series, simulate promotions, and assess business impact.
    """)

    # Load data
    reconciled_df = load_reconciled_forecasts()
    quantile_df = load_quantile_predictions()
    eval_df = load_evaluation_results()
    error_df = load_error_by_family()

    # Check data availability
    if reconciled_df is None:
        st.error("❌ Could not load forecast data. Please ensure data files exist in `data/predictions/`")
        st.stop()

    # Sidebar
    st.sidebar.header("🔧 Settings")

    # Method selector
    methods = ['BottomUp', 'MinTrace_ols', 'base']
    selected_method = st.sidebar.selectbox(
        "Reconciliation Method",
        methods,
        index=0,
        help="Select which reconciliation method's forecasts to display"
    )

    # Date range
    min_date = reconciled_df['date'].min()
    max_date = reconciled_df['date'].max()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview",
        "🔍 Drill-Down",
        "🎯 What-If Simulator",
        "📊 Uncertainty & Impact"
    ])

    # ==========================================================================
    # TAB 1: OVERVIEW
    # ==========================================================================
    with tab1:
        st.header("Company-Wide Forecast Overview")

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)

        total_df = reconciled_df[reconciled_df['level'] == 'Total'].copy()

        with col1:
            total_actual = total_df['sales'].sum()
            st.metric("Total Actual Sales", format_number(total_actual))

        with col2:
            pred_col = f'pred_{selected_method}'
            total_forecast = total_df[pred_col].sum()
            delta = (total_forecast - total_actual) / total_actual * 100
            st.metric("Total Forecasted", format_number(total_forecast), f"{delta:+.1f}%")

        with col3:
            if eval_df is not None:
                rmsle = eval_df[(eval_df['level'] == 'Total') & (eval_df['method'] == selected_method)]['rmsle'].values
                if len(rmsle) > 0:
                    st.metric("Total RMSLE", f"{rmsle[0]:.4f}")
                else:
                    st.metric("Total RMSLE", "N/A")

        with col4:
            n_series = reconciled_df[reconciled_df['level'] == 'Bottom']['series_id'].nunique()
            st.metric("Bottom Series", format_number(n_series))

        st.markdown("---")

        # Total level chart
        st.subheader("Total Company Sales: Actual vs Forecast")
        fig = create_forecast_chart(
            total_df,
            'sales',
            [f'pred_{m}' for m in methods],
            f'Total Sales - All Methods Comparison'
        )
        st.plotly_chart(fig, use_container_width=True)

        # RMSLE by level chart
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("RMSLE by Hierarchy Level")
            if eval_df is not None:
                level_order = ['Total', 'Store', 'Family', 'Bottom']
                eval_pivot = eval_df.pivot(index='level', columns='method', values='rmsle')
                eval_pivot = eval_pivot.reindex(level_order)

                fig = px.bar(
                    eval_df[eval_df['method'] == selected_method],
                    x='level',
                    y='rmsle',
                    color='level',
                    title=f'RMSLE by Level ({selected_method})',
                    category_orders={'level': level_order}
                )
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Series Count by Level")
            level_counts = reconciled_df.groupby('level')['series_id'].nunique().reset_index()
            level_counts.columns = ['level', 'count']

            fig = px.pie(
                level_counts,
                values='count',
                names='level',
                title='Distribution of Series by Level',
                hole=0.4
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    # ==========================================================================
    # TAB 2: DRILL-DOWN
    # ==========================================================================
    with tab2:
        st.header("Drill-Down by Store / Family")

        # Level selector
        level = st.radio("Select Level", ['Store', 'Family', 'Bottom'], horizontal=True)

        level_df = reconciled_df[reconciled_df['level'] == level].copy()

        # Series selector
        series_list = sorted(level_df['series_id'].unique().tolist())

        if level == 'Store':
            default_series = [s for s in series_list if 'Store_1' in s or 'Store_44' in s][:2]
        elif level == 'Family':
            default_series = [s for s in series_list if 'GROCERY' in s or 'BEVERAGES' in s][:2]
        else:
            default_series = series_list[:2]

        selected_series = st.multiselect(
            f"Select {level} Series (max 6)",
            series_list,
            default=default_series[:2],
            max_selections=6
        )

        if selected_series:
            # Filter data
            plot_df = level_df[level_df['series_id'].isin(selected_series)]

            # Create faceted chart
            n_series = len(selected_series)
            n_cols = min(3, n_series)
            n_rows = (n_series + n_cols - 1) // n_cols

            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=selected_series,
                shared_xaxes=True
            )

            for idx, series in enumerate(selected_series):
                row = idx // n_cols + 1
                col = idx % n_cols + 1

                series_df = plot_df[plot_df['series_id'] == series]

                fig.add_trace(
                    go.Scatter(
                        x=series_df['date'],
                        y=series_df['sales'],
                        mode='lines+markers',
                        name='Actual',
                        line=dict(color='#1f77b4'),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )

                fig.add_trace(
                    go.Scatter(
                        x=series_df['date'],
                        y=series_df[f'pred_{selected_method}'],
                        mode='lines+markers',
                        name=selected_method,
                        line=dict(color='#ff7f0e', dash='dash'),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )

            fig.update_layout(height=300 * n_rows, title_text=f'{level}-Level Forecasts')
            st.plotly_chart(fig, use_container_width=True)

            # Data table
            st.subheader("Detailed Forecast Data")
            display_df = plot_df[['date', 'series_id', 'sales', f'pred_{selected_method}']].copy()
            display_df['error'] = display_df[f'pred_{selected_method}'] - display_df['sales']
            display_df['error_pct'] = (display_df['error'] / (display_df['sales'] + 1) * 100).round(1)

            st.dataframe(display_df.round(2), use_container_width=True, height=300)

            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download Forecast Data (CSV)",
                csv,
                f"forecast_{level.lower()}.csv",
                "text/csv"
            )
        else:
            st.info("👆 Select one or more series to view forecasts")

        # Error by family summary
        if level == 'Family' and error_df is not None:
            st.markdown("---")
            st.subheader("Forecast Error by Product Family")

            fig = px.bar(
                error_df.sort_values('rmsle'),
                x='family',
                y='rmsle',
                color='rmsle',
                color_continuous_scale='RdYlGn_r',
                title='RMSLE by Product Family (Lower is Better)'
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)

    # ==========================================================================
    # TAB 3: WHAT-IF SIMULATOR
    # ==========================================================================
    with tab3:
        st.header("🎯 Promotion What-If Simulator")
        st.markdown("""
        Simulate the impact of promotions on demand. Based on EDA findings, promotions
        increase sales by an average of **618%** (varies by product family).
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Scenario Setup")

            # Get bottom-level data for simulation
            if quantile_df is not None:
                families = sorted(quantile_df['family'].unique().tolist())
                stores = sorted(quantile_df['store_nbr'].unique().tolist())

                sim_family = st.selectbox("Product Family", families, index=families.index('GROCERY I') if 'GROCERY I' in families else 0)
                sim_store = st.selectbox("Store", stores, index=0)

                # Promotion elasticity by family (from EDA)
                promo_elasticity = {
                    'SCHOOL AND OFFICE SUPPLIES': 43.57,
                    'BABY CARE': 15.15,
                    'MAGAZINES': 13.45,
                    'BOOKS': 10.98,
                    'LADIESWEAR': 8.36,
                    'LINGERIE': 7.99,
                    'HOME APPLIANCES': 7.23,
                    'LAWN AND GARDEN': 6.88,
                    'PLAYERS AND ELECTRONICS': 6.22,
                    'HARDWARE': 5.49,
                    'default': 6.18  # average
                }

                elasticity = promo_elasticity.get(sim_family, promo_elasticity['default'])

                st.markdown(f"**Promotion Lift Factor:** {elasticity:.1f}x")

                promo_intensity = st.slider(
                    "Promotion Intensity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="0 = no promotion, 1 = full promotion effect"
                )

                simulate_btn = st.button("🚀 Simulate", type="primary")
            else:
                st.warning("Quantile predictions not available for simulation")
                simulate_btn = False

        with col2:
            st.subheader("Simulation Results")

            if quantile_df is not None and simulate_btn:
                # Filter data for selected store-family
                sim_df = quantile_df[
                    (quantile_df['family'] == sim_family) &
                    (quantile_df['store_nbr'] == sim_store)
                ].copy()

                if len(sim_df) > 0:
                    # Calculate simulated forecast with promotion
                    lift = 1 + (elasticity - 1) * promo_intensity
                    sim_df['pred_simulated'] = sim_df['pred'] * lift
                    sim_df['pred_q05_sim'] = sim_df['pred_q05'] * lift
                    sim_df['pred_q95_sim'] = sim_df['pred_q95'] * lift

                    # Chart
                    fig = go.Figure()

                    # Simulated interval
                    fig.add_trace(go.Scatter(
                        x=pd.concat([sim_df['date'], sim_df['date'][::-1]]),
                        y=pd.concat([sim_df['pred_q95_sim'], sim_df['pred_q05_sim'][::-1]]),
                        fill='toself',
                        fillcolor='rgba(255, 127, 14, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Simulated 90% PI'
                    ))

                    # Original forecast
                    fig.add_trace(go.Scatter(
                        x=sim_df['date'],
                        y=sim_df['pred'],
                        mode='lines+markers',
                        name='Baseline Forecast',
                        line=dict(color='#1f77b4', width=2)
                    ))

                    # Simulated forecast
                    fig.add_trace(go.Scatter(
                        x=sim_df['date'],
                        y=sim_df['pred_simulated'],
                        mode='lines+markers',
                        name=f'With Promotion ({promo_intensity*100:.0f}%)',
                        line=dict(color='#ff7f0e', width=2, dash='dash')
                    ))

                    # Actual
                    fig.add_trace(go.Scatter(
                        x=sim_df['date'],
                        y=sim_df['sales'],
                        mode='lines+markers',
                        name='Actual',
                        line=dict(color='#2ca02c', width=2)
                    ))

                    fig.update_layout(
                        title=f'{sim_family} @ Store {sim_store}: Promotion Simulation',
                        xaxis_title='Date',
                        yaxis_title='Sales',
                        hovermode='x unified',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Impact metrics
                    col_a, col_b, col_c = st.columns(3)

                    baseline_total = sim_df['pred'].sum()
                    simulated_total = sim_df['pred_simulated'].sum()
                    delta_units = simulated_total - baseline_total

                    with col_a:
                        st.metric("Baseline Forecast", format_number(baseline_total))
                    with col_b:
                        st.metric("Simulated Forecast", format_number(simulated_total))
                    with col_c:
                        st.metric("Delta (Units)", f"+{format_number(delta_units)}", f"+{delta_units/baseline_total*100:.1f}%")
                else:
                    st.warning("No data available for selected store-family combination")
            elif quantile_df is not None:
                st.info("👈 Configure scenario and click Simulate")

    # ==========================================================================
    # TAB 4: UNCERTAINTY & BUSINESS IMPACT
    # ==========================================================================
    with tab4:
        st.header("📊 Uncertainty & Business Impact Analysis")

        if quantile_df is not None:
            col1, col2 = st.columns([2, 1])

            with col2:
                st.subheader("Cost Parameters")
                unit_cost = st.number_input("Unit Holding Cost ($)", value=1.0, min_value=0.1, step=0.1)
                stockout_penalty = st.number_input("Stockout Penalty ($)", value=5.0, min_value=0.1, step=0.5)
                safety_factor = st.slider("Safety Stock Factor", min_value=1.0, max_value=2.0, value=1.1, step=0.05)

            with col1:
                st.subheader("Prediction Intervals (90%)")

                # Aggregate to total for visualization
                total_by_date = quantile_df.groupby('date').agg({
                    'sales': 'sum',
                    'pred': 'sum',
                    'pred_q05': 'sum',
                    'pred_q50': 'sum',
                    'pred_q95': 'sum'
                }).reset_index()

                fig = create_interval_chart(
                    total_by_date,
                    'sales',
                    'pred_q50',
                    'pred_q05',
                    'pred_q95',
                    'Total Sales: Forecast with 90% Prediction Interval'
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("Inventory Cost Simulation")

            # Calculate costs
            quantile_df_cost = quantile_df.copy()

            # Safety stock forecast (upper quantile * factor)
            quantile_df_cost['order_qty'] = quantile_df_cost['pred_q95'] * safety_factor

            # Overstock: ordered more than sold
            quantile_df_cost['overstock'] = np.maximum(quantile_df_cost['order_qty'] - quantile_df_cost['sales'], 0)
            quantile_df_cost['overstock_cost'] = quantile_df_cost['overstock'] * unit_cost

            # Stockout: sold more than ordered (demand > order)
            quantile_df_cost['stockout'] = np.maximum(quantile_df_cost['sales'] - quantile_df_cost['order_qty'], 0)
            quantile_df_cost['stockout_cost'] = quantile_df_cost['stockout'] * stockout_penalty

            # Total cost
            quantile_df_cost['total_cost'] = quantile_df_cost['overstock_cost'] + quantile_df_cost['stockout_cost']

            # Compare with naive ordering (just use point forecast)
            quantile_df_cost['order_naive'] = quantile_df_cost['pred']
            quantile_df_cost['overstock_naive'] = np.maximum(quantile_df_cost['order_naive'] - quantile_df_cost['sales'], 0)
            quantile_df_cost['stockout_naive'] = np.maximum(quantile_df_cost['sales'] - quantile_df_cost['order_naive'], 0)
            quantile_df_cost['total_cost_naive'] = (
                quantile_df_cost['overstock_naive'] * unit_cost +
                quantile_df_cost['stockout_naive'] * stockout_penalty
            )

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_cost_pi = quantile_df_cost['total_cost'].sum()
                st.metric("Total Cost (with PI)", f"${format_number(total_cost_pi)}")

            with col2:
                total_cost_naive = quantile_df_cost['total_cost_naive'].sum()
                st.metric("Total Cost (Naive)", f"${format_number(total_cost_naive)}")

            with col3:
                savings = total_cost_naive - total_cost_pi
                savings_pct = savings / total_cost_naive * 100 if total_cost_naive > 0 else 0
                st.metric("Cost Savings", f"${format_number(savings)}", f"{savings_pct:.1f}%")

            with col4:
                coverage = ((quantile_df_cost['sales'] >= quantile_df_cost['pred_q05']) &
                           (quantile_df_cost['sales'] <= quantile_df_cost['pred_q95'])).mean() * 100
                st.metric("90% PI Coverage", f"{coverage:.1f}%")

            # Cost breakdown chart
            cost_by_date = quantile_df_cost.groupby('date').agg({
                'overstock_cost': 'sum',
                'stockout_cost': 'sum',
                'total_cost': 'sum',
                'total_cost_naive': 'sum'
            }).reset_index()

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=cost_by_date['date'],
                y=cost_by_date['overstock_cost'],
                name='Overstock Cost',
                marker_color='#ff7f0e'
            ))

            fig.add_trace(go.Bar(
                x=cost_by_date['date'],
                y=cost_by_date['stockout_cost'],
                name='Stockout Cost',
                marker_color='#d62728'
            ))

            fig.add_trace(go.Scatter(
                x=cost_by_date['date'],
                y=cost_by_date['total_cost_naive'],
                mode='lines+markers',
                name='Naive Strategy Cost',
                line=dict(color='#7f7f7f', dash='dash', width=2)
            ))

            fig.update_layout(
                barmode='stack',
                title='Daily Inventory Cost Breakdown (PI-Based Ordering)',
                xaxis_title='Date',
                yaxis_title='Cost ($)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            # Explanation
            with st.expander("ℹ️ How is this calculated?"):
                st.markdown(f"""
                **Order Strategy:**
                - **PI-Based:** Order = 95th percentile forecast × {safety_factor} (safety factor)
                - **Naive:** Order = Point forecast (no buffer)

                **Cost Calculation:**
                - **Overstock Cost:** (Order - Actual Sales) × ${unit_cost} per unit
                - **Stockout Cost:** (Actual Sales - Order) × ${stockout_penalty} per unit (lost sales penalty)

                The PI-based strategy uses prediction intervals to buffer against uncertainty,
                reducing costly stockouts at the expense of some overstock.
                """)
        else:
            st.warning("Quantile predictions not available for uncertainty analysis")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Favorita Hierarchical Demand Forecast Dashboard | Built with Streamlit & Plotly</p>
        <p>Data: Corporacion Favorita Grocery Sales (Kaggle)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
