"""
Canadian Population Dynamics Explorer
=====================================
Interactive Streamlit application for exploring how immigration policy
and demographic parameters affect Canadian population projections using
a system dynamics (World3-based) model.

Features:
- Real Canadian economic data (1971-2023) driving exogenous inputs
- Immigration-to-economy feedback loop
- Canada-calibrated nonlinear table functions
- Formal train/test validation with Theil decomposition
- Multiple immigration policy scenarios

Run with:  streamlit run app_population.py
Requires:  streamlit, plotly, numpy, scipy, pandas
"""

import sys
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path so we can import myworld3
sys.path.insert(0, os.path.dirname(__file__))
from myworld3.population_canada import PopulationCanada

# ─────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Canadian Population Dynamics Explorer",
    page_icon="\U0001f341",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_observed_data():
    """Load observed Canadian population and immigration data."""
    data_path = os.path.join(os.path.dirname(__file__), "data",
                             "canada_population.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

@st.cache_data
def load_economic_data():
    """Load observed Canadian economic data for comparison."""
    data_path = os.path.join(os.path.dirname(__file__), "data",
                             "canada_economic.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None


# ─────────────────────────────────────────────────────────────────────
# Simulation runner
# ─────────────────────────────────────────────────────────────────────
def run_simulation(year_min, year_max, dt,
                   p1i, p2i, p3i, p4i,
                   immigration_series, emigration_series,
                   imm_frac_p1, imm_frac_p2, imm_frac_p3, imm_frac_p4,
                   fallback_rate,
                   dcfsn, mtfn, rlt, len_param,
                   observed_population,
                   train_end_year=2005):
    """Run a single population simulation and return results dict."""
    imm_fracs = {
        'p1': imm_frac_p1, 'p2': imm_frac_p2,
        'p3': imm_frac_p3, 'p4': imm_frac_p4,
    }

    model = PopulationCanada(
        year_min=year_min, year_max=year_max, dt=dt,
        immigration_series=immigration_series,
        emigration_series=emigration_series,
        immigration_age_fractions=imm_fracs,
        fallback_immigration_rate=fallback_rate,
        observed_population=observed_population,
        verbose=False,
    )
    model.set_population_table_functions()
    model.init_population_constants(
        p1i=p1i, p2i=p2i, p3i=p3i, p4i=p4i,
        dcfsn=dcfsn, mtfn=mtfn, rlt=rlt, len=len_param,
    )
    model.init_population_variables()
    model.init_exogenous_inputs()
    model.set_population_delay_functions()
    model.run_population()

    summary = model.get_summary()
    validation = model.validate(train_end_year=train_end_year)

    return {
        'time': model.time,
        'pop': model.pop,
        'p1': model.p1, 'p2': model.p2,
        'p3': model.p3, 'p4': model.p4,
        'births': model.b, 'deaths': model.d,
        'immigration': model.immigration,
        'cbr': model.cbr, 'cdr': model.cdr,
        'le': model.le, 'tf': model.tf,
        'immigration_p1': model.immigration_p1,
        'immigration_p2': model.immigration_p2,
        'immigration_p3': model.immigration_p3,
        'immigration_p4': model.immigration_p4,
        # Economic feedback variables
        'iopc': model.iopc, 'sopc': model.sopc,
        'fpc': model.fpc, 'ppolx': model.ppolx,
        'lmf': model.lmf, 'lmhs': model.lmhs,
        'lmp': model.lmp, 'lmc': model.lmc,
        'summary': summary,
        'validation': validation,
    }


# ─────────────────────────────────────────────────────────────────────
# Sidebar: Parameter Controls
# ─────────────────────────────────────────────────────────────────────
st.sidebar.title("Model Parameters")

obs_df = load_observed_data()
econ_df = load_economic_data()
obs_pop = None
obs_imm = None
if obs_df is not None:
    obs_pop = obs_df['Total_Population'].values
    obs_imm = obs_df['Immigration'].values

# --- Time range ---
st.sidebar.header("Simulation Period")
year_min = 1971
col_a, col_b = st.sidebar.columns(2)
year_max = col_a.number_input("End Year", min_value=2000, max_value=2100,
                               value=2050, step=5)

# --- Immigration scenario ---
st.sidebar.header("Immigration Policy")

scenario = st.sidebar.selectbox(
    "Immigration Scenario",
    ["Historical + Projection", "Constant Rate", "Custom Level",
     "Zero Immigration", "High Immigration (1.2%)", "Canada Levels Plan"],
    index=0,
)

# Build immigration series based on scenario
n_steps = int(year_max - year_min)
hist_len = len(obs_imm) if obs_imm is not None else 0

if scenario == "Historical + Projection":
    fallback_rate = st.sidebar.slider(
        "Projection rate (% of population)", 0.0, 3.0, 0.8, 0.1,
        help="After historical data ends, immigration = this % of population"
    ) / 100.0
    immigration_series = obs_imm if obs_imm is not None else None

elif scenario == "Constant Rate":
    const_rate = st.sidebar.slider(
        "Annual immigration rate (%)", 0.0, 3.0, 0.8, 0.1
    ) / 100.0
    fallback_rate = const_rate
    immigration_series = None

elif scenario == "Custom Level":
    custom_level = st.sidebar.number_input(
        "Annual immigrants", min_value=0, max_value=2000000,
        value=400000, step=50000
    )
    immigration_series = np.full(n_steps, custom_level, dtype=float)
    fallback_rate = 0.008

elif scenario == "Zero Immigration":
    immigration_series = np.zeros(n_steps)
    fallback_rate = 0.0

elif scenario == "High Immigration (1.2%)":
    fallback_rate = 0.012
    immigration_series = obs_imm if obs_imm is not None else None

elif scenario == "Canada Levels Plan":
    st.sidebar.caption("Uses historical data + government targets")
    plan_targets = {2024: 485000, 2025: 500000, 2026: 500000}
    if obs_imm is not None:
        series = list(obs_imm)
        for yr in range(year_min + len(obs_imm), int(year_max)):
            series.append(plan_targets.get(yr, 500000))
        immigration_series = np.array(series[:n_steps], dtype=float)
    else:
        immigration_series = np.full(n_steps, 500000, dtype=float)
    fallback_rate = 0.012

# --- Emigration ---
st.sidebar.header("Emigration")
include_emigration = st.sidebar.checkbox("Include emigration", value=False,
    help="Canadian emigration is ~50-80K/year")
if include_emigration:
    emigration_level = st.sidebar.slider(
        "Annual emigration", 0, 200000, 65000, 5000
    )
    emigration_series = np.full(n_steps, emigration_level, dtype=float)
else:
    emigration_series = None

# --- Immigration age distribution ---
st.sidebar.header("Immigration Age Distribution")
with st.sidebar.expander("Age fractions (must sum to 1.0)", expanded=False):
    frac_p1 = st.slider("Ages 0-14 (children)", 0.0, 0.5, 0.18, 0.01,
                         key="frac_p1")
    frac_p2 = st.slider("Ages 15-44 (working)", 0.0, 0.8, 0.62, 0.01,
                         key="frac_p2")
    frac_p3 = st.slider("Ages 45-64 (older working)", 0.0, 0.5, 0.14, 0.01,
                         key="frac_p3")
    frac_p4 = 1.0 - frac_p1 - frac_p2 - frac_p3
    frac_p4 = max(0.0, frac_p4)
    st.write(f"Ages 65+ (elderly): **{frac_p4:.2f}** (auto-calculated)")
    frac_sum = frac_p1 + frac_p2 + frac_p3 + frac_p4
    if not np.isclose(frac_sum, 1.0, atol=0.02):
        st.error(f"Fractions sum to {frac_sum:.2f}, must be ~1.0")

# --- Demographic parameters ---
st.sidebar.header("Demographic Parameters")
with st.sidebar.expander("Fertility & Mortality", expanded=False):
    dcfsn = st.slider("Desired family size (children)", 1.0, 6.0, 4.0, 0.1,
                       help="Normal desired completed family size")
    mtfn = st.slider("Max total fertility", 4.0, 16.0, 12.0, 0.5,
                      help="Biological maximum fertility rate")
    rlt = st.slider("Reproductive lifetime (years)", 15, 40, 30, 1,
                     help="Duration of reproductive period")
    len_param = st.slider("Life expectancy normal (years)", 20, 50, 28, 1,
                           help="Base life expectancy before multipliers")

# --- Initial population (1971 Census) ---
st.sidebar.header("Initial Population (1971)")
with st.sidebar.expander("Age cohort sizes", expanded=False):
    p1i = st.number_input("Ages 0-14", value=6433266, step=100000)
    p2i = st.number_input("Ages 15-44", value=9697470, step=100000)
    p3i = st.number_input("Ages 45-64", value=4068886, step=100000)
    p4i = st.number_input("Ages 65+", value=1762410, step=100000)
    st.write(f"**Total: {p1i+p2i+p3i+p4i:,}**")

# --- Validation split ---
st.sidebar.header("Validation")
train_end = st.sidebar.slider("Train/Test split year", 1990, 2020, 2005, 1,
    help="Train on data before this year, test on data after")


# ─────────────────────────────────────────────────────────────────────
# Run simulation
# ─────────────────────────────────────────────────────────────────────
try:
    results = run_simulation(
        year_min=year_min, year_max=int(year_max), dt=1,
        p1i=p1i, p2i=p2i, p3i=p3i, p4i=p4i,
        immigration_series=immigration_series,
        emigration_series=emigration_series,
        imm_frac_p1=frac_p1, imm_frac_p2=frac_p2,
        imm_frac_p3=frac_p3, imm_frac_p4=frac_p4,
        fallback_rate=fallback_rate,
        dcfsn=dcfsn, mtfn=mtfn, rlt=rlt, len_param=len_param,
        observed_population=obs_pop,
        train_end_year=train_end,
    )
    simulation_ok = True
except Exception as e:
    st.error(f"Simulation failed: {e}")
    import traceback
    st.code(traceback.format_exc())
    simulation_ok = False

# Also run baseline (zero immigration) for comparison
try:
    baseline = run_simulation(
        year_min=year_min, year_max=int(year_max), dt=1,
        p1i=p1i, p2i=p2i, p3i=p3i, p4i=p4i,
        immigration_series=np.zeros(n_steps),
        emigration_series=None,
        imm_frac_p1=0.18, imm_frac_p2=0.62,
        imm_frac_p3=0.14, imm_frac_p4=0.06,
        fallback_rate=0.0,
        dcfsn=dcfsn, mtfn=mtfn, rlt=rlt, len_param=len_param,
        observed_population=None,
    )
    baseline_ok = True
except Exception:
    baseline_ok = False


# ─────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────
st.title("Canadian Population Dynamics Explorer")
st.markdown(
    f"**Scenario:** {scenario} &nbsp;|&nbsp; "
    f"**Period:** {year_min}\u2013{int(year_max)} &nbsp;|&nbsp; "
    f"**Emigration:** {'Yes' if include_emigration else 'No'}"
)

if not simulation_ok:
    st.stop()

# ─── KPI Cards ───
summary = results['summary']
validation = results['validation']
col1, col2, col3, col4, col5 = st.columns(5)

final_pop = summary.get('final_population', 0)
init_pop = summary.get('initial_population', 0)
total_imm = summary.get('total_net_immigration', 0)
avg_growth = summary.get('avg_annual_growth_pct', 0)
imm_share = summary.get('immigration_share_of_growth_pct', 0)

col1.metric("Initial Population", f"{init_pop/1e6:,.1f}M")
col2.metric("Final Population", f"{final_pop/1e6:,.1f}M",
            delta=f"{(final_pop-init_pop)/1e6:+,.1f}M")
col3.metric("Avg Annual Growth", f"{avg_growth:.2f}%")
col4.metric("Total Immigration", f"{total_imm/1e6:,.1f}M")
col5.metric("Imm. Share of Growth", f"{imm_share:.0f}%")

# # Validation summary bar
# if 'test_mape' in validation and not np.isnan(validation.get('test_mape', float('nan'))):
#     v = validation
#     pub_ready = v.get('publication_ready', False)
#     status_color = "green" if pub_ready else "orange"
#     status_text = "PASS" if pub_ready else "NEEDS IMPROVEMENT"
#     st.markdown(
#         f"**Model Validation:** "
#         f"Train MAPE = {v.get('train_mape', 0):.2f}% | "
#         f"Test MAPE = {v.get('test_mape', 0):.2f}% | "
#         f"R\u00b2 = {v.get('r_squared', 0):.4f} | "
#         f"Theil U = {v.get('theil_U', 0):.4f} | "
#         f"Publication: :{status_color}[**{status_text}**]"
#     )


# ─────────────────────────────────────────────────────────────────────
# Tab layout for plots
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Total Population", "Age Structure", "Vital Rates",
    "Immigration Impact", "Economic Feedback", "Data Table"
])

time = results['time']
valid = ~np.isnan(results['pop'])

# ─── Tab 1: Total Population ───
with tab1:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time[valid], y=results['pop'][valid],
        mode='lines', name='Simulated Population',
        line=dict(color='#1f77b4', width=3),
    ))

    if baseline_ok:
        bv = ~np.isnan(baseline['pop'])
        fig.add_trace(go.Scatter(
            x=baseline['time'][bv], y=baseline['pop'][bv],
            mode='lines', name='No Immigration (baseline)',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
        ))

    if obs_df is not None:
        fig.add_trace(go.Scatter(
            x=obs_df['Year'], y=obs_df['Total_Population'],
            mode='markers', name='Observed (Statistics Canada)',
            marker=dict(color='#2ca02c', size=5, symbol='circle'),
        ))

    if obs_df is not None:
        last_obs_year = obs_df['Year'].iloc[-1]
        fig.add_vline(x=last_obs_year, line_dash="dot", line_color="gray",
                      annotation_text="Projection starts",
                      annotation_position="top left")

    # Mark train/test split
    fig.add_vline(x=train_end, line_dash="dash", line_color="red",
                  annotation_text=f"Train/Test split ({train_end})",
                  annotation_position="bottom right")

    fig.update_layout(
        title="Total Population Over Time",
        xaxis_title="Year", yaxis_title="Population",
        hovermode="x unified", template="plotly_white", height=500,
        yaxis=dict(tickformat=","),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Population growth rate
    pop_valid = results['pop'][valid]
    growth_rates = np.diff(pop_valid) / pop_valid[:-1] * 100
    fig_gr = go.Figure()
    fig_gr.add_trace(go.Scatter(
        x=time[valid][1:], y=growth_rates,
        mode='lines', name='Annual Growth Rate',
        line=dict(color='#9467bd', width=2),
    ))
    fig_gr.update_layout(
        title="Annual Population Growth Rate",
        xaxis_title="Year", yaxis_title="Growth Rate (%)",
        template="plotly_white", height=300,
        hovermode="x unified",
    )
    st.plotly_chart(fig_gr, use_container_width=True)


# ─── Tab 2: Age Structure ───
with tab2:
    fig_age = go.Figure()
    for arr, name, color in [
        (results['p1'], 'Ages 0-14', 'rgba(31,119,180,0.6)'),
        (results['p2'], 'Ages 15-44', 'rgba(255,127,14,0.6)'),
        (results['p3'], 'Ages 45-64', 'rgba(44,160,44,0.6)'),
        (results['p4'], 'Ages 65+', 'rgba(214,39,40,0.6)'),
    ]:
        fig_age.add_trace(go.Scatter(
            x=time[valid], y=arr[valid],
            mode='lines', name=name,
            stackgroup='one', line=dict(width=0),
            fillcolor=color,
        ))
    fig_age.update_layout(
        title="Population by Age Group (Stacked)",
        xaxis_title="Year", yaxis_title="Population",
        template="plotly_white", height=500,
        hovermode="x unified", yaxis=dict(tickformat=","),
    )
    st.plotly_chart(fig_age, use_container_width=True)

    # Dependency ratio
    working = results['p2'][valid] + results['p3'][valid]
    dependent = results['p1'][valid] + results['p4'][valid]
    dep_ratio = dependent / np.maximum(working, 1) * 100

    fig_dep = go.Figure()
    fig_dep.add_trace(go.Scatter(
        x=time[valid], y=dep_ratio,
        mode='lines', name='Dependency Ratio',
        line=dict(color='#d62728', width=2),
    ))
    fig_dep.update_layout(
        title="Dependency Ratio (Dependents per 100 Working-Age)",
        xaxis_title="Year", yaxis_title="Dependents per 100 workers",
        template="plotly_white", height=300, hovermode="x unified",
    )
    st.plotly_chart(fig_dep, use_container_width=True)

    # Age share percentages
    total = results['pop'][valid]
    fig_pct = go.Figure()
    for arr, name, color in [
        (results['p1'], 'Ages 0-14', '#1f77b4'),
        (results['p2'], 'Ages 15-44', '#ff7f0e'),
        (results['p3'], 'Ages 45-64', '#2ca02c'),
        (results['p4'], 'Ages 65+', '#d62728'),
    ]:
        fig_pct.add_trace(go.Scatter(
            x=time[valid], y=arr[valid] / total * 100,
            mode='lines', name=name, line=dict(color=color, width=2),
        ))
    fig_pct.update_layout(
        title="Age Group Share (%)",
        xaxis_title="Year", yaxis_title="Share (%)",
        template="plotly_white", height=350, hovermode="x unified",
    )
    st.plotly_chart(fig_pct, use_container_width=True)


# ─── Tab 3: Vital Rates ───
with tab3:
    fig_vital = make_subplots(rows=2, cols=2,
                              subplot_titles=("Births & Deaths",
                                              "Crude Birth & Death Rates",
                                              "Life Expectancy",
                                              "Total Fertility"))

    fig_vital.add_trace(go.Scatter(
        x=time[valid], y=results['births'][valid],
        mode='lines', name='Births', line=dict(color='#2ca02c'),
    ), row=1, col=1)
    fig_vital.add_trace(go.Scatter(
        x=time[valid], y=results['deaths'][valid],
        mode='lines', name='Deaths', line=dict(color='#d62728'),
    ), row=1, col=1)

    fig_vital.add_trace(go.Scatter(
        x=time[valid], y=results['cbr'][valid],
        mode='lines', name='CBR', line=dict(color='#2ca02c'),
    ), row=1, col=2)
    fig_vital.add_trace(go.Scatter(
        x=time[valid], y=results['cdr'][valid],
        mode='lines', name='CDR', line=dict(color='#d62728'),
    ), row=1, col=2)

    # Add observed CBR/CDR from economic data
    if econ_df is not None:
        fig_vital.add_trace(go.Scatter(
            x=econ_df['Year'], y=econ_df['Crude_Birth_Rate'],
            mode='markers', name='Obs CBR',
            marker=dict(color='#2ca02c', size=4, symbol='x'),
        ), row=1, col=2)
        fig_vital.add_trace(go.Scatter(
            x=econ_df['Year'], y=econ_df['Crude_Death_Rate'],
            mode='markers', name='Obs CDR',
            marker=dict(color='#d62728', size=4, symbol='x'),
        ), row=1, col=2)

    le_valid = ~np.isnan(results['le'])
    fig_vital.add_trace(go.Scatter(
        x=time[le_valid], y=results['le'][le_valid],
        mode='lines', name='Life Exp.', line=dict(color='#9467bd'),
    ), row=2, col=1)
    if econ_df is not None:
        fig_vital.add_trace(go.Scatter(
            x=econ_df['Year'], y=econ_df['Life_Expectancy'],
            mode='markers', name='Obs LE',
            marker=dict(color='#9467bd', size=4, symbol='x'),
        ), row=2, col=1)

    tf_valid = ~np.isnan(results['tf'])
    fig_vital.add_trace(go.Scatter(
        x=time[tf_valid], y=results['tf'][tf_valid],
        mode='lines', name='Total Fertility', line=dict(color='#8c564b'),
    ), row=2, col=2)
    if econ_df is not None:
        fig_vital.add_trace(go.Scatter(
            x=econ_df['Year'], y=econ_df['Total_Fertility_Rate'],
            mode='markers', name='Obs TFR',
            marker=dict(color='#8c564b', size=4, symbol='x'),
        ), row=2, col=2)

    fig_vital.update_layout(
        height=650, template="plotly_white",
        showlegend=True, hovermode="x unified",
    )
    fig_vital.update_yaxes(tickformat=",", row=1, col=1)
    st.plotly_chart(fig_vital, use_container_width=True)


# ─── Tab 4: Immigration Impact ───
with tab4:
    imm_valid = ~np.isnan(results['immigration'])
    fig_imm = go.Figure()
    fig_imm.add_trace(go.Scatter(
        x=time[imm_valid], y=results['immigration'][imm_valid],
        mode='lines', name='Model Immigration',
        line=dict(color='#1f77b4', width=2),
    ))
    if obs_df is not None:
        fig_imm.add_trace(go.Scatter(
            x=obs_df['Year'], y=obs_df['Immigration'],
            mode='markers', name='Observed',
            marker=dict(color='#2ca02c', size=5),
        ))
        last_obs_year = obs_df['Year'].iloc[-1]
        fig_imm.add_vline(x=last_obs_year, line_dash="dot",
                          line_color="gray")
    fig_imm.update_layout(
        title="Annual Immigration",
        xaxis_title="Year", yaxis_title="Immigrants/year",
        template="plotly_white", height=400,
        hovermode="x unified", yaxis=dict(tickformat=","),
    )
    st.plotly_chart(fig_imm, use_container_width=True)

    # Immigration by age group
    fig_imm_age = go.Figure()
    for key, name, color in [
        ('immigration_p1', 'Ages 0-14', '#1f77b4'),
        ('immigration_p2', 'Ages 15-44', '#ff7f0e'),
        ('immigration_p3', 'Ages 45-64', '#2ca02c'),
        ('immigration_p4', 'Ages 65+', '#d62728'),
    ]:
        arr = results[key]
        v = ~np.isnan(arr)
        fig_imm_age.add_trace(go.Scatter(
            x=time[v], y=arr[v],
            mode='lines', name=name, stackgroup='one',
        ))
    fig_imm_age.update_layout(
        title="Immigration by Age Group",
        xaxis_title="Year", yaxis_title="Immigrants/year",
        template="plotly_white", height=400,
        hovermode="x unified", yaxis=dict(tickformat=","),
    )
    st.plotly_chart(fig_imm_age, use_container_width=True)

    # Immigration contribution to population growth
    if baseline_ok:
        pop_with = results['pop'][valid]
        bv = ~np.isnan(baseline['pop'])
        pop_without = baseline['pop'][bv]
        min_len = min(len(pop_with), len(pop_without))
        imm_contribution = pop_with[:min_len] - pop_without[:min_len]

        fig_contrib = go.Figure()
        fig_contrib.add_trace(go.Scatter(
            x=time[valid][:min_len], y=imm_contribution,
            mode='lines', name='Immigration contribution',
            fill='tozeroy',
            line=dict(color='rgba(31,119,180,0.8)'),
            fillcolor='rgba(31,119,180,0.3)',
        ))
        fig_contrib.update_layout(
            title="Cumulative Immigration Contribution to Population",
            xaxis_title="Year",
            yaxis_title="Additional Population from Immigration",
            template="plotly_white", height=400,
            hovermode="x unified", yaxis=dict(tickformat=","),
        )
        st.plotly_chart(fig_contrib, use_container_width=True)


# ─── Tab 5: Economic Feedback (NEW) ───
with tab5:
    st.subheader("Economic Variables & Immigration Feedback")
    st.markdown(
        "These plots show the exogenous economic inputs driving the model. "
        "Values are based on **real Canadian data** (1971-2023) with linear "
        "extrapolation for projections. The immigration\u2192economy feedback "
        "adjusts GDP and service output based on labor force composition."
    )

    fig_econ = make_subplots(rows=2, cols=2,
                             subplot_titles=(
                                 "Industrial Output Per Capita (IOPC)",
                                 "Service Output Per Capita (SOPC)",
                                 "Food Per Capita (FPC)",
                                 "Pollution Index (PPOLX)",
                             ))

    iopc_v = ~np.isnan(results['iopc'])
    fig_econ.add_trace(go.Scatter(
        x=time[iopc_v], y=results['iopc'][iopc_v],
        mode='lines', name='IOPC (model)',
        line=dict(color='#1f77b4', width=2),
    ), row=1, col=1)
    if econ_df is not None:
        fig_econ.add_trace(go.Scatter(
            x=econ_df['Year'], y=econ_df['GDP_Per_Capita_2015USD'],
            mode='markers', name='GDP/capita (observed)',
            marker=dict(color='#2ca02c', size=4, symbol='x'),
        ), row=1, col=1)

    sopc_v = ~np.isnan(results['sopc'])
    fig_econ.add_trace(go.Scatter(
        x=time[sopc_v], y=results['sopc'][sopc_v],
        mode='lines', name='SOPC (model)',
        line=dict(color='#ff7f0e', width=2),
    ), row=1, col=2)
    if econ_df is not None:
        fig_econ.add_trace(go.Scatter(
            x=econ_df['Year'], y=econ_df['Service_Output_Per_Capita'],
            mode='markers', name='SOPC (observed)',
            marker=dict(color='#2ca02c', size=4, symbol='x'),
        ), row=1, col=2)

    fpc_v = ~np.isnan(results['fpc'])
    fig_econ.add_trace(go.Scatter(
        x=time[fpc_v], y=results['fpc'][fpc_v],
        mode='lines', name='FPC (model)',
        line=dict(color='#d62728', width=2),
    ), row=2, col=1)
    if econ_df is not None:
        fig_econ.add_trace(go.Scatter(
            x=econ_df['Year'], y=econ_df['Food_Per_Capita_kg'],
            mode='markers', name='FPC (observed)',
            marker=dict(color='#2ca02c', size=4, symbol='x'),
        ), row=2, col=1)

    pp_v = ~np.isnan(results['ppolx'])
    fig_econ.add_trace(go.Scatter(
        x=time[pp_v], y=results['ppolx'][pp_v],
        mode='lines', name='PPOLX',
        line=dict(color='#9467bd', width=2),
    ), row=2, col=2)

    fig_econ.update_layout(
        height=650, template="plotly_white",
        showlegend=True, hovermode="x unified",
    )
    st.plotly_chart(fig_econ, use_container_width=True)

    # Life expectancy multipliers
    st.subheader("Life Expectancy Multipliers")
    st.markdown(
        "These multipliers combine to determine life expectancy: "
        "LE = LEN \u00d7 LMF \u00d7 LMHS \u00d7 LMP \u00d7 LMC"
    )
    fig_mult = go.Figure()
    for key, name, color in [
        ('lmf', 'Food (LMF)', '#2ca02c'),
        ('lmhs', 'Health Services (LMHS)', '#1f77b4'),
        ('lmp', 'Pollution (LMP)', '#9467bd'),
        ('lmc', 'Crowding (LMC)', '#ff7f0e'),
    ]:
        arr = results[key]
        v = ~np.isnan(arr)
        fig_mult.add_trace(go.Scatter(
            x=time[v], y=arr[v],
            mode='lines', name=name,
            line=dict(color=color, width=2),
        ))
    fig_mult.update_layout(
        title="Life Expectancy Multiplier Components",
        xaxis_title="Year", yaxis_title="Multiplier",
        template="plotly_white", height=400,
        hovermode="x unified",
    )
    st.plotly_chart(fig_mult, use_container_width=True)


# # ─── Tab 6: Validation (NEW) ───
# with tab6:
#     st.subheader("Model Validation Report")
#     st.markdown(
#         "Following Barlas (1996) and Sterman (2000) methodology for system "
#         "dynamics model validation. The model is trained on historical data "
#         f"up to **{train_end}** and tested out-of-sample on data after that year."
#     )

#     if 'error' in validation:
#         st.warning(f"Validation could not be performed: {validation['error']}")
#     else:
#         # Metrics cards
#         v = validation
#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric("Train MAPE", f"{v.get('train_mape', 0):.2f}%",
#                    help="Mean Absolute Percentage Error on training data. Target: <5%")
#         c2.metric("Test MAPE", f"{v.get('test_mape', 0):.2f}%",
#                    help="Mean Absolute Percentage Error on test data (out-of-sample). Target: <10%")
#         c3.metric("R\u00b2", f"{v.get('r_squared', 0):.4f}",
#                    help="Coefficient of determination. Target: >0.95")
#         c4.metric("Theil U", f"{v.get('theil_U', 0):.4f}",
#                    help="Theil inequality coefficient. Target: <0.3 (0=perfect)")

#         # Theil decomposition
#         st.subheader("Theil Decomposition")
#         st.markdown(
#             "Error decomposition reveals whether errors are systematic (fixable) "
#             "or random (irreducible). Ideally, Uc (covariance) should dominate."
#         )
#         theil_data = {
#             'Component': ['Um (Bias)', 'Us (Variance)', 'Uc (Covariance)'],
#             'Proportion': [
#                 v.get('theil_Um', 0),
#                 v.get('theil_Us', 0),
#                 v.get('theil_Uc', 0),
#             ],
#             'Meaning': [
#                 'Systematic over/under-prediction',
#                 'Different spread (variance mismatch)',
#                 'Unsystematic error (good - means random)',
#             ],
#         }
#         st.dataframe(pd.DataFrame(theil_data), use_container_width=True,
#                       hide_index=True)

#         # Theil decomposition pie chart
#         fig_theil = go.Figure(data=[go.Pie(
#             labels=['Bias (Um)', 'Variance (Us)', 'Covariance (Uc)'],
#             values=[v.get('theil_Um', 0), v.get('theil_Us', 0),
#                     v.get('theil_Uc', 0)],
#             marker_colors=['#d62728', '#ff7f0e', '#2ca02c'],
#             hole=0.4,
#         )])
#         fig_theil.update_layout(
#             title="Error Decomposition (Theil)",
#             height=350, template="plotly_white",
#         )
#         st.plotly_chart(fig_theil, use_container_width=True)

#         # Publication criteria checklist
#         st.subheader("Publication Readiness Criteria")
#         criteria = v.get('criteria', {})
#         for criterion, passed in criteria.items():
#             icon = "\u2705" if passed else "\u274c"
#             label = criterion.replace('_', ' ').replace('lt ', '< ').replace(
#                 'gt ', '> ').replace('pct', '%')
#             st.markdown(f"{icon} **{label}**")

#         pub_ready = v.get('publication_ready', False)
#         if pub_ready:
#             st.success(
#                 "The model meets all publication criteria. The population "
#                 "dynamics are well-calibrated for Canadian conditions."
#             )
#         else:
#             st.warning(
#                 "Some publication criteria are not met. Consider adjusting "
#                 "parameters or reviewing the table functions calibration."
#             )

#         # Error time series plot
#         if obs_pop is not None:
#             st.subheader("Error Analysis Over Time")
#             n_obs = min(len(obs_pop), len(results['pop']))
#             obs_valid_mask = ~np.isnan(results['pop'][:n_obs])
#             years_obs = time[:n_obs][obs_valid_mask]
#             sim_obs = results['pop'][:n_obs][obs_valid_mask]
#             obs_obs = obs_pop[:n_obs][obs_valid_mask]
#             pct_errors = 100 * (obs_obs - sim_obs) / obs_obs

#             fig_err = make_subplots(rows=1, cols=2,
#                                     subplot_titles=(
#                                         "Percentage Error Over Time",
#                                         "Simulated vs Observed (45\u00b0 line)",
#                                     ))

#             # Color by train/test
#             train_m = years_obs <= train_end
#             test_m = years_obs > train_end

#             fig_err.add_trace(go.Scatter(
#                 x=years_obs[train_m], y=pct_errors[train_m],
#                 mode='lines+markers', name='Train period',
#                 line=dict(color='#1f77b4', width=2),
#                 marker=dict(size=4),
#             ), row=1, col=1)
#             fig_err.add_trace(go.Scatter(
#                 x=years_obs[test_m], y=pct_errors[test_m],
#                 mode='lines+markers', name='Test period',
#                 line=dict(color='#d62728', width=2),
#                 marker=dict(size=4),
#             ), row=1, col=1)
#             fig_err.add_hline(y=0, line_dash="dash", line_color="gray",
#                               row=1, col=1)

#             # 45-degree scatter
#             fig_err.add_trace(go.Scatter(
#                 x=obs_obs[train_m], y=sim_obs[train_m],
#                 mode='markers', name='Train',
#                 marker=dict(color='#1f77b4', size=6),
#             ), row=1, col=2)
#             fig_err.add_trace(go.Scatter(
#                 x=obs_obs[test_m], y=sim_obs[test_m],
#                 mode='markers', name='Test',
#                 marker=dict(color='#d62728', size=6),
#             ), row=1, col=2)
#             # Perfect fit line
#             min_val = min(np.min(obs_obs), np.min(sim_obs))
#             max_val = max(np.max(obs_obs), np.max(sim_obs))
#             fig_err.add_trace(go.Scatter(
#                 x=[min_val, max_val], y=[min_val, max_val],
#                 mode='lines', name='Perfect fit',
#                 line=dict(color='gray', dash='dash'),
#             ), row=1, col=2)

#             fig_err.update_layout(
#                 height=400, template="plotly_white",
#                 hovermode="closest",
#             )
#             fig_err.update_xaxes(title_text="Year", row=1, col=1)
#             fig_err.update_yaxes(title_text="Error (%)", row=1, col=1)
#             fig_err.update_xaxes(title_text="Observed", tickformat=",",
#                                   row=1, col=2)
#             fig_err.update_yaxes(title_text="Simulated", tickformat=",",
#                                   row=1, col=2)
#             st.plotly_chart(fig_err, use_container_width=True)

#         # Additional metrics table
#         st.subheader("Detailed Metrics")
#         metrics_data = {
#             'Metric': [
#                 'Training RMSE', 'Training MAE', 'Training MAPE',
#                 'Test RMSE', 'Test MAE', 'Test MAPE',
#                 'Overall R\u00b2', 'Theil U',
#                 'Worst Error Year', 'Worst Error %',
#             ],
#             'Value': [
#                 f"{v.get('train_rmse', 0):,.0f}",
#                 f"{v.get('train_mae', 0):,.0f}",
#                 f"{v.get('train_mape', 0):.3f}%",
#                 f"{v.get('test_rmse', 0):,.0f}",
#                 f"{v.get('test_mae', 0):,.0f}",
#                 f"{v.get('test_mape', 0):.3f}%",
#                 f"{v.get('r_squared', 0):.6f}",
#                 f"{v.get('theil_U', 0):.6f}",
#                 f"{v.get('max_error_year', 'N/A')}",
#                 f"{v.get('max_error_pct', 0):.3f}%",
#             ],
#             'Target': [
#                 '-', '-', '< 5%',
#                 '-', '-', '< 10%',
#                 '> 0.95', '< 0.3',
#                 '-', '-',
#             ],
#         }
#         st.dataframe(pd.DataFrame(metrics_data), use_container_width=True,
#                       hide_index=True)


# ─── Tab 6: Data Table ───
with tab6:
    st.subheader("Simulation Results Table")
    table_data = {
        'Year': time[valid].astype(int),
        'Population': results['pop'][valid].astype(int),
        'Ages 0-14': results['p1'][valid].astype(int),
        'Ages 15-44': results['p2'][valid].astype(int),
        'Ages 45-64': results['p3'][valid].astype(int),
        'Ages 65+': results['p4'][valid].astype(int),
        'Births': np.round(results['births'][valid]).astype(int),
        'Deaths': np.round(results['deaths'][valid]).astype(int),
        'Immigration': np.round(results['immigration'][valid]).astype(int),
        'Life Exp': np.round(results['le'][valid], 1),
        'TFR': np.round(results['tf'][valid], 2),
    }
    if obs_pop is not None:
        obs_col = np.full(np.sum(valid), np.nan)
        obs_len = min(len(obs_pop), len(obs_col))
        obs_col[:obs_len] = obs_pop[:obs_len]
        table_data['Observed'] = obs_col
        err = obs_col - results['pop'][valid]
        pct_err = np.where(obs_col > 0, 100 * err / obs_col, np.nan)
        table_data['Error (%)'] = np.round(pct_err, 2)

    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True, height=500)

    csv = df_table.to_csv(index=False)
    st.download_button(
        "Download Results CSV",
        csv, "population_simulation_results.csv",
        "text/csv",
    )


# ─────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Canadian Population Dynamics Explorer** \u2014 "
    "Based on World3 system dynamics model adapted for Canadian demographics. "
    "Data source: Statistics Canada, World Bank, FRED. "
    "Model: PopulationCanada with Canada-calibrated table functions, "
    "real economic data inputs, and immigration\u2192economy feedback loop."
)
