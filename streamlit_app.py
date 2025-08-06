import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import BlackScholes as BSP
import MonteCarlo as MC 

st.set_page_config(
    page_icon="📈",
    page_title="Black Scholes & Monte Carlo",
    layout="wide",
    initial_sidebar_state='expanded'
)

st.markdown(
    """<style>
/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px;
    width: auto;
    margin: 0 auto;
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #d4edda; /* Light green */
    color: #155724; /* Dark green */
    margin-right: 10px;
    border-radius: 10px;
}

.metric-put {
    background-color: #f8d7da; /* Light red */
    color: #721c24; /* Dark red */
    border-radius: 10px;
}

/* Style for the value text */
.metric-value {
    font-size: 1.5rem;
    font-weight: bold;
    margin: 0;
}

/* Style for the label text */
.metric-label {
    font-size: 1rem;
    margin-bottom: 4px;
}

</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("📊 Options Pricing and Monte Carlo")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/shlok-thakkar/"
    github_url = "https://www.github.com/shlok1806"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Shlok Thakkar`</a>', unsafe_allow_html=True)
    st.markdown(f'<a href="{github_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Shlok Thakkar`</a>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Shared Parameters")
    current_price = st.number_input("Current Asset Price", value=100.0, min_value=0.01)
    strike = st.number_input("Strike Price", value=100.0, min_value=0.01)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0, min_value=0.01)
    volatility = st.slider("Volatility (σ)", min_value=0.01, max_value=2.0, value=0.2, step=0.01)
    interest_rate = st.slider("Risk-Free Interest Rate", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

    st.markdown("---")
    st.subheader("Monte Carlo Parameters")
    expected_return = st.slider("Expected Annual Return", min_value=-0.5, max_value=0.5, value=interest_rate, step=0.01, help="The expected annual growth rate of the asset. Often assumed to be the risk-free rate in risk-neutral pricing.")
    num_sims = st.number_input("Number of Simulations", value=1000, min_value=100, max_value=50000, step=100)
    num_steps = st.number_input("Number of Time Steps", value=100, min_value=10, max_value=1000, step=10)


    st.markdown("---")
    st.subheader("Heatmap Parameters")
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
    vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=max(0.01, volatility*0.5), step=0.01)
    vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=min(1.0, volatility*1.5), step=0.01)
    
    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)


def heatmap(time_to_maturity, strike_price, risk_free_rate, spot_range, vol_range):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_temp = BSP.BlackScholesPricer(
                time_to_maturity=time_to_maturity,
                strike_price=strike_price,
                curr_price=spot,
                volatility=vol,
                risk_free_rate=risk_free_rate
            )
            bs_temp.run()
            call_prices[i, j] = bs_temp.call_option
            put_prices[i, j] = bs_temp.put_option
    
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="RdYlGn", ax=ax_call, cbar_kws={'label': 'Option Price'})
    ax_call.set_title('Call Option Prices', fontsize=16, fontweight='bold')
    ax_call.set_xlabel('Spot Price', fontsize=12)
    ax_call.set_ylabel('Volatility', fontsize=12)
    
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="RdYlGn_r", ax=ax_put, cbar_kws={'label': 'Option Price'})
    ax_put.set_title('Put Option Prices', fontsize=16, fontweight='bold')
    ax_put.set_xlabel('Spot Price', fontsize=12)
    ax_put.set_ylabel('Volatility', fontsize=12)
    
    return fig_call, fig_put


st.title("Black-Scholes Pricing & Monte Carlo Simulation")

# --- Black-Scholes Section ---
st.header("Black-Scholes Option Pricing")
bs_model = BSP.BlackScholesPricer(
    time_to_maturity=time_to_maturity,
    strike_price=strike,
    curr_price=current_price,
    volatility=volatility,
    risk_free_rate=interest_rate
)
bs_model.run()

col1, col2 = st.columns([1, 1], gap="small")
with col1:
    st.markdown(f'<div class="metric-container metric-call"><div><div class="metric-label">CALL Value</div><div class="metric-value">${bs_model.call_option:.2f}</div></div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-container metric-put"><div><div class="metric-label">PUT Value</div><div class="metric-value">${bs_model.put_option:.2f}</div></div></div>', unsafe_allow_html=True)

# --- Monte Carlo Simulation Section ---
st.markdown("---")
st.header("Monte Carlo Price Path Simulation")
st.info(f"Simulating **{num_sims:,}** possible price paths for the asset over the next **{time_to_maturity}** years.")

with st.spinner("Running Monte Carlo simulation..."):
    mc_model = MC.MonteCarlo(
        time_to_maturity=time_to_maturity,
        curr_price=current_price,
        volatility=volatility,
        risk_free_rate=interest_rate,
        num_of_sim_paths=num_sims,
        num_of_steps=num_steps,
        expected_return=expected_return
    )
    price_paths = mc_model.run()
    final_prices = price_paths[:, -1]

mc_col1, mc_col2 = st.columns([2, 1])
with mc_col1:
    fig_mc = go.Figure()
    paths_to_plot = num_sims
    time_axis = np.linspace(0, time_to_maturity, num_steps + 1)
    for i in range(paths_to_plot):
        fig_mc.add_trace(go.Scatter(x=time_axis, y=price_paths[i, :], mode='lines', line=dict(width=1.5), opacity=0.5, showlegend=False))
    
    fig_mc.update_layout(title=f"Sample of {paths_to_plot} Simulated Price Paths", xaxis_title="Time (Years)", yaxis_title="Asset Price ($)", height=450)
    st.plotly_chart(fig_mc, use_container_width=True)

with mc_col2:
    avg_final_price = final_prices.mean()
    st.metric("Average Final Price", f"${avg_final_price:.2f}")
    
    fig_hist = go.Figure(data=[go.Histogram(x=final_prices, nbinsx=50)])
    fig_hist.add_vline(x=avg_final_price, line_dash="dash", line_color="red", annotation_text="Avg")
    fig_hist.add_vline(x=strike, line_dash="dash", line_color="green", annotation_text="Strike")
    fig_hist.update_layout(title="Distribution of Final Prices", xaxis_title="Final Price ($)", yaxis_title="Frequency", height=350, margin=dict(t=30, b=10))
    st.plotly_chart(fig_hist, use_container_width=True)


# --- Heatmap Section ---
st.markdown("---")
st.header("Options Price Sensitivity Heatmaps")
st.info("🔍 Explore how option prices fluctuate with varying **Spot Prices** and **Volatility** levels.")

with st.spinner("Generating heatmaps..."):
    heatmap_fig_call, heatmap_fig_put = heatmap(time_to_maturity, strike, interest_rate, spot_range, vol_range)

hm_col1, hm_col2 = st.columns([1, 1], gap="medium")
with hm_col1:
    st.subheader("📈 Call Option Price Heatmap")
    st.pyplot(heatmap_fig_call)
with hm_col2:
    st.subheader("📉 Put Option Price Heatmap")
    st.pyplot(heatmap_fig_put)
