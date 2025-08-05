import streamlit as st 
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
import seaborn as sns
import BlackScholes as BSP

st.set_page_config(
    page_icon="📈",
    page_title="Black Scholes Pricer",
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
    background-color: #90ee90;
    color: black;
    margin-right: 10px;
    border-radius: 10px;
}

.metric-put {
    background-color: #ffcccb;
    color: black;
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
    st.title("📊 Black-Scholes Model")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/shlok-thakkar/"
    github_url = "https://www.github.com/shlok1806"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Shlok Thakkar`</a>', unsafe_allow_html=True)
    st.markdown(f'<a href="{github_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Shlok Thakkar`</a>', unsafe_allow_html=True)

    current_price = st.number_input("Current Asset Price", value=100.0, min_value=0.01)
    strike = st.number_input("Strike Price", value=100.0, min_value=0.01)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0, min_value=0.01)
    volatility = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, max_value=2.0)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05, min_value=0.0, max_value=1.0)

    st.markdown("---")
    st.subheader("Heatmap Parameters")
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
    vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=max(0.01, volatility*0.5), step=0.01)
    vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=min(1.0, volatility*1.5), step=0.01)
    
    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)


def heatmap(time_to_maturity, strike_price, risk_free_rate, spot_range, vol_range):
    """
    Generate heatmap data for call and put options
    """
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            # Create a new BlackScholes instance for each spot/vol combination
            bs_temp = BSP.BlackScholesPricer(
                time_to_maturity=time_to_maturity,
                strike_price=strike_price,
                curr_price=spot,  # Use the varying spot price
                volatility=vol,   # Use the varying volatility
                risk_free_rate=risk_free_rate
            )
            bs_temp.run()  # Run the calculation
            call_prices[i, j] = bs_temp.call_option
            put_prices[i, j] = bs_temp.put_option
    
    # Create Call Price Heatmap
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_prices, 
                xticklabels=np.round(spot_range, 2), 
                yticklabels=np.round(vol_range, 2), 
                annot=True, 
                fmt=".2f", 
                cmap="RdYlGn", 
                ax=ax_call,
                cbar_kws={'label': 'Option Price'})
    ax_call.set_title('Call Option Prices', fontsize=16, fontweight='bold')
    ax_call.set_xlabel('Spot Price', fontsize=12)
    ax_call.set_ylabel('Volatility', fontsize=12)
    
    # Create Put Price Heatmap
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_prices, 
                xticklabels=np.round(spot_range, 2), 
                yticklabels=np.round(vol_range, 2), 
                annot=True, 
                fmt=".2f", 
                cmap="RdYlGn_r", 
                ax=ax_put,
                cbar_kws={'label': 'Option Price'})
    ax_put.set_title('Put Option Prices', fontsize=16, fontweight='bold')
    ax_put.set_xlabel('Spot Price', fontsize=12)
    ax_put.set_ylabel('Volatility', fontsize=12)
    
    return fig_call, fig_put


# Main content
st.title("Black-Scholes Pricing Model")

# Display input parameters
input_data = {
    "Current Asset Price": [f"${current_price:.2f}"],
    "Strike Price": [f"${strike:.2f}"],
    "Time to Maturity": [f"{time_to_maturity:.2f} years"],
    "Volatility (σ)": [f"{volatility:.2%}"],
    "Risk-Free Rate": [f"{interest_rate:.2%}"],
}
input_df = pd.DataFrame(input_data)
st.table(input_df)

# Calculate Call and Put values
bs_model = BSP.BlackScholesPricer(
    time_to_maturity=time_to_maturity,
    strike_price=strike,
    curr_price=current_price,
    volatility=volatility,
    risk_free_rate=interest_rate
)
bs_model.run()

# Display Call and Put Values
col1, col2 = st.columns([1, 1], gap="small")

with col1:
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${bs_model.call_option:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${bs_model.put_option:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Display Greeks if available
if hasattr(bs_model, 'call_delta'):
    st.markdown("### Greeks")
    greeks_col1, greeks_col2, greeks_col3 = st.columns(4)
    
    with greeks_col1:
        st.metric("Call Delta", f"{bs_model.call_delta:.4f}")
        st.metric("Put Delta", f"{bs_model.put_delta:.4f}")
    
    with greeks_col2:
        st.metric("Gamma (Call)", f"{bs_model.call_gamma:.4f}")
        st.metric("Gamma (Put)", f"{bs_model.put_gamma:.4f}")
    
    with greeks_col3:
        # Add moneyness indicator
        moneyness = current_price / strike
        if moneyness > 1.05:
            status = "In-the-Money (ITM)"
        elif moneyness < 0.95:
            status = "Out-of-the-Money (OTM)"
        else:
            status = "At-the-Money (ATM)"
        st.metric("Moneyness", f"{moneyness:.2f}")
        st.caption(status)

st.markdown("---")
st.title("Options Price - Interactive Heatmap")
st.info("🔍 Explore how option prices fluctuate with varying **Spot Prices** and **Volatility** levels while keeping the **Strike Price** constant.")

# Generate heatmaps
with st.spinner("Generating heatmaps..."):
    heatmap_fig_call, heatmap_fig_put = heatmap(
        time_to_maturity=time_to_maturity,
        strike_price=strike,
        risk_free_rate=interest_rate,
        spot_range=spot_range,
        vol_range=vol_range
    )

# Display heatmaps
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("📈 Call Option Price Heatmap")
    st.pyplot(heatmap_fig_call)
    st.caption("Higher prices shown in green, lower in red")

with col2:
    st.subheader("📉 Put Option Price Heatmap")
    st.pyplot(heatmap_fig_put)
    st.caption("Higher prices shown in red, lower in green")

# Add additional insights
st.markdown("---")
st.subheader("💡 Key Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"""
    **Current Market Status**
    - Spot/Strike Ratio: {current_price/strike:.2f}
    - Call is worth {(bs_model.call_option/current_price)*100:.1f}% of spot
    - Put is worth {(bs_model.put_option/current_price)*100:.1f}% of spot
    """)

with col2:
    st.warning(f"""
    **Volatility Impact**
    - Current volatility: {volatility:.1%}
    - Heatmap vol range: {vol_min:.1%} - {vol_max:.1%}
    - Higher volatility → Higher option prices
    """)

with col3:
    st.success(f"""
    **Time Value**
    - Time to expiry: {time_to_maturity:.2f} years
    - Call intrinsic value: ${max(0, current_price - strike):.2f}
    - Put intrinsic value: ${max(0, strike - current_price):.2f}
    """)