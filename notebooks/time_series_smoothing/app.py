import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.ndimage import gaussian_filter1d
from pykalman import KalmanFilter

# Load data
df_full = pd.read_csv("data/noisy_sine_timeseries.csv", parse_dates=["timestamp"], index_col="timestamp")

st.set_page_config(layout="wide")

# --- Styling ---
st.markdown("""
    <style>
    .method-label {
        display: flex;
        align-items: center;
        gap: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .color-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .stSlider > label {
        font-size: 0.55rem;
        font-weight: 500;
        color: #333;
        margin-bottom: 0.3rem;
    }
    .stSlider .css-1y4p8pa {
        font-size: 0.65rem !important;
    }
    .stSlider .css-1cpxqw2 {
        padding-top: 0.2rem;
        padding-bottom: 0.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Styling ---
st.markdown("""
    <style>
    .method-label {
        display: flex;
        align-items: center;
        gap: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .color-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .stSlider > label {
        font-size: 0.55rem;
        font-weight: 500;
        color: #333;
        margin-bottom: 0.3rem;
    }
    .stSlider .css-1y4p8pa {
        font-size: 0.65rem !important;
    }
    .stSlider .css-1cpxqw2 {
        padding-top: 0.2rem;
        padding-bottom: 0.2rem;
    }
    .left-panel {
        background-color: #f4f6fa;
        padding: 20px;
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)



# --- Colors ---
method_colors = {
    "Original": "black",
    "MA": "#1f77b4",
    "EMA": "#ff7f0e",
    "SavGol": "#2ca02c",
    "LOESS": "#d62728",
    "Gaussian": "#9467bd",
    "Kalman": "#17becf"
}

st.title("Time Series Smoothing Methods: an Interactive Visualizer")

left_col, right_col = st.columns([2, 3])

with left_col:
    st.markdown(
        "This interactive app demonstrates six common time series smoothing techniques: "
        "Moving Average, Exponential Moving Average, Savitzky-Golay, LOESS, Gaussian Filter, and Kalman Filter. "
        "Use the controls to adjust parameters and compare how each method smooths a noisy signal in real time."
    )

    st.subheader("Data Parameters")
    st.caption("Select the dataset and adjust the amount of data visualized, as well as how visible the raw signal is.")
    dp_col1, dp_col2, dp_col3 = st.columns([1, 1, 1])
    with dp_col1:
        dataset_name = st.selectbox("Dataset", options=["Noisy Sine"], index=0)
    with dp_col2:
        subset_size = st.slider("Points to display from start of series", min_value=50, max_value=len(df_full), value=500, step=20)
    with dp_col3:
        signal_opacity = st.slider("Noisy signal opacity", 0.0, 1.0, 0.7, step=0.05)
    df = df_full.iloc[:subset_size].copy()

    st.subheader("Smoothing Parameters")
    st.caption("Select which methods to display and adjust their smoothing parameters below.")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        show_ma = st.checkbox("", value=True, key="show_ma")
        st.markdown(f'<div class="method-label"><span class="color-dot" style="background-color:{method_colors["MA"]}"></span>Moving Avg</div>', unsafe_allow_html=True)
        ma_window = st.slider("Window", 3, 51, 15, step=2, key="ma")

    with col2:
        show_ema = st.checkbox("", value=True, key="show_ema")
        st.markdown(f'<div class="method-label"><span class="color-dot" style="background-color:{method_colors["EMA"]}"></span>EMA</div>', unsafe_allow_html=True)
        ema_alpha = st.slider("Alpha", 0.01, 1.0, 0.1, step=0.01, key="ema")

    with col3:
        show_savgol = st.checkbox("", value=True, key="show_sg")
        st.markdown(f'<div class="method-label"><span class="color-dot" style="background-color:{method_colors["SavGol"]}"></span>SavGol</div>', unsafe_allow_html=True)
        sg_window = st.slider("Window", 5, 51, 15, step=2, key="sg_win")
        sg_poly = st.slider("Poly", 1, 5, 2, key="sg_poly")

    with col4:
        show_loess = st.checkbox("", value=True, key="show_loess")
        st.markdown(f'<div class="method-label"><span class="color-dot" style="background-color:{method_colors["LOESS"]}"></span>LOESS</div>', unsafe_allow_html=True)
        loess_frac = st.slider("Frac", 0.01, 0.5, 0.05, step=0.01, key="loess")

    with col5:
        show_gauss = st.checkbox("", value=True, key="show_gauss")
        st.markdown(f'<div class="method-label"><span class="color-dot" style="background-color:{method_colors["Gaussian"]}"></span>Gaussian</div>', unsafe_allow_html=True)
        gauss_sigma = st.slider("Sigma", 0.1, 10.0, 2.0, step=0.1, key="gauss")

    with col6:
        show_kalman = st.checkbox("", value=True, key="show_kf")
        st.markdown(f'<div class="method-label"><span class="color-dot" style="background-color:{method_colors["Kalman"]}"></span>Kalman</div>', unsafe_allow_html=True)
        kf_transition_noise = st.slider("Transition std", 0.001, 1.0, 0.05, step=0.01, key="kf_trans")
        kf_obs_noise = st.slider("Observation std", 0.001, 1.0, 0.2, step=0.01, key="kf_obs")

# --- Smoothing Calculations ---
df["ma"] = df["noisy_signal"].rolling(window=ma_window, center=True).mean().bfill().ffill()
df["ema"] = df["noisy_signal"].ewm(alpha=ema_alpha).mean()
df["savgol"] = savgol_filter(df["noisy_signal"], window_length=sg_window, polyorder=sg_poly)
lowess_smoothed = lowess(df["noisy_signal"], np.arange(len(df)), frac=loess_frac)
df["loess"] = pd.Series(lowess_smoothed[:, 1], index=df.index)
df["gaussian"] = gaussian_filter1d(df["noisy_signal"], sigma=gauss_sigma)

kf = KalmanFilter(
    transition_matrices=[1],
    observation_matrices=[1],
    initial_state_mean=df["noisy_signal"].iloc[0],
    initial_state_covariance=1,
    transition_covariance=kf_transition_noise,
    observation_covariance=kf_obs_noise
)
kalman_state_means, _ = kf.filter(df["noisy_signal"].values)
df["kalman"] = kalman_state_means

# --- Plotting ---
smooth_opacity = 0.8
sm_width = 1.5
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["noisy_signal"], name="Original", line=dict(color=method_colors["Original"], width=1), opacity=signal_opacity))
if show_ma:
    fig.add_trace(go.Scatter(x=df.index, y=df["ma"], name="MA", line=dict(color=method_colors["MA"], width=sm_width), opacity=smooth_opacity))
if show_ema:
    fig.add_trace(go.Scatter(x=df.index, y=df["ema"], name="EMA", line=dict(color=method_colors["EMA"], width=sm_width), opacity=smooth_opacity))
if show_savgol:
    fig.add_trace(go.Scatter(x=df.index, y=df["savgol"], name="SavGol", line=dict(color=method_colors["SavGol"], width=sm_width), opacity=smooth_opacity))
if show_loess:
    fig.add_trace(go.Scatter(x=df.index, y=df["loess"], name="LOESS", line=dict(color=method_colors["LOESS"], width=sm_width), opacity=smooth_opacity))
if show_gauss:
    fig.add_trace(go.Scatter(x=df.index, y=df["gaussian"], name="Gaussian", line=dict(color=method_colors["Gaussian"], width=sm_width), opacity=smooth_opacity))
if show_kalman:
    fig.add_trace(go.Scatter(x=df.index, y=df["kalman"], name="Kalman", line=dict(color=method_colors["Kalman"], width=sm_width), opacity=smooth_opacity))

fig.update_layout(
    title="Smoothing Methods Comparison",
    xaxis=dict(showgrid=True),
    xaxis_title="Time",
    yaxis_title="Value",
    height=600,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

with right_col:
    st.plotly_chart(fig, use_container_width=True)