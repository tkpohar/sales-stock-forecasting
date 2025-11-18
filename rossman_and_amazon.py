import streamlit as st
import pandas as pd
import numpy as np
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import io
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima.utils import ndiffs, nsdiffs
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

st.title("📈 Stock / Custom Data Forecasting with ARIMA (Cloud-Friendly)")

# ------------------ Frequency Selection ------------------
freq_option = st.selectbox(
    "Select Frequency for Aggregation",
    ["Monthly", "Weekly", "Daily"],
    help="Choose how your data should be aggregated. ARIMA performs differently on daily, weekly, or monthly data."
)
freq_map = {"Monthly": "M", "Weekly": "W", "Daily": "D"}

# ------------------ CSV Upload / Default Data ------------------
uploaded_file = st.file_uploader(
    "Upload your CSV file (Date + Value columns)",
    type=["csv"],
    help="Upload a CSV containing at least one date column and one numeric column."
)

def load_data():
    if uploaded_file is not None:
        try:
            user_data = pd.read_csv(uploaded_file)

            # Auto-detect columns
            date_cols = user_data.select_dtypes(include=['object', 'datetime']).columns.tolist()
            value_cols = user_data.select_dtypes(include=['int', 'float']).columns.tolist()

            if len(date_cols) == 0 or len(value_cols) == 0:
                st.error("No suitable date or numeric columns found.")
                return pd.Series(dtype=float)

            date_col = st.selectbox("Select Date Column", date_cols)
            value_col = st.selectbox("Select Value Column", value_cols)

            # Convert date column
            user_data[date_col] = pd.to_datetime(user_data[date_col], errors='coerce')
            user_data = user_data.dropna(subset=[date_col, value_col])

            grouped = user_data.groupby(
                pd.Grouper(key=date_col, freq=freq_map[freq_option])
            )[value_col].sum()

            grouped = grouped.asfreq(freq_map[freq_option], method='ffill')
            return grouped

        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return pd.Series(dtype=float)

    else:
        # Default: Amazon Stock
        import yfinance as yf
        amazon_data = yf.download('AMZN', start='2015-01-01', end='2025-01-01', interval='1mo')

        if amazon_data.empty:
            st.warning("Could not fetch Amazon stock data.")
            return pd.Series(dtype=float)

        close_col = 'Adj Close' if 'Adj Close' in amazon_data.columns else 'Close'
        amazon_close = amazon_data[close_col].ffill()

        return amazon_close.asfreq(freq_map[freq_option], method='ffill')

# ------------------ Load Data ------------------
data_series = load_data()
if data_series.empty:
    st.stop()

# ------------------ Force Series Format (CRITICAL FIX) ------------------
if isinstance(data_series, pd.DataFrame):
    if data_series.shape[1] == 1:
        data_series = data_series.iloc[:, 0]
    else:
        st.error("Select only ONE numeric value column.")
        st.stop()

data_series = pd.to_numeric(data_series, errors='coerce').dropna()
data_series = data_series.ffill().bfill()

# ------------------ Seasonal & Noise Sliders ------------------
st.subheader("Synthetic Adjustments (Optional)")

amplitude = st.slider(
    "Seasonality Amplitude",
    0.0, 2.0, 0.05, 0.01,
    help="Controls the strength of added seasonality. Higher values create more pronounced repeating patterns."
)

noise_level = st.slider(
    "Noise Std Dev",
    0.0, 0.5, 0.01, 0.01,
    help="Adds random noise to simulate real-world fluctuations."
)

n = len(data_series)
seasonal_component = amplitude * np.sin(np.linspace(0, 6 * np.pi, n))
noise_component = np.random.normal(0, noise_level, n)

adjusted_series = data_series + seasonal_component + noise_component
st.line_chart(adjusted_series)

# ------------------ ARIMA Options ------------------
seasonal = st.checkbox(
    "Use Seasonal ARIMA",
    value=True,
    help="Enable SARIMA if your data has repeating seasonal patterns."
)

forecast_periods = st.number_input(
    "Forecast Periods",
    min_value=1, max_value=36, value=12,
    help="How many future points to forecast."
)

log_transform = st.checkbox(
    "Apply Log Transformation",
    value=True,
    help="Stabilizes variance for trending datasets."
)

evaluate_split = st.checkbox(
    "Train/Test Evaluation",
    value=False,
    help="Splits 80% training, 20% testing to measure accuracy."
)

# Clean series for modeling
ts_series = adjusted_series.copy().dropna()

if log_transform:
    min_val = ts_series.min()
    if min_val <= 0:
        shift = abs(min_val) + 1e-6
        st.warning(f"Series shifted by {shift:.6f} for log transform.")
        ts_series = ts_series + shift
    ts_series = np.log(ts_series)

# ------------------ Seasonality Detection ------------------
def detect_seasonality(ts, max_lag=36):
    ts = ts.dropna()
    if len(ts) < 3:
        return 1
    acv = acf(ts, nlags=min(max_lag, len(ts)-1))
    acv[0] = 0
    return max(int(np.argmax(acv)), 1)

detected_m = detect_seasonality(ts_series) if seasonal else 1
st.write(f"Detected seasonality (m): {detected_m}")

# ------------------ Train/Test Split ------------------
if evaluate_split and len(ts_series) > 12:
    split = int(len(ts_series) * 0.8)
    train, test = ts_series[:split], ts_series[split:]
else:
    train, test = ts_series, None

# Minimum Points Check
if len(train) < 12:
    st.warning("Dataset too small for SARIMA. Using naive forecast.")
    last_val = train.iloc[-1]

    fc_idx = pd.date_range(
        start=train.index[-1] + pd.tseries.frequencies.to_offset(freq_map[freq_option]),
        periods=forecast_periods,
        freq=freq_map[freq_option]
    )

    forecast_series = pd.Series([last_val] * forecast_periods, index=fc_idx)
else:
    # Differencing
    train_clean = train.dropna()
    d = ndiffs(train_clean) if len(train_clean) > 2 else 0
    D = nsdiffs(train_clean, m=detected_m) if seasonal and detected_m > 1 else 0

    # Fit SARIMA
    sarima_model = SARIMAX(
        train,
        order=(1, d, 1),
        seasonal_order=(1, D, 1, detected_m) if D > 0 else (0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    # Forecast
    fc = sarima_model.get_forecast(steps=forecast_periods)
    forecast_series = fc.predicted_mean
    conf_int = fc.conf_int()

    # Reverse log
    if log_transform:
        forecast_series = np.exp(forecast_series) - 1
        conf_int = np.exp(conf_int) - 1

    # Set index
    fc_idx = pd.date_range(
        start=ts_series.index[-1] + pd.tseries.frequencies.to_offset(freq_map[freq_option]),
        periods=forecast_periods,
        freq=freq_map[freq_option]
    )

    forecast_series.index = fc_idx
    conf_int.index = fc_idx

# ------------------ Combine Historical + Forecast ------------------
combined_df = pd.concat(
    [data_series.rename("Historical"), forecast_series.rename("Forecast")],
    axis=1
)

# ------------------ Plotly Interactive Chart ------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data_series.index, y=data_series,
    mode="lines", name="Historical", line=dict(color="blue")
))

fig.add_trace(go.Scatter(
    x=forecast_series.index, y=forecast_series,
    mode="lines", name="Forecast", line=dict(color="red", dash="dash")
))

if "conf_int" in locals():
    fig.add_trace(go.Scatter(
        x=conf_int.index.tolist() + conf_int.index[::-1].tolist(),
        y=conf_int.iloc[:,0].tolist() + conf_int.iloc[:,1][::-1].tolist(),
        fill="toself", fillcolor="rgba(255,0,0,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip", showlegend=False
    ))

fig.update_layout(
    title="📊 Forecast with Interactive Zoom",
    xaxis_title="Date",
    yaxis_title="Value",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------ Download Buttons ------------------
st.download_button("Download Forecast CSV", combined_df.to_csv().encode("utf-8"), "forecast.csv")

if "sarima_model" in locals():
    st.download_button("Download Model Summary", str(sarima_model.summary()), "arima_summary.txt")

# ------------------ Evaluation ------------------
if evaluate_split and test is not None:
    overlap = test.index.intersection(forecast_series.index)
    if len(overlap) > 0:
        mae = mean_absolute_error(test.loc[overlap], forecast_series.loc[overlap])
        rmse = np.sqrt(mean_squared_error(test.loc[overlap], forecast_series.loc[overlap]))
        st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    else:
        st.warning("No overlapping points between test and forecast.")
