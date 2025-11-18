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

st.title("Stock / Custom Data Forecasting with ARIMA (Cloud-Friendly)")

# ------------------ Frequency Selection ------------------
freq_option = st.selectbox(
    "Select Frequency for Aggregation",
    ["Monthly", "Weekly", "Daily"],
    help="Choose how your data should be grouped: Monthly (M), Weekly (W), or Daily (D)."
)
freq_map = {"Monthly": "M", "Weekly": "W", "Daily": "D"}

# ------------------ CSV Upload / Default Data ------------------
uploaded_file = st.file_uploader(
    "Upload your CSV file (must include Date & Value columns)",
    type=["csv"],
    help="Upload a dataset with at least 1 date column and 1 numerical column. Example: sales, prices, revenue."
)

def load_data():
    if uploaded_file is not None:
        try:
            user_data = pd.read_csv(uploaded_file)

            date_cols = user_data.select_dtypes(include=['object', 'datetime']).columns.tolist()
            value_cols = user_data.select_dtypes(include=['int', 'float']).columns.tolist()

            if len(date_cols) == 0 or len(value_cols) == 0:
                st.error("No suitable date or numeric columns found in the uploaded CSV.")
                return pd.Series(dtype=float)

            date_col = st.selectbox("Select Date Column", date_cols, help="Pick the column that contains dates.")
            value_col = st.selectbox("Select Value Column", value_cols, help="Pick the column containing numeric values.")

            user_data[date_col] = pd.to_datetime(user_data[date_col], errors='coerce')
            user_data = user_data.dropna(subset=[date_col, value_col])

            grouped = user_data.groupby(pd.Grouper(key=date_col, freq=freq_map[freq_option]))[value_col].sum()
            grouped = grouped.asfreq(freq_map[freq_option], method='ffill')

            return grouped

        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return pd.Series(dtype=float)
    else:
        # Default: Amazon Stock (if no CSV uploaded)
        import yfinance as yf
        amazon_data = yf.download('AMZN', start='2015-01-01', end='2025-01-01', interval='1mo')

        close_col = 'Adj Close' if 'Adj Close' in amazon_data.columns else 'Close'
        amazon_close = amazon_data[close_col].ffill().dropna()

        return amazon_close.asfreq(freq_map[freq_option], method='ffill').dropna()

data_series = load_data()
if data_series.empty:
    st.stop()

# Clean series (no NaNs)
data_series = data_series.ffill().bfill().dropna()

# ------------------ Seasonal & Noise Sliders ------------------
st.subheader("Adjust Seasonality & Noise")

amplitude = st.slider(
    "Seasonality Amplitude",
    0.0, 2.0, 0.05, 0.01,
    help="Controls the strength of the artificial seasonal pattern. Higher = larger repeating ups & downs."
)

noise_level = st.slider(
    "Noise Std Dev",
    0.0, 0.5, 0.01, 0.01,
    help="Adds random variability to the data, simulating real-world noise."
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
    help="Enable SARIMA for modeling repeating seasonal cycles (e.g., yearly, monthly, weekly)."
)

forecast_periods = st.number_input(
    "Forecast Periods",
    min_value=1, max_value=36, value=12,
    help="How many future time periods to predict. Example: 12 months ahead."
)

log_transform = st.checkbox(
    "Apply Log Transformation",
    value=True,
    help="Stabilizes variance and helps ARIMA learn upward trends more effectively."
)

evaluate_split = st.checkbox(
    "Train/Test Evaluation",
    value=False,
    help="Split the dataset (80/20) to measure forecasting accuracy (MAE/RMSE)."
)

ts_series = adjusted_series.copy().ffill().bfill().dropna()

# Log transform
if log_transform:
    min_val = ts_series.min()
    if min_val <= 0:
        shift = abs(min_val) + 1e-6
        st.info(f"Data contains non-positive values. Auto-shifting by {shift} before applying log.")
        ts_series = ts_series + shift
    ts_series = np.log(ts_series)

# ------------------ Automatic Seasonality Detection ------------------
def detect_seasonality(ts_local, max_lag=36):
    ts_local = ts_local.dropna()
    if len(ts_local) < 3:
        return 1
    ac_vals = acf(ts_local, nlags=min(max_lag, len(ts_local)-1), fft=False)
    ac_vals[0] = 0
    return max(int(np.argmax(ac_vals)), 1)

detected_m = detect_seasonality(ts_series) if seasonal else 1
st.write(f"Detected seasonality period (m): **{detected_m}**")

# ------------------ Train/Test Split ------------------
if evaluate_split and len(ts_series) > 12:
    split_index = int(len(ts_series) * 0.8)
    train, test = ts_series[:split_index], ts_series[split_index:]
else:
    train = ts_series
    test = None

train = train.ffill().bfill().dropna()

# ------------------ Differencing ------------------
train_for_ndiffs = train.ffill().bfill().dropna()
d = ndiffs(train_for_ndiffs) if len(train_for_ndiffs) > 2 else 0
D = nsdiffs(train_for_ndiffs, m=detected_m) if seasonal and detected_m > 1 else 0

# ------------------ Fit SARIMA ------------------
if D > 0:
    sarima_model = SARIMAX(train, order=(1, d, 1), seasonal_order=(1, D, 1, detected_m),
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
else:
    sarima_model = SARIMAX(train, order=(1, d, 1),
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

# ------------------ Forecast ------------------
forecast_result = sarima_model.get_forecast(steps=forecast_periods)

forecast_index = pd.date_range(
    start=ts_series.index[-1] + pd.tseries.frequencies.to_offset(freq_map[freq_option]),
    periods=forecast_periods, freq=freq_map[freq_option].replace("M", "ME")
)

forecast_series = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

if log_transform:
    forecast_series = np.exp(forecast_series) - 1
    conf_int = np.exp(conf_int) - 1

forecast_series.index = forecast_index
conf_int.index = forecast_index

combined_df = pd.concat([data_series.rename("Historical"), forecast_series.rename("Forecast")], axis=1)

# ------------------ Plot with PLOTLY (Interactive Zoom) ------------------
fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(x=data_series.index, y=data_series,
                         mode='lines', name='Historical',
                         line=dict(color='blue', width=2)))

# Forecast
fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series,
                         mode='lines', name='Forecast',
                         line=dict(color='red', width=2, dash='dash')))

# Confidence Interval
fig.add_trace(go.Scatter(
    x=conf_int.index.tolist() + conf_int.index[::-1].tolist(),
    y=conf_int.iloc[:,0].tolist() + conf_int.iloc[:,1][::-1].tolist(),
    fill='toself',
    fillcolor='rgba(255,0,0,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.update_layout(
    title="Forecast with ARIMA",
    xaxis_title="Date",
    yaxis_title="Value",
    hovermode="x unified",
    template="plotly_white"
)

fig.update_xaxes(rangeslider_visible=True)

st.plotly_chart(fig, use_container_width=True)

# ------------------ Downloads ------------------
st.download_button("Download Forecast CSV", combined_df.to_csv().encode(), "forecast.csv")
st.download_button("Download ARIMA Summary TXT", str(sarima_model.summary()), "arima_summary.txt")

# ------------------ Evaluation Metrics ------------------
if evaluate_split and test is not None:
    common_idx = test.index.intersection(forecast_series.index)

    if len(common_idx) > 0:
        test_clean = test.loc[common_idx]
        preds_clean = forecast_series.loc[common_idx]

        if log_transform:
            test_clean = np.exp(test_clean) - 1

        mae = mean_absolute_error(test_clean, preds_clean)
        rmse = np.sqrt(mean_squared_error(test_clean, preds_clean))

        st.success(f"**Model Accuracy:** MAE = {mae:.2f}, RMSE = {rmse:.2f}")
    else:
        st.warning("No overlapping dates between test & forecast for evaluation.")
