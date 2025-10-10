import streamlit as st
import pandas as pd
import numpy as np
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import io
from statsmodels.tsa.stattools import acf
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("Stock / Custom Data Forecasting with ARIMA (Cloud-Friendly)")

# Frequency selection
freq_option = st.selectbox("Select Frequency for Aggregation", ["Monthly", "Weekly", "Daily"])
freq_map = {"Monthly": "M", "Weekly": "W", "Daily": "D"}

# CSV upload
uploaded_file = st.file_uploader("Upload your CSV file (Date + Value columns)", type=["csv"])

# Load data
def load_data():
    if uploaded_file is not None:
        try:
            user_data = pd.read_csv(uploaded_file)
            date_cols = user_data.select_dtypes(include=['object', 'datetime']).columns.tolist()
            value_cols = user_data.select_dtypes(include=['int', 'float']).columns.tolist()
            if len(date_cols) == 0 or len(value_cols) == 0:
                st.error("No suitable date or numeric columns found in CSV.")
                return pd.Series(dtype=float)
            date_col = st.selectbox("Select Date Column", date_cols)
            value_col = st.selectbox("Select Value Column", value_cols)
            user_data[date_col] = pd.to_datetime(user_data[date_col], errors='coerce', infer_datetime_format=True)
            user_data = user_data.dropna(subset=[date_col, value_col])
            if user_data.empty:
                st.warning("No valid data after parsing dates.")
                return pd.Series(dtype=float)
            grouped = user_data.groupby(pd.Grouper(key=date_col, freq=freq_map[freq_option]))[value_col].sum()
            return grouped
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return pd.Series(dtype=float)
    else:
        # Default: Amazon Stock
        import yfinance as yf
        amazon_data = yf.download('AMZN', start='2015-01-01', end='2025-01-01', interval='1mo')
        if amazon_data.empty or 'Close' not in amazon_data.columns:
            st.warning("Amazon stock data could not be fetched.")
            return pd.Series(dtype=float)
        close_col = 'Adj Close' if 'Adj Close' in amazon_data.columns else 'Close'
        amazon_close = amazon_data[close_col].ffill().dropna()
        amazon_close.index = pd.to_datetime(amazon_close.index)
        return amazon_close.resample(freq_map[freq_option]).last()

data_series = load_data()
if data_series.empty:
    st.stop()

# Ensure Series
if isinstance(data_series, pd.DataFrame):
    if data_series.shape[1] == 1:
        data_series = data_series.squeeze()
    else:
        st.error("Uploaded CSV has multiple numeric columns. Please select only one value column.")
        st.stop()

# Interactive seasonality and noise sliders
st.subheader("Adjust Seasonality & Noise")
amplitude = st.slider("Seasonality Amplitude", 0.0, 0.5, 0.05, 0.01)
noise_level = st.slider("Noise Std Dev", 0.0, 0.2, 0.01, 0.01)
n = len(data_series)
seasonal_component = amplitude * np.sin(np.linspace(0, 6 * np.pi, n))
noise_component = np.random.normal(0, noise_level, n)
adjusted_series = data_series + seasonal_component + noise_component
st.line_chart(adjusted_series)

# Detect seasonality automatically
def detect_seasonality(ts_series, max_lag=36):
    ts_series = ts_series.dropna()
    acf_vals = acf(ts_series, nlags=max_lag, fft=False)
    acf_vals[0] = 0
    seasonal_period = int(np.argmax(acf_vals))
    return max(seasonal_period, 1)

# ARIMA options
seasonal = st.checkbox("Use Seasonal ARIMA", value=True)
forecast_periods = st.number_input("Forecast Periods", min_value=1, max_value=36, value=12)
log_transform = st.checkbox("Apply Log Transformation", value=True)
evaluate_split = st.checkbox("Train/Test Evaluation", value=False)

ts_series = adjusted_series.copy()
if log_transform:
    ts_series = np.log(ts_series + 1)

# Real-time ARIMA forecast
try:
    detected_m = detect_seasonality(ts_series) if seasonal else 1
    st.write(f"Detected seasonality period: m = {detected_m}")

    if evaluate_split and len(ts_series) > 12:
        split_index = int(len(ts_series) * 0.8)
        train, test = ts_series[:split_index], ts_series[split_index:]
    else:
        train = ts_series
        test = None

    arima_model = auto_arima(
        train, seasonal=seasonal, m=detected_m,
        start_p=1, start_q=1, max_p=5, max_q=5,
        error_action='ignore', suppress_warnings=True
    )

    forecast = arima_model.predict(n_periods=forecast_periods)
    if log_transform:
        forecast = np.exp(forecast) - 1
    forecast_index = pd.date_range(
        start=data_series.index[-1] + pd.tseries.frequencies.to_offset(freq_map[freq_option]),
        periods=forecast_periods, freq=freq_map[freq_option].replace("M", "ME")
    )
    forecast_series = pd.Series(forecast, index=forecast_index)
    combined_df = pd.concat([data_series.rename("Historical"), forecast_series.rename("Forecast")], axis=1)

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(data_series, label="Historical", linewidth=2)
    plt.plot(forecast_series, label="Forecast", color='red', linestyle="--")
    plt.title("ARIMA Forecast", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(plt)

    # Downloads
    st.download_button("Download Forecast CSV", combined_df.to_csv().encode("utf-8"), "forecast.csv")
    st.download_button("Download ARIMA Summary TXT", str(arima_model.summary()), "arima_summary.txt")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    st.download_button("Download Forecast Chart PNG", buf, "forecast.png")

    # Evaluation
    if evaluate_split and test is not None:
        predictions = arima_model.predict(n_periods=len(test))
        if log_transform:
            predictions = np.exp(predictions) - 1
        mae = mean_absolute_error(test if not log_transform else np.exp(test)-1, predictions)
        rmse = np.sqrt(mean_squared_error(test if not log_transform else np.exp(test)-1, predictions))
        st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

except Exception as e:
    st.error(f"Error during ARIMA modeling: {e}")
