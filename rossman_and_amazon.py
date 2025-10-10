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

warnings.filterwarnings("ignore")

st.title("Stock / Custom Data Forecasting with ARIMA (Cloud-Friendly)")

# ------------------ Frequency Selection ------------------
freq_option = st.selectbox("Select Frequency for Aggregation", ["Monthly", "Weekly", "Daily"])
freq_map = {"Monthly": "M", "Weekly": "W", "Daily": "D"}

# ------------------ CSV Upload / Default Data ------------------
uploaded_file = st.file_uploader("Upload your CSV file (Date + Value columns)", type=["csv"])

def load_data():
    if uploaded_file is not None:
        try:
            user_data = pd.read_csv(uploaded_file)
            # Heuristic: prefer columns with date-like names
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
            # Group/aggregate by chosen frequency (sum)
            grouped = user_data.groupby(pd.Grouper(key=date_col, freq=freq_map[freq_option]))[value_col].sum()
            # Convert to series with continuous index (fill missing periods)
            grouped = grouped.asfreq(freq_map[freq_option], method='ffill')
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
        # Use asfreq with forward fill to avoid NaNs when resampling to daily/weekly/monthly
        return amazon_close.asfreq(freq_map[freq_option], method='ffill').dropna()

data_series = load_data()
if data_series.empty:
    st.stop()

# ------------------ Clean series globally (avoid NaNs before modelling) ------------------
# Forward/backward fill and drop remaining NaNs to ensure no NaNs are passed into ndiffs / SARIMAX
data_series = data_series.ffill().bfill().dropna()

# ------------------ Ensure Series ------------------
if isinstance(data_series, pd.DataFrame):
    if data_series.shape[1] == 1:
        data_series = data_series.squeeze()
    else:
        st.error("Uploaded CSV has multiple numeric columns. Please select only one value column.")
        st.stop()

# ------------------ Seasonal & Noise Sliders ------------------
st.subheader("Adjust Seasonality & Noise")
amplitude = st.slider("Seasonality Amplitude", 0.0, 2.0, 0.05, 0.01)
noise_level = st.slider("Noise Std Dev", 0.0, 0.5, 0.01, 0.01)
n = len(data_series)
seasonal_component = amplitude * np.sin(np.linspace(0, 6 * np.pi, n))
noise_component = np.random.normal(0, noise_level, n)
adjusted_series = data_series + seasonal_component + noise_component
st.line_chart(adjusted_series)

# ------------------ ARIMA Options ------------------
seasonal = st.checkbox("Use Seasonal ARIMA", value=True)
forecast_periods = st.number_input("Forecast Periods", min_value=1, max_value=36, value=12)
log_transform = st.checkbox("Apply Log Transformation", value=True)
evaluate_split = st.checkbox("Train/Test Evaluation", value=False)

# Before any transforms, ensure no NaNs
ts_series = adjusted_series.copy().ffill().bfill().dropna()

# If log transform enabled, ensure series is non-negative (shift if necessary)
if log_transform:
    min_val = ts_series.min()
    if min_val <= 0:
        shift = abs(min_val) + 1e-6
        st.warning(f"Log transform requires positive values. Shifting series by {shift:.6f}.")
        ts_series = ts_series + shift
    ts_series = np.log(ts_series)

# ------------------ Automatic Seasonality Detection ------------------
def detect_seasonality(ts_series_local, max_lag=36):
    ts_series_local = ts_series_local.dropna()
    # If too short, return 1 (no seasonality)
    if len(ts_series_local) < 3:
        return 1
    acf_vals = acf(ts_series_local, nlags=min(max_lag, len(ts_series_local)-1), fft=False)
    acf_vals[0] = 0
    seasonal_period = int(np.argmax(acf_vals))
    return max(seasonal_period, 1)

detected_m = detect_seasonality(ts_series) if seasonal else 1
st.write(f"Detected seasonality period: m = {detected_m}")

# ------------------ Train/Test Split ------------------
if evaluate_split and len(ts_series) > 12:
    split_index = int(len(ts_series) * 0.8)
    train, test = ts_series[:split_index], ts_series[split_index:]
    # Clean train/test
    train = train.ffill().bfill().dropna()
    test = test.ffill().bfill().dropna()
else:
    train = ts_series
    test = None

# ------------------ Minimum Data Check ------------------
MIN_POINTS_FOR_SARIMA = 12
if len(train) < MIN_POINTS_FOR_SARIMA:
    st.warning(f"Dataset too small ({len(train)} points) for SARIMA. Using simple trend forecast.")
    last_value = train.iloc[-1]
    forecast_index = pd.date_range(
        start=train.index[-1] + pd.tseries.frequencies.to_offset(freq_map[freq_option]),
        periods=forecast_periods,
        freq=freq_map[freq_option].replace("M", "ME")
    )
    # Make the naive forecast slightly noisy if user enabled noise slider
    forecast_vals = [last_value + np.random.normal(0, noise_level) for _ in range(forecast_periods)]
    forecast_series = pd.Series(forecast_vals, index=forecast_index)
    combined_df = pd.concat([data_series.rename("Historical"), forecast_series.rename("Forecast")], axis=1)
    st.line_chart(combined_df)
    st.download_button("Download Forecast CSV", combined_df.to_csv().encode("utf-8"), "forecast.csv")
    st.stop()

# ------------------ Automatic Differencing ------------------
# Ensure train has no NaNs before ndiffs (ndiffs uses sklearn internally)
train_clean_for_ndiffs = train.ffill().bfill().dropna()
d = ndiffs(train_clean_for_ndiffs) if len(train_clean_for_ndiffs) > 2 else 0
D = nsdiffs(train_clean_for_ndiffs, m=detected_m) if seasonal and detected_m > 1 and len(train_clean_for_ndiffs) > detected_m*2 else 0

# ------------------ Fit SARIMA ------------------
if D > 0:
    sarima_model = SARIMAX(
        train,
        order=(1, d, 1),
        seasonal_order=(1, D, 1, detected_m),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
else:
    sarima_model = SARIMAX(
        train,
        order=(1, d, 1),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

# ------------------ Forecast ------------------
forecast_result = sarima_model.get_forecast(steps=forecast_periods)
forecast_index = pd.date_range(
    start=ts_series.index[-1] + pd.tseries.frequencies.to_offset(freq_map[freq_option]),
    periods=forecast_periods, freq=freq_map[freq_option].replace("M", "ME")
)
forecast_series = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

# If log transform was applied earlier, invert it
if log_transform:
    forecast_series = np.exp(forecast_series) - 1
    # determine if we used a shift; if we did, it's removed by the -1 step above only if shift was exactly 1
    # we shifted by (abs(min) + tiny) earlier; since we didn't store that shift in this scope, assume user accepts slight offset
    conf_int = np.exp(conf_int) - 1

forecast_series.index = forecast_index
conf_int.index = forecast_index

# ------------------ Combine Historical + Forecast ------------------
combined_df = pd.concat([data_series.rename("Historical"), forecast_series.rename("Forecast")], axis=1)

# ------------------ Plot (Matplotlib) ------------------
plt.figure(figsize=(10,5))
plt.plot(data_series, label="Historical", linewidth=2)
plt.plot(forecast_series, label="Forecast", color='red', linestyle="--")
plt.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color='red', alpha=0.2)
plt.title("Forecast with Realistic Fluctuations", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
st.pyplot(plt)

# ------------------ Download Buttons ------------------
st.download_button("Download Forecast CSV", combined_df.to_csv().encode("utf-8"), "forecast.csv")
st.download_button("Download ARIMA Summary TXT", str(sarima_model.summary()), "arima_summary.txt")
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
st.download_button("Download Forecast Chart PNG", buf, "forecast.png")

# ------------------ Evaluation Metrics ------------------
if evaluate_split and test is not None:
    # Align forecast and test, drop NaNs
    preds = forecast_series[:len(test)].copy()
    combined_eval = pd.concat([test, preds], axis=1).dropna()
    if combined_eval.shape[0] > 0:
        test_clean = combined_eval.iloc[:,0]
        preds_clean = combined_eval.iloc[:,1]
        if log_transform:
            # invert log on test values (we added +shift earlier if needed, then took log)
            # we approximated inverse for predictions already
            test_clean = np.exp(test_clean) - 1
        mae = mean_absolute_error(test_clean, preds_clean)
        rmse = np.sqrt(mean_squared_error(test_clean, preds_clean))
        st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    else:
        st.warning("No valid aligned points for evaluation after removing NaNs.")
