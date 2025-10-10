import streamlit as st
import pandas as pd
import numpy as np
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import io
from statsmodels.tsa.stattools import acf
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("Enhanced Sales / Stock Forecasting with ARIMA (Auto-Seasonality + Interactive)")

# Dataset selection
dataset_option = st.selectbox("Select Dataset", ["Rossmann Sales", "Amazon Stock"])

# CSV upload option
uploaded_file = st.file_uploader("Or upload your own CSV file", type=["csv"])

# Frequency selection for aggregation
freq_option = st.selectbox("Select Frequency for Aggregation", ["Monthly", "Weekly", "Daily"])
freq_map = {"Monthly": "M", "Weekly": "W", "Daily": "D"}

# Function to automatically detect seasonality
def detect_seasonality(ts_series, max_lag=36):
    ts_series = ts_series.dropna()
    acf_vals = acf(ts_series, nlags=max_lag, fft=False)
    acf_vals[0] = 0  # ignore lag 0
    seasonal_period = int(np.argmax(acf_vals))
    if seasonal_period < 2:  # fallback if no seasonality detected
        seasonal_period = 1
    return seasonal_period

# Load default datasets
def load_data(dataset):
    if dataset == "Rossmann Sales":
        sales = pd.read_csv(
            "/Users/tanvaipohare/AI_Predictive_Analytics/notebooks/rossmann_data.csv",
            encoding="ISO-8859-1"
        )
        sales['InvoiceDate'] = pd.to_datetime(sales['InvoiceDate'])
        monthly_sales = sales.groupby(pd.Grouper(key='InvoiceDate', freq='ME'))['Quantity'].sum()
        return monthly_sales

    elif dataset == "Amazon Stock":
        import yfinance as yf
        amazon_data = yf.download(
            'AMZN', start='2015-01-01', end='2025-01-01', interval='1mo'
        )
        if amazon_data.empty or 'Close' not in amazon_data.columns:
            return pd.Series(dtype=float)
        close_col = 'Adj Close' if 'Adj Close' in amazon_data.columns else 'Close'
        amazon_close = amazon_data[close_col].ffill().dropna()
        amazon_close.index = pd.to_datetime(amazon_close.index)
        freq = freq_map[freq_option]
        amazon_series = amazon_close.resample(freq).last()
        return amazon_series

# Load or process CSV
data_series = pd.Series(dtype=float)
if uploaded_file is not None:
    try:
        user_data = pd.read_csv(uploaded_file)
        st.info("Tip: Your date column can be in formats like YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY.")
        date_cols = user_data.select_dtypes(include=['object', 'datetime']).columns.tolist()
        value_cols = user_data.select_dtypes(include=['int', 'float']).columns.tolist()

        if len(date_cols) == 0 or len(value_cols) == 0:
            st.error("No suitable date or numeric columns found in CSV.")
        else:
            date_col = st.selectbox("Select Date Column", date_cols)
            value_col = st.selectbox("Select Value Column", value_cols)
            user_data[date_col] = pd.to_datetime(user_data[date_col], errors='coerce', infer_datetime_format=True)
            user_data = user_data.dropna(subset=[date_col, value_col])
            if user_data.empty:
                st.warning("No valid data after parsing dates.")
            else:
                data_series = user_data.groupby(pd.Grouper(key=date_col, freq=freq_map[freq_option]))[value_col].sum()
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")

# If no CSV uploaded, use default dataset
if data_series.empty:
    data_series = load_data(dataset_option)
    st.write(f"Using default dataset: {dataset_option}")

# Ensure Series type
if isinstance(data_series, pd.DataFrame):
    data_series = data_series.squeeze()

# Safe check
if data_series.empty or bool(data_series.isna().all()):
    st.warning("No data available for ARIMA modeling.")
else:
    st.subheader("Optional: Adjust Seasonality & Noise for Uploaded Data")
    amplitude = st.slider("Seasonality Amplitude", 0.0, 0.5, 0.05, 0.01)
    noise_level = st.slider("Noise Standard Deviation", 0.0, 0.2, 0.01, 0.01)
    n = len(data_series)
    seasonal_component = amplitude * np.sin(np.linspace(0, 6 * np.pi, n))
    noise_component = np.random.normal(0, noise_level, n)
    adjusted_series = data_series + seasonal_component + noise_component
    st.line_chart(adjusted_series)

    seasonal = st.checkbox("Use Seasonal ARIMA", value=True)
    forecast_periods = st.number_input("Forecast Periods", min_value=1, max_value=36, value=12)
    log_transform = st.checkbox("Apply Log Transformation for Stabilization", value=True)
    evaluate_split = st.checkbox("Split for Train/Test Evaluation (optional)", value=False)

    ts_series = adjusted_series.copy()
    if log_transform:
        ts_series = np.log(ts_series + 1)

    if st.button("Run ARIMA"):
        try:
            # Automatic seasonality detection
            if seasonal:
                detected_m = detect_seasonality(ts_series)
                st.write(f"Detected seasonality period: m = {detected_m}")
            else:
                detected_m = 1

            # Train/test split
            if evaluate_split and len(ts_series) > 12:
                split_index = int(len(ts_series) * 0.8)
                train, test = ts_series[:split_index], ts_series[split_index:]
            else:
                train = ts_series
                test = None

            # Fit ARIMA
            arima_model = auto_arima(
                train, seasonal=seasonal, m=detected_m,
                start_p=1, start_q=1, max_p=5, max_q=5,
                error_action='ignore', suppress_warnings=True
            )
            st.write("ARIMA Summary:")
            st.text(arima_model.summary())

            # Forecast
            forecast = arima_model.predict(n_periods=forecast_periods)
            if log_transform:
                forecast = np.exp(forecast) - 1
            forecast_index = pd.date_range(
                start=data_series.index[-1] + pd.tseries.frequencies.to_offset(freq_map[freq_option]),
                periods=forecast_periods, freq=freq_map[freq_option].replace("M", "ME")
            )
            forecast_series = pd.Series(forecast, index=forecast_index)

            # Combine historical + forecast
            combined_df = pd.concat([
                data_series.rename("Historical"),
                forecast_series.rename("Forecast")
            ], axis=1)

            # Plot
            plt.figure(figsize=(10,5))
            plt.plot(data_series, label="Historical", linewidth=2)
            plt.plot(forecast_series, label="Forecast", color='red', linestyle="--")
            plt.title(f"{dataset_option if uploaded_file is None else 'Uploaded CSV'} - ARIMA Forecast", fontsize=14)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Value", fontsize=12)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            st.pyplot(plt)

            # Download buttons
            csv = combined_df.to_csv().encode("utf-8")
            st.download_button("Download Forecast CSV", csv,
                               f"{dataset_option.lower().replace(' ', '_')}_forecast.csv", "text/csv")
            summary_text = str(arima_model.summary())
            st.download_button("Download ARIMA Summary TXT", summary_text,
                               f"{dataset_option.lower().replace(' ', '_')}_arima_summary.txt", "text/plain")
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            st.download_button("Download Forecast Chart PNG", buf,
                               f"{dataset_option.lower().replace(' ', '_')}_forecast.png", "image/png")

            # Evaluation metrics
            if evaluate_split and test is not None:
                predictions = arima_model.predict(n_periods=len(test))
                if log_transform:
                    predictions = np.exp(predictions) - 1
                mae = mean_absolute_error(test if not log_transform else np.exp(test)-1, predictions)
                rmse = np.sqrt(mean_squared_error(test if not log_transform else np.exp(test)-1, predictions))
                st.write(f"Train/Test Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        except Exception as e:
            st.error(f"Error during ARIMA modeling: {e}")
