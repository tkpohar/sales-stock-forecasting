import streamlit as st
import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import io

st.title("Sales / Stock Forecasting with ARIMA")

# Dataset selection
dataset_option = st.selectbox("Select Dataset", ["Rossmann Sales", "Amazon Stock"])

# CSV upload option
uploaded_file = st.file_uploader("Or upload your own CSV file", type=["csv"])

# Frequency selection for aggregation
freq_option = st.selectbox("Select Frequency for Aggregation", ["Monthly", "Weekly", "Daily"])
freq_map = {"Monthly": "M", "Weekly": "W", "Daily": "D"}

# Load default datasets safely
def load_data(dataset):
    if dataset == "Rossmann Sales":
        sales = pd.read_csv(
            "/Users/tanvaipohare/AI_Predictive_Analytics/notebooks/rossmann_data.csv",
            encoding="ISO-8859-1"
        )
        sales['InvoiceDate'] = pd.to_datetime(sales['InvoiceDate'])
        monthly_sales = sales.groupby(pd.Grouper(key='InvoiceDate', freq='M'))['Quantity'].sum()
        return monthly_sales

    elif dataset == "Amazon Stock":
        import yfinance as yf
        amazon_data = yf.download(
            'AMZN', start='2015-01-01', end='2025-01-01', interval='1mo'
        )

        if amazon_data.empty or 'Close' not in amazon_data.columns:
            return pd.Series(dtype=float)

        # Use Adjusted Close if available, else Close
        close_col = 'Adj Close' if 'Adj Close' in amazon_data.columns else 'Close'
        amazon_close = amazon_data[close_col].ffill().dropna()

        # Convert index to datetime just to be safe
        amazon_close.index = pd.to_datetime(amazon_close.index)

        # Aggregate by selected frequency
        freq = freq_map[freq_option]  # Monthly/Weekly/Daily
        amazon_series = amazon_close.resample(freq).last()  # Use last price in period
        return amazon_series


# Process uploaded CSV if provided
data_series = pd.Series(dtype=float)
if uploaded_file is not None:
    try:
        user_data = pd.read_csv(uploaded_file)
        st.info("Tip: Your date column can be in formats like YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY.")

        # Let user pick date and value columns
        date_cols = user_data.select_dtypes(include=['object', 'datetime']).columns.tolist()
        value_cols = user_data.select_dtypes(include=['int', 'float']).columns.tolist()

        if len(date_cols) == 0 or len(value_cols) == 0:
            st.error("No suitable date or numeric columns found in CSV.")
        else:
            date_col = st.selectbox("Select Date Column", date_cols)
            value_col = st.selectbox("Select Value Column", value_cols)

            # Parse date with multiple formats
            user_data[date_col] = pd.to_datetime(user_data[date_col], errors='coerce', infer_datetime_format=True)
            user_data = user_data.dropna(subset=[date_col, value_col])

            if user_data.empty:
                st.warning("No valid data after parsing dates.")
            else:
                # Aggregate based on frequency
                data_series = user_data.groupby(pd.Grouper(key=date_col, freq=freq_map[freq_option]))[value_col].sum()
                st.line_chart(data_series)

    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")

# If no CSV uploaded, use default dataset
if data_series.empty:
    data_series = load_data(dataset_option)
    st.write(f"Using default dataset: {dataset_option}")
    st.line_chart(data_series)

# Ensure Series type
if isinstance(data_series, pd.DataFrame):
    data_series = data_series.squeeze()

# Safe check
if data_series.empty or bool(data_series.isna().all()):
    st.warning("No data available for ARIMA modeling.")
else:
    seasonal = st.checkbox("Use Seasonal ARIMA", value=False)
    forecast_periods = st.number_input("Forecast Periods", min_value=1, max_value=36, value=12)

    if st.button("Run ARIMA"):
        try:
            arima_model = auto_arima(
                data_series, seasonal=seasonal, m=12 if seasonal else 1,
                error_action='ignore', suppress_warnings=True
            )

            st.write("ARIMA Summary:")
            st.text(arima_model.summary())

            # Forecast
            forecast = arima_model.predict(n_periods=forecast_periods)
            forecast_index = pd.date_range(
                start=data_series.index[-1] + pd.offsets.MonthEnd(),
                periods=forecast_periods, freq='M'
            )
            forecast_series = pd.Series(forecast, index=forecast_index)

            # Combine historical + forecast
            combined_df = pd.concat([
                data_series.rename("Historical"),
                forecast_series.rename("Forecast")
            ], axis=1)

            # Plot
            plt.figure(figsize=(10, 5))
            plt.plot(data_series, label="Historical", linewidth=2)
            plt.plot(forecast_series, label="Forecast", color='red', linestyle="--")
            plt.title(f"{dataset_option if uploaded_file is None else 'Uploaded CSV'} - ARIMA Forecast", fontsize=14)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Value", fontsize=12)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(plt)

            # Download CSV
            csv = combined_df.to_csv().encode("utf-8")
            st.download_button(
                label="Download Forecast Data as CSV",
                data=csv,
                file_name=f"{dataset_option.lower().replace(' ', '_')}_forecast.csv",
                mime="text/csv",
            )

            # Download ARIMA summary
            summary_text = str(arima_model.summary())
            st.download_button(
                label="Download ARIMA Summary as TXT",
                data=summary_text,
                file_name=f"{dataset_option.lower().replace(' ', '_')}_arima_summary.txt",
                mime="text/plain",
            )

            # Download chart
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            st.download_button(
                label="Download Forecast Chart as PNG",
                data=buf,
                file_name=f"{dataset_option.lower().replace(' ', '_')}_forecast.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"Error during ARIMA modeling: {e}")
