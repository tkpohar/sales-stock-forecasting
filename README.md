Sales & Stock Forecasting with ARIMA
This Streamlit app allows forecasting of Rossmann Sales, Amazon Stock Prices, or any custom business dataset uploaded as a CSV.
It uses ARIMA modeling (pmdarima) to predict future trends.
How to Run the App
Option 1: Run Online (Streamlit Cloud)
Simply click this link to open the app in your browser:
[App Link Here](https://share.streamlit.io/your-username/sales-stock-forecasting/main/rossman_and_amazon.py)
_No installation required._
Option 2: Run Locally
1. Clone the repository:
git clone https://github.com/tkpohar/sales-stock-forecasting.git
cd sales-stock-forecasting
2. Install dependencies:
pip install -r requirements.txt
3. Run the Streamlit app:
streamlit run rossman_and_amazon.py
Uploading Your Own CSV
- You can upload any CSV with:
  - A date column (formats supported: YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY)
  - A numeric column (sales, revenue, stock price, etc.)

- After upload, choose:
  - Date column
  - Value column
  - Aggregation frequency: Daily / Weekly / Monthly
Features
- Default datasets:
  - ✅ Rossmann Sales
  - ✅ Amazon Stock (live data via yfinance)
- Upload your own dataset (CSV)
- Frequency aggregation (Daily, Weekly, Monthly)
- ARIMA modeling (Seasonal / Non-Seasonal)
- Download:
  - Forecasted data (CSV)
  - ARIMA summary (TXT)
  - Forecast chart (PNG)
