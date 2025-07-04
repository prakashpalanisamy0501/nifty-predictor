import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
import pytz

# Title
st.title("📈 NIFTY Index Movement Predictor (AI-based, 2-Min Interval)")
st.markdown("This model predicts whether NIFTY will go **UP or DOWN** in upcoming 2-minute intervals over the next **1 hour** using technical indicators.")

# Manual Refresh Button
if st.button("🔄 Refresh Data Now"):
    st.cache_data.clear()

# Load 2-minute data
@st.cache_data(ttl=120)
def load_data():
    try:
        data = yf.download("^NSEI", interval="2m", period="7d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data = data.dropna()
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data = data.dropna(subset=['Close'])

        # Convert index to IST
        if data.index.tzinfo is None or data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        data.index = data.index.tz_convert('Asia/Kolkata')

        return data
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return pd.DataFrame()

# Load Data
nifty = load_data()

if not nifty.empty:
    st.subheader("📊 Latest NIFTY Data (2-min)")
    st.dataframe(nifty.tail(3))

    try:
        # Feature Engineering
        nifty['RSI'] = RSIIndicator(close=nifty['Close'], window=14).rsi()
        macd = MACD(close=nifty['Close'])
        nifty['MACD'] = macd.macd()
        nifty['Signal'] = macd.macd_signal()
        boll = BollingerBands(close=nifty['Close'])
        nifty['BB_High'] = boll.bollinger_hband()
        nifty['BB_Low'] = boll.bollinger_lband()

        # Label creation for training (next candle movement)
        future_period = 1
        nifty['Future_Close'] = nifty['Close'].shift(-future_period)
        nifty['Target'] = (nifty['Future_Close'] > nifty['Close']).astype(int)
        nifty.dropna(inplace=True)

        # Prepare training data
        features = ['RSI', 'MACD', 'Signal', 'BB_High', 'BB_Low']
        X = nifty[features]
        y = nifty['Target']

        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Simulate next 1 hour (30 steps x 2-min)
        st.subheader("🕒 Next 1 Hour Forecast (2-min Intervals)")

        future_steps = 30
        future_times = []
        future_preds = []
        future_confs = []

        latest_data = nifty.copy()

        for step in range(future_steps):
            # Recalculate indicators
            latest_data['RSI'] = RSIIndicator(close=latest_data['Close'], window=14).rsi()
            macd = MACD(close=latest_data['Close'])
            latest_data['MACD'] = macd.macd()
            latest_data['Signal'] = macd.macd_signal()
            boll = BollingerBands(close=latest_data['Close'])
            latest_data['BB_High'] = boll.bollinger_hband()
            latest_data['BB_Low'] = boll.bollinger_lband()

            input_row = latest_data[features].iloc[-1:].dropna()

            if input_row.empty:
                break

            pred = model.predict(input_row)[0]
            conf = model.predict_proba(input_row)[0][pred]

            last_price = latest_data['Close'].iloc[-1]
            next_price = last_price * (1 + (0.001 if pred == 1 else -0.001))

            next_time = latest_data.index[-1] + pd.Timedelta(minutes=2)
            next_row = latest_data.iloc[-1:].copy()
            next_row.index = [next_time]
            next_row['Close'] = next_price

            # Store forecast
            future_times.append(next_time.strftime('%H:%M'))
            future_preds.append("📈 UP" if pred == 1 else "📉 DOWN")
            future_confs.append(f"{conf:.2%}")

            # Add new row to data
            latest_data = pd.concat([latest_data, next_row])

        forecast_df = pd.DataFrame({
            "Time (IST)": future_times,
            "Prediction": future_preds,
            "Confidence": future_confs
        })

        st.dataframe(forecast_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error during model training or forecasting: {e}")

else:
    st.warning("No data available. Please check your internet connection or try again later.")
