import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
import pytz

st.set_page_config(page_title="NIFTY Predictor", layout="centered")

# Title
st.title("📈 NIFTY AI-based Future Movement Predictor")
st.markdown("Predicts whether NIFTY will go **UP or DOWN** in the next 15, 30, and 60 minutes using technical indicators.")

# Manual refresh button
if st.button("🔄 Refresh Now"):
    st.cache_data.clear()

# Load NIFTY data (auto-refresh every 2 minutes)
@st.cache_data(ttl=120)
def load_data():
    data = yf.download("^NSEI", interval="5m", period="5d", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data.dropna(subset=['Close'])

    # Convert index to IST timezone
    if data.index.tzinfo is None or data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    data.index = data.index.tz_convert("Asia/Kolkata")
    return data

nifty = load_data()

if not nifty.empty:
    st.subheader("📊 Latest NIFTY 5-min Data")
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

        features = ['RSI', 'MACD', 'Signal', 'BB_High', 'BB_Low']
        nifty.dropna(inplace=True)

        # Create labels for 15, 30, and 60 minute future movement
        prediction_intervals = [15, 30, 60]
        predictions = {}

        for mins in prediction_intervals:
            steps = mins // 5
            nifty[f'Future_Close_{mins}'] = nifty['Close'].shift(-steps)
            nifty[f'Target_{mins}'] = (nifty[f'Future_Close_{mins}'] > nifty['Close']).astype(int)

        nifty.dropna(inplace=True)

        X = nifty[features]

        st.subheader("🔮 AI Predictions")

        # Use the second last row (fully formed 5-min candle)
        latest_input = nifty.iloc[[-2]][features]
        used_time = nifty.index[-2]

        for mins in prediction_intervals:
            y = nifty[f'Target_{mins}']
            model = RandomForestClassifier(n_estimators=100, random_state=mins)
            model.fit(X, y)
            pred = model.predict(latest_input)[0]
            conf = model.predict_proba(latest_input)[0][pred]
            direction = "📈 UP" if pred == 1 else "📉 DOWN"
            color = "green" if pred == 1 else "red"
            st.markdown(f"### Next {mins} Minutes: <span style='color:{color}'>{direction}</span> (Confidence: {conf:.2%})", unsafe_allow_html=True)
            predictions[mins] = (pred, conf)

        # Table of future movement labels
        st.subheader("⏱️ 5-Minute Interval Movement Table")
        movement_table = pd.DataFrame()
        for step in range(1, 13):  # 5 to 60 min
            mins = step * 5
            label_col = f'Target_{mins}'
            steps = mins // 5
            nifty[label_col] = (nifty['Close'].shift(-steps) > nifty['Close']).astype(int)
            direction = "UP" if nifty[label_col].iloc[-2] == 1 else "DOWN"
            movement_table.loc[mins, 'Direction'] = direction

        movement_table.index.name = "Minutes Ahead"
        st.dataframe(movement_table)

        # Show time of candle used
        st.markdown(f"**Prediction Based On Candle Ending At (IST):** {used_time.strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        st.error(f"Error during prediction or processing: {e}")
else:
    st.warning("No data available. Please check your internet connection.")
