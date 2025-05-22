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
st.title("📈 NIFTY Quick Profit Predictor (AI-based)")
st.markdown("Predicts whether NIFTY will go **UP or DOWN** in the next few minutes or 1 hour using technical indicators. Ideal for ₹1000 profit scalping.")

# Timeframe selection
timeframe = st.selectbox("⏱ Select Timeframe for Trading", ['2m', '5m', '15m'])

# Manual Refresh Button
if st.button("🔄 Refresh Data Now"):
    st.cache_data.clear()

# Load Data
@st.cache_data(ttl=120)
def load_data(interval):
    try:
        data = yf.download("^NSEI", interval=interval, period="5d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data = data.dropna()
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data = data.dropna(subset=['Close'])
        if data.index.tzinfo is None or data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        data.index = data.index.tz_convert('Asia/Kolkata')
        return data
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return pd.DataFrame()

nifty = load_data(timeframe)

if not nifty.empty:
    st.subheader("📊 Latest NIFTY Data")
    st.dataframe(nifty.tail(3))

    try:
        # Feature Engineering
        nifty['RSI'] = RSIIndicator(close=nifty['Close'], window=10).rsi()
        macd = MACD(close=nifty['Close'], window_slow=26, window_fast=12, window_sign=9)
        nifty['MACD'] = macd.macd()
        nifty['Signal'] = macd.macd_signal()
        boll = BollingerBands(close=nifty['Close'], window=20, window_dev=2)
        nifty['BB_High'] = boll.bollinger_hband()
        nifty['BB_Low'] = boll.bollinger_lband()

        # Helper to train model & predict
        def train_and_predict(future_shift):
            df = nifty.copy()
            df['Future_Close'] = df['Close'].shift(-future_shift)
            df['Target'] = (df['Future_Close'] > df['Close']).astype(int)
            df.dropna(inplace=True)
            features = ['RSI', 'MACD', 'Signal', 'BB_High', 'BB_Low']
            X = df[features]
            y = df['Target']
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            latest_input = df.iloc[-1:][features]
            prediction = model.predict(latest_input)[0]
            confidence = model.predict_proba(latest_input)[0][prediction]
            price_now = df.iloc[-1]['Close']
            price_future = df.iloc[-1]['Future_Close']
            return prediction, confidence, price_now, price_future

        # 5-min prediction
        pred_5m, conf_5m, price_now, price_5m = train_and_predict(1)
        # 1-hour prediction (12 candles ahead for 5m, 30 for 2m, 4 for 15m)
        shift_map = {'2m': 30, '5m': 12, '15m': 4}
        future_shift_1h = shift_map.get(timeframe, 12)
        pred_1h, conf_1h, _, price_1h = train_and_predict(future_shift_1h)

        # Display results
        st.subheader("🔮 Predictions")
        up_down_5m = "📈 UP" if pred_5m == 1 else "📉 DOWN"
        up_down_1h = "📈 UP" if pred_1h == 1 else "📉 DOWN"
        color_5m = "green" if pred_5m == 1 else "red"
        color_1h = "green" if pred_1h == 1 else "red"

        st.markdown(f"### <span style='color:{color_5m}'>Next {timeframe} Move: {up_down_5m}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {conf_5m:.2%} | **Price Now:** ₹{price_now:.2f} | **Expected:** ₹{price_5m:.2f}")

        # ₹1000 target check
        price_diff = abs(price_5m - price_now)
        point_target = 20  # ~₹1000 in futures (50 units × 20 pts)
        if price_diff >= point_target:
            st.success(f"✅ Expected move is ~{price_diff:.2f} points → Potential ₹1000+ profit")
        else:
            st.warning(f"⚠️ Expected move is only ~{price_diff:.2f} points → May not hit ₹1000 target")

        # 1-hour prediction table
        st.subheader("🕒 One-Hour Prediction")
        st.markdown(f"<span style='color:{color_1h}'><b>Next 1 Hour Move: {up_down_1h}</b></span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {conf_1h:.2%} | **Expected Price:** ₹{price_1h:.2f}")

        # Timestamp
        st.markdown(f"**Last Data Time (IST):** {nifty.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z')}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

else:
    st.warning("No data available. Please check your internet connection or try again later.")
