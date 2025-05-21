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
st.title("📈 NIFTY AI-based Future Movement Predictor")
st.markdown("Predicts whether NIFTY will go **UP or DOWN** in the next 15, 30, and 60 minutes using technical indicators.")

# Manual refresh
if st.button("🔄 Refresh Now"):
    st.cache_data.clear()

@st.cache_data(ttl=120)
def load_data():
    data = yf.download("^NSEI", interval="5m", period="5d", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data.dropna(subset=['Close'])
    if data.index.tzinfo is None or data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    data.index = data.index.tz_convert("Asia/Kolkata")
    return data

nifty = load_data()

if not nifty.empty:
    st.subheader("📊 Latest NIFTY 5-min Data")
    st.dataframe(nifty.tail(3))

    try:
        # Technical indicators
        nifty['RSI'] = RSIIndicator(close=nifty['Close'], window=14).rsi()
        macd = MACD(close=nifty['Close'])
        nifty['MACD'] = macd.macd()
        nifty['Signal'] = macd.macd_signal()
        boll = BollingerBands(close=nifty['Close'])
        nifty['BB_High'] = boll.bollinger_hband()
        nifty['BB_Low'] = boll.bollinger_lband()
        nifty.dropna(inplace=True)

        features = ['RSI', 'MACD', 'Signal', 'BB_High', 'BB_Low']
        base_row = nifty.iloc[-2]  # second last (complete) row
        base_index = nifty.index[-2]

        # Prepare model and labels
        X = nifty[features]
        prediction_intervals = [15, 30, 60]
        results = []
        interval_table = []

        st.subheader("🔮 AI Predictions")
        for mins in prediction_intervals:
            shift_steps = mins // 5
            label = (nifty['Close'].shift(-shift_steps) > nifty['Close']).astype(int)
            y = label.loc[nifty.index.intersection(X.index)]
            model = RandomForestClassifier(n_estimators=100, random_state=mins)
            model.fit(X, y)

            # Prediction using the same row
            latest_input = base_row[features].values.reshape(1, -1)
            pred = model.predict(latest_input)[0]
            conf = model.predict_proba(latest_input)[0][pred]
            direction = "📈 UP" if pred == 1 else "📉 DOWN"
            color = "green" if pred == 1 else "red"
            st.markdown(f"### Next {mins} Minutes: <span style='color:{color}'>{direction}</span> (Confidence: {conf:.2%})", unsafe_allow_html=True)

            results.append((mins, pred, conf))

        # Interval Table from same row
        st.subheader("⏱️ 5-Minute Interval Movement Table")
        table = []
        for step in range(1, 13):
            mins = step * 5
            future_idx = nifty.index.get_loc(base_index) + step
            if future_idx < len(nifty):
                current_close = base_row['Close']
                future_close = nifty.iloc[future_idx]['Close']
                direction = "UP" if future_close > current_close else "DOWN"
                table.append({"Minutes Ahead": mins, "Direction": direction})
        st.dataframe(pd.DataFrame(table).set_index("Minutes Ahead"))

        st.markdown(f"**Prediction Based On Candle Ending At (IST):** {base_index.strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

else:
    st.warning("No data available. Please check your connection.")
