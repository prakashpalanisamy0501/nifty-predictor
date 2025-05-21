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
st.markdown("Predicts whether NIFTY will go **UP or DOWN** in the next 5, 15, 30, and 60 minutes using technical indicators.")

# Manual refresh
if st.button("🔄 Refresh Now"):
    st.cache_data.clear()

# Auto-refresh every 2 minutes
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
        base_index = nifty.index[-2]
        base_row = nifty.loc[base_index]

        prediction_intervals = [5, 15, 30, 60]
        st.subheader("🔮 AI Predictions")

        for mins in prediction_intervals:
            steps = mins // 5
            future_col = f"Future_Close_{mins}"
            target_col = f"Target_{mins}"
            nifty[future_col] = nifty['Close'].shift(-steps)
            nifty[target_col] = (nifty[future_col] > nifty['Close']).astype(int)

        nifty.dropna(inplace=True)

        X = nifty[features]

        for mins in prediction_intervals:
            y = nifty[f"Target_{mins}"]
            model = RandomForestClassifier(n_estimators=100, random_state=mins)
            model.fit(X, y)

            input_features = base_row[features].values.reshape(1, -1)
            prediction = model.predict(input_features)[0]
            confidence = model.predict_proba(input_features)[0][prediction]
            direction = "📈 UP" if prediction == 1 else "📉 DOWN"
            color = "green" if prediction == 1 else "red"

            st.markdown(
                f"### Next {mins} Minutes: <span style='color:{color}'>{direction}</span> "
                f"(Confidence: {confidence:.2%})",
                unsafe_allow_html=True
            )

        # Movement Table based on same candle
        st.subheader("⏱️ 5-Minute Interval Movement Table")
        movement_rows = []
        base_loc = nifty.index.get_loc(base_index)

        for step in range(1, 13):  # 5 to 60 min
            mins = step * 5
            future_loc = base_loc + step
            if future_loc < len(nifty):
                current_close = base_row['Close']
                future_close = nifty.iloc[future_loc]['Close']
                direction = "UP" if future_close > current_close else "DOWN"
                movement_rows.append({"Minutes Ahead": mins, "Direction": direction})

        movement_df = pd.DataFrame(movement_rows).set_index("Minutes Ahead")
        st.dataframe(movement_df)

        st.markdown(f"**Predictions Based on Candle Ending At (IST):** `{base_index.strftime('%Y-%m-%d %H:%M:%S')}`")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

else:
    st.warning("No data available. Please check your connection.")
