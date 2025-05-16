import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("📈 NIFTY Index Movement Predictor (AI-based)")
st.markdown("This model predicts whether NIFTY will go **UP or DOWN** in the next 5 minutes using technical indicators.")

# --- Manual Refresh Button ---
if st.button("🔄 Refresh Data Now"):
    st.cache_data.clear()  # Clear cache to reload fresh data

# --- Cached Data Loader (auto-refresh every 2 minutes) ---
@st.cache_data(ttl=120)  # Refresh cache every 120 seconds (2 minutes)
def load_data():
    try:
        data = yf.download("^NSEI", interval="5m", period="5d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)  # flatten columns if needed
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data = data.dropna()
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data = data.dropna(subset=['Close'])
        return data
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return pd.DataFrame()

# --- Load Data ---
nifty = load_data()

if not nifty.empty:
    # Show last few rows
    st.subheader("📊 Latest NIFTY Data (5-min)")
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

        # Create label
        future_period = 1
        nifty['Future_Close'] = nifty['Close'].shift(-future_period)
        nifty['Target'] = (nifty['Future_Close'] > nifty['Close']).astype(int)

        # Drop NaN rows
        nifty.dropna(inplace=True)

        # Model inputs
        features = ['RSI', 'MACD', 'Signal', 'BB_High', 'BB_Low']
        X = nifty[features]
        y = nifty['Target']

        # Train Random Forest Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Prediction
        latest_input = nifty.iloc[-1:][features]
        prediction = model.predict(latest_input)[0]
        confidence = model.predict_proba(latest_input)[0][prediction]

        # Show results
        movement = "📈 UP" if prediction == 1 else "📉 DOWN"
        color = "green" if prediction == 1 else "red"
        st.subheader("🔮 Prediction")
        st.markdown(f"### Predicted Next 5-min Move: <span style='color:{color}'>{movement}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {confidence:.2%}")

        # Show chart
        st.subheader("📉 NIFTY Price Chart")
        st.line_chart(nifty['Close'])

    except Exception as e:
        st.error(f"Error during model prediction or indicators: {e}")
else:
    st.warning("No data available. Please check your internet connection or try again later.")
