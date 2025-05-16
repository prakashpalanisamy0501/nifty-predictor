import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
import pytz

# Title
st.title("📈 NIFTY Index Movement Predictor (AI-based)")
st.markdown("This model predicts whether NIFTY will go **UP or DOWN** in the next 5 minutes using technical indicators.")

# Manual Refresh Button
if st.button("🔄 Refresh Data Now"):
    st.cache_data.clear()

# Load Data Function with IST timezone and 2-minute cache
@st.cache_data(ttl=120)
def load_data():
    try:
        data = yf.download("^NSEI", interval="5m", period="5d", progress=False)
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
    # Show recent data
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

        # Label
        future_period = 1
        nifty['Future_Close'] = nifty['Close'].shift(-future_period)
        nifty['Target'] = (nifty['Future_Close'] > nifty['Close']).astype(int)
        nifty.dropna(inplace=True)

        # Features and Labels
        features = ['RSI', 'MACD', 'Signal', 'BB_High', 'BB_Low']
        X = nifty[features]
        y = nifty['Target']

        # Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Prediction
        latest_input = nifty.iloc[-1:][features]
        prediction = model.predict(latest_input)[0]
        confidence = model.predict_proba(latest_input)[0][prediction]

        # Display prediction
        movement = "📈 UP" if prediction == 1 else "📉 DOWN"
        color = "green" if prediction == 1 else "red"
        st.subheader("🔮 Prediction")
        st.markdown(f"### Predicted Next 5-min Move: <span style='color:{color}'>{movement}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {confidence:.2%}")

        # Show timestamp in IST
        last_time = nifty.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z')
        st.markdown(f"**Last Data Time (IST):** {last_time}")

        # Plot chart with prediction
        st.subheader("📉 NIFTY Price Chart with Prediction")
        fig, ax = plt.subplots(figsize=(10, 4))

        # Plot last 30 points
        ax.plot(nifty.index[-30:], nifty['Close'].iloc[-30:], label='NIFTY Close', color='blue', marker='o')

        # Current point
        last_price = nifty['Close'].iloc[-1]
        last_index = nifty.index[-1]
        ax.annotate('Current', xy=(last_index, last_price), xytext=(last_index, last_price + 10),
                    arrowprops=dict(facecolor='black', shrink=0.05), fontsize=9)

        # Arrow for prediction
        arrow_dir = 1 if prediction == 1 else -1
        arrow_color = 'green' if prediction == 1 else 'red'
        ax.annotate('Next Move',
                    xy=(last_index, last_price),
                    xytext=(last_index + pd.Timedelta(minutes=5), last_price + (arrow_dir * 20)),
                    arrowprops=dict(facecolor=arrow_color, shrink=0.05),
                    fontsize=10, color=arrow_color)

        ax.set_title("NIFTY Close Price (Last 30 Intervals)")
        ax.set_xlabel("Time (IST)")
        ax.set_ylabel("Price")
        ax.grid(True)
        fig.autofmt_xdate()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during model prediction or chart rendering: {e}")
else:
    st.warning("No data available. Please check your internet connection or try again later.")
