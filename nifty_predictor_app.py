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
st.title("📈 NIFTY Index Movement Predictor (AI-based)")
st.markdown("This model predicts whether NIFTY will go **UP or DOWN** in the next 5 minutes and next **1 hour** (in 5-minute intervals) using technical indicators.")

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

        # Labels for 5-min ahead
        nifty['Future_Close_5m'] = nifty['Close'].shift(-1)
        nifty['Target_5m'] = (nifty['Future_Close_5m'] > nifty['Close']).astype(int)

        # Drop NaNs
        nifty.dropna(inplace=True)

        # Features and Labels
        features = ['RSI', 'MACD', 'Signal', 'BB_High', 'BB_Low']
        X = nifty[features]
        y = nifty['Target_5m']

        # Train Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # --- 5-Minute Prediction ---
        latest_input = nifty.iloc[-1:][features]
        prediction_5m = model.predict(latest_input)[0]
        confidence_5m = model.predict_proba(latest_input)[0][prediction_5m]

        movement_5m = "📈 UP" if prediction_5m == 1 else "📉 DOWN"
        color_5m = "green" if prediction_5m == 1 else "red"

        st.subheader("🔮 Prediction (Next 5 Minutes)")
        st.markdown(f"### Predicted Move: <span style='color:{color_5m}'>{movement_5m}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {confidence_5m:.2%}")

        # Show timestamp in IST
        last_time = nifty.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z')
        st.markdown(f"**Last Data Time (IST):** {last_time}")

        # --- One-Hour Ahead Prediction (12 steps of 5-min) ---
        st.subheader("🕐 One-Hour Forecast (5-min Steps)")

        # Clone latest values for simulation
        simulated = nifty.copy()

        future_predictions = []
        current_index = simulated.index[-1]

        for i in range(12):
            latest_row = simulated.iloc[-1:].copy()

            # Predict and simulate next step
            X_future = latest_row[features]
            pred = model.predict(X_future)[0]
            prob = model.predict_proba(X_future)[0][pred]

            move = 1 if pred == 1 else -1
            step_change = np.random.uniform(0.1, 0.3)  # Simulate % change

            new_close = latest_row['Close'].values[0] * (1 + move * step_change / 100)
            new_row = latest_row.copy()
            new_row['Close'] = new_close
            new_row['Open'] = new_row['High'] = new_row['Low'] = new_close
            new_row['Volume'] = new_row['Volume'].values[0]  # keep volume same
            new_row.index = [current_index + pd.Timedelta(minutes=5)]

            # Recalculate indicators
            new_row['RSI'] = RSIIndicator(close=pd.concat([simulated['Close'], pd.Series([new_close])]).reset_index(drop=True)).rsi().iloc[-1]
            macd_temp = MACD(close=pd.concat([simulated['Close'], pd.Series([new_close])]).reset_index(drop=True))
            new_row['MACD'] = macd_temp.macd().iloc[-1]
            new_row['Signal'] = macd_temp.macd_signal().iloc[-1]
            bb_temp = BollingerBands(close=pd.concat([simulated['Close'], pd.Series([new_close])]).reset_index(drop=True))
            new_row['BB_High'] = bb_temp.bollinger_hband().iloc[-1]
            new_row['BB_Low'] = bb_temp.bollinger_lband().iloc[-1]

            simulated = pd.concat([simulated, new_row])

            future_predictions.append({
                "Time (IST)": new_row.index[0].strftime('%H:%M'),
                "Predicted Move": "UP" if pred == 1 else "DOWN",
                "Confidence": f"{prob:.2%}"
            })

        future_df = pd.DataFrame(future_predictions)
        st.table(future_df)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.warning("No data available. Please check your internet connection or try again later.")
