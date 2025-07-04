import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import pytz

# --- Parameters ---
NIFTY_TICKER = "^NSEI"
INTERVAL = "5m"
LOOKBACK = 12  # Use last 1 hour (12x5min) for prediction
PRED_STEPS = 12  # Predict next 1 hour (12x5min)
SHOW_LAST = 3  # Show last 3 data points

IST = pytz.timezone('Asia/Kolkata')

st.title("Nifty Index 1-Hour Movement Prediction (5-min Intervals)")

# --- Manual Refresh Button ---
if st.button("Refresh Data"):
    st.cache_data.clear()

# --- Data Loading with Error Handling ---
@st.cache_data
def load_data():
    try:
        data = yf.download(
            tickers=NIFTY_TICKER,
            period="5d",
            interval=INTERVAL,
            progress=False
        )
        data = data.dropna()
        if data.empty:
            return pd.DataFrame()
        # Convert index to IST
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert(IST)
        else:
            data.index = data.index.tz_convert(IST)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

data = load_data()

# --- Validate Data ---
if data.empty or len(data) < LOOKBACK + PRED_STEPS + 1:
    st.error("Not enough data loaded. Please check your internet connection or ticker symbol, or try again later.")
    st.stop()

# --- Show Last 3 Data Points (Always latest, not random) ---
st.subheader("Last 3 Data Points (5-min interval, IST)")
st.write(data.tail(SHOW_LAST)[['Open', 'High', 'Low', 'Close', 'Volume']])

# --- Data Preprocessing ---
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(data[['Close']])

X, y = [], []
for i in range(LOOKBACK, len(scaled_close) - PRED_STEPS):
    X.append(scaled_close[i-LOOKBACK:i, 0])
    y.append(scaled_close[i:i+PRED_STEPS, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# --- Model Definition ---
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(PRED_STEPS))
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Model Training (for demo, use last 1000 samples) ---
model = build_model((LOOKBACK, 1))
model.fit(X[-1000:], y[-1000:], epochs=10, batch_size=32, verbose=0)

# --- Prediction ---
last_sequence = scaled_close[-LOOKBACK:].reshape(1, LOOKBACK, 1)
pred_scaled = model.predict(last_sequence)
pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

# --- Confidence Estimation (using model's MSE on last 100 samples) ---
y_true = scaler.inverse_transform(y[-100:])
y_pred = scaler.inverse_transform(model.predict(X[-100:]))
mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
std_dev = np.sqrt(mse)

conf_intervals = [(p - 1.96*std_dev, p + 1.96*std_dev) for p in pred]
conf_levels = [f"±{std_dev:.2f}" for _ in pred]

# --- Results Table ---
future_times = pd.date_range(data.index[-1], periods=PRED_STEPS+1, freq="5min", tz=IST)[1:]
result_df = pd.DataFrame({
    "Time (IST)": future_times,
    "Predicted Close": pred,
    "Confidence Interval": [f"{low:.2f} - {high:.2f}" for low, high in conf_intervals],
    "Confidence (Std Dev)": conf_levels
})

st.subheader("Next 1 Hour Prediction (5-min intervals, IST)")
st.table(result_df)
