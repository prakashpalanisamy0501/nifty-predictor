import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import pytz
import requests
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, time

# --- Pushbullet notification function ---
def send_pushbullet_notification(title, body, api_key):
    data_send = {"type": "note", "title": title, "body": body}
    resp = requests.post(
        'https://api.pushbullet.com/v2/pushes',
        data=data_send,
        headers={'Access-Token': api_key}
    )
    return resp.status_code == 200

# --- Your Pushbullet API Key ---
PUSHBULLET_API_KEY = "YOUR_PUSHBULLET_API_KEY"  # <-- Replace with your Pushbullet API key

# --- Parameters ---
NIFTY_TICKER = "^NSEI"
INTERVAL = "5m"
LOOKBACK = 12
PRED_STEPS = 12
SHOW_LAST = 3

IST = pytz.timezone('Asia/Kolkata')

def is_market_open(now_ist):
    market_open = time(9, 15)
    market_close = time(15, 30)
    return market_open <= now_ist.time() <= market_close

# --- Get current IST time ---
now_utc = datetime.utcnow()
now_ist = now_utc.replace(tzinfo=pytz.utc).astimezone(IST)

# --- Auto-refresh every 5 minutes during market hours ---
if is_market_open(now_ist):
    st_autorefresh(interval=300_000, limit=None, key="auto_refresh")

st.title("Nifty Index 1-Hour Movement Prediction (5-min Intervals)")

if st.button("Refresh Data"):
    st.cache_data.clear()

# --- Show market status ---
if is_market_open(now_ist):
    st.info("Market is OPEN. Showing live predictions.")
else:
    st.info("Market is CLOSED. Showing predictions based on the last trading session.")

@st.cache_data
def load_data():
    # yfinance will always return latest available data, so this logic suffices
    data = yf.download(
        tickers=NIFTY_TICKER,
        period="5d",
        interval=INTERVAL,
        progress=False
    )
    data = data.dropna()
    if data.empty:
        return pd.DataFrame()
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC').tz_convert(IST)
    else:
        data.index = data.index.tz_convert(IST)
    return data

data = load_data()

if data.empty or len(data) < LOOKBACK + PRED_STEPS + 1:
    st.error("Not enough data loaded. Please check your internet connection or ticker symbol, or try again later.")
    st.stop()

st.subheader("Last 3 Data Points (5-min interval, IST)")
st.write(data.tail(SHOW_LAST)[['Open', 'High', 'Low', 'Close', 'Volume']])

scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(data[['Close']])

X, y = [], []
for i in range(LOOKBACK, len(scaled_close) - PRED_STEPS):
    X.append(scaled_close[i-LOOKBACK:i, 0])
    y.append(scaled_close[i:i+PRED_STEPS, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(PRED_STEPS))
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_model((LOOKBACK, 1))
model.fit(X[-1000:], y[-1000:], epochs=10, batch_size=32, verbose=0)

last_sequence = scaled_close[-LOOKBACK:].reshape(1, LOOKBACK, 1)
pred_scaled = model.predict(last_sequence)
pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

# --- Per-step accuracy calculation ---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true != 0
    y_true = y_true[non_zero_idx]
    y_pred = y_pred[non_zero_idx]
    if len(y_true) == 0:
        return 100.0
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

per_step_accuracy = []
for step in range(PRED_STEPS):
    y_true_step = scaler.inverse_transform(y[-100:, step].reshape(-1, 1)).flatten()
    y_pred_step = scaler.inverse_transform(model.predict(X[-100:])[:, step].reshape(-1, 1)).flatten()
    mape = mean_absolute_percentage_error(y_true_step, y_pred_step)
    accuracy = max(0.0, 100 - mape)
    per_step_accuracy.append(accuracy)

# --- Confidence Estimation ---
y_true_all = scaler.inverse_transform(y[-100:])
y_pred_all = scaler.inverse_transform(model.predict(X[-100:]))
mse = mean_squared_error(y_true_all.flatten(), y_pred_all.flatten())
std_dev = np.sqrt(mse)
conf_intervals = [(p - 1.96*std_dev, p + 1.96*std_dev) for p in pred]
conf_levels = [f"±{std_dev:.2f}" for _ in pred]

future_times = pd.date_range(data.index[-1], periods=PRED_STEPS+1, freq="5min", tz=IST)[1:]
result_df = pd.DataFrame({
    "Time (IST)": future_times,
    "Predicted Close": pred,
    "Accuracy (%)": [f"{acc:.2f}" for acc in per_step_accuracy],
    "Confidence Interval": [f"{low:.2f} - {high:.2f}" for low, high in conf_intervals],
    "Confidence (Std Dev)": conf_levels
})

st.subheader("Next 1 Hour Prediction (5-min intervals, IST)")
st.table(result_df)

# --- Notification to phone if any interval accuracy > 90% ---
notify_rows = result_df[result_df["Accuracy (%)"].astype(float) > 90]
if not notify_rows.empty:
    st.success("🎉 One or more intervals have accuracy above 90%! Notification sent to your phone.")
    # Send notification (only once per refresh)
    message = "High Accuracy Prediction Intervals:\n"
    for idx, row in notify_rows.iterrows():
        message += f"{row['Time (IST)'].strftime('%H:%M')} - Accuracy: {row['Accuracy (%)']}%\n"
    send_pushbullet_notification(
        title="Nifty High Accuracy Alert",
        body=message,
        api_key=PUSHBULLET_API_KEY
    )
