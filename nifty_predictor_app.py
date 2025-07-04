import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objs as go

# --- Parameters ---
NIFTY_TICKER = "^NSEI"
INTERVAL = "5m"
LOOKBACK = 12  # Use last 1 hour (12x5min) for prediction
PRED_STEPS = 12  # Predict next 1 hour (12x5min)
SHOW_LAST = 3  # Show last 3 data points

# --- Data Loading ---
@st.cache_data
def load_data():
    data = yf.download(tickers=NIFTY_TICKER, period="5d", interval=INTERVAL, progress=False)
    data = data.dropna()
    return data

data = load_data()

st.title("Nifty Index 1-Hour Movement Prediction (5-min Intervals)")

# --- Show Last 3 Data Points ---
st.subheader("Last 3 Data Points (5-min interval)")
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
from sklearn.metrics import mean_squared_error

y_true = scaler.inverse_transform(y[-100:])
y_pred = scaler.inverse_transform(model.predict(X[-100:]))
mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
std_dev = np.sqrt(mse)

conf_intervals = [(p - 1.96*std_dev, p + 1.96*std_dev) for p in pred]
conf_levels = [f"±{std_dev:.2f}" for _ in pred]

# --- Results Table ---
future_times = pd.date_range(data.index[-1], periods=PRED_STEPS+1, freq="5min")[1:]
result_df = pd.DataFrame({
    "Time": future_times,
    "Predicted Close": pred,
    "Confidence Interval": [f"{low:.2f} - {high:.2f}" for low, high in conf_intervals],
    "Confidence (Std Dev)": conf_levels
})

st.subheader("Next 1 Hour Prediction (5-min intervals)")
st.table(result_df)

# --- Plot ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index[-50:], y=data['Close'][-50:], mode='lines+markers', name='Actual'))
fig.add_trace(go.Scatter(x=future_times, y=pred, mode='lines+markers', name='Predicted'))
fig.update_layout(title="Nifty Index: Actual vs Predicted (Next 1 Hour)", xaxis_title="Time", yaxis_title="Nifty Close")
st.plotly_chart(fig)
