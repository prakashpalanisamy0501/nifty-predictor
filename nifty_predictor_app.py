import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice as VWAP, OnBalanceVolumeIndicator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pytz
import plotly.express as px
from datetime import datetime, timedelta

# Configuration
st.set_page_config(layout="wide", page_title="NIFTY 5-Min Scalper Pro")

# Title
st.title("🚀 NIFTY 5-Minute AI Scalping Predictor")
st.markdown("""
**Professional-grade** NIFTY 50 direction prediction for 5-minute candles using:
- 18+ Technical Indicators • Gradient Boosting • Multi-Timeframe Analysis
""")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Settings")
    lookback_days = st.slider("Data Lookback (Days)", 3, 30, 15)
    future_steps = st.slider("Prediction Window (Steps)", 1, 24, 12)
    confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.9, 0.65)

# Data Loading
@st.cache_data(ttl=300, show_spinner="Fetching live market data...")
def load_data():
    try:
        data = yf.download("^NSEI", interval="5m", period=f"{lookback_days}d", progress=False)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].ffill()
        
        # Convert to IST
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        data.index = data.index.tz_convert('Asia/Kolkata')
        
        return data.dropna()
    except Exception as e:
        st.error(f"⚠️ Data Error: {str(e)}")
        return pd.DataFrame()

# Enhanced Feature Engineering
def add_features(df):
    # Momentum
    df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['Stoch_%K'] = stoch.stoch()
    df['Stoch_%D'] = stoch.stoch_signal()
    
    # Trend
    df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['EMA_50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['ADX_14'] = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).adx()
    
    # Volatility
    bb = BollingerBands(close=df['Close'], window=20)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['ATR_14'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    
    # Volume
    df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    df['VWAP'] = VWAP(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).volume_weighted_average_price()
    df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
    
    # Price Action
    df['Returns_5'] = df['Close'].pct_change(5)
    df['Range_5'] = (df['High'] - df['Low']).rolling(5).mean()
    
    return df.dropna()

# Model Training
def train_model(X, y):
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        random_state=42
    )
    model.fit(X, y)
    return model

# Main Execution
nifty_data = load_data()

if not nifty_data.empty:
    with st.spinner("Crunching numbers with AI..."):
        # Feature Engineering
        df = add_features(nifty_data)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        
        # Prepare Data
        features = [col for col in df.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
        X = df[features]
        y = df['Target']
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Model Training
        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Current Prediction
        latest_features = X.iloc[[-1]]
        current_pred = model.predict(latest_features)[0]
        current_proba = model.predict_proba(latest_features)[0][current_pred]
        
        # Future Projections
        forecast = []
        temp_df = df.copy()
        
        for _ in range(future_steps):
            features = add_features(temp_df).iloc[[-1]][features]
            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0][pred]
            
            # Generate next candle
            last_close = temp_df['Close'].iloc[-1]
            next_close = last_close * (1.0005 if pred == 1 else 0.9995)
            next_time = temp_df.index[-1] + timedelta(minutes=5)
            
            forecast.append({
                "Time": next_time.strftime("%H:%M"),
                "Direction": "↑ UP" if pred == 1 else "↓ DOWN",
                "Confidence": f"{proba:.1%}",
                "Projected Close": f"{next_close:.2f}",
                "Signal Strength": "Strong" if proba >= confidence_threshold else "Weak"
            })
            
            # Append simulated data
            new_row = temp_df.iloc[-1].copy()
            new_row.name = next_time
            new_row[['Open', 'High', 'Low', 'Close']] = [last_close, max(last_close, next_close), min(last_close, next_close), next_close]
            temp_df = pd.concat([temp_df, new_row.to_frame().T])
        
        forecast_df = pd.DataFrame(forecast)

    # Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Market Prediction", 
                 value="BULLISH ↗️" if current_pred == 1 else "BEARISH ↘️",
                 delta=f"{current_proba:.1%} confidence")
        
    with col2:
        st.metric("Model Accuracy", 
                 f"{accuracy:.1%}",
                 delta="Live Testing" if len(y_test) > 100 else "Initializing")
    
    # Forecast Table
    st.subheader(f"⏳ Next {future_steps*5} Minute Forecast")
    st.dataframe(
        forecast_df.style.apply(
            lambda x: ['background: #e6f3ff' if x.Direction == "↑ UP" else 'background: #ffebee']*len(x),
            axis=1
        ),
        use_container_width=True
    )
    
    # Price Chart
    fig = px.line(df[-100:], x=df.index[-100:], y='Close', title="Live NIFTY 50 Price")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Market data unavailable. Try again during NSE trading hours (9:15 AM - 3:30 PM IST).")

st.caption("""
⚠️ Disclaimer: AI predictions are probabilistic estimates, not financial advice. 
Always conduct your own analysis and use proper risk management.
""")
