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
**Professional-grade** NIFTY 50 direction prediction for 5-minute candles
""")

# Data Loading with Robust Timezone Handling
@st.cache_data(ttl=300, show_spinner="Fetching live market data...")
def load_data():
    try:
        data = yf.download("^NSEI", interval="5m", period="15d", progress=False)
        
        # Clean and verify data
        if data.empty:
            raise ValueError("Empty DataFrame returned from yfinance")
            
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].ffill().dropna()
        
        # Convert index to timezone-aware (UTC first, then IST)
        if not hasattr(data.index, 'tz'):
            data.index = pd.to_datetime(data.index).tz_localize('UTC')
        data.index = data.index.tz_convert('Asia/Kolkata')
        
        # Filter for NSE market hours (9:15 AM to 3:30 PM IST)
        data = data.between_time('09:15', '15:30')
        
        return data
    
    except Exception as e:
        st.error(f"⚠️ Data Loading Failed: {str(e)}")
        return pd.DataFrame()

# Enhanced Feature Engineering (unchanged)
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
    
    return df.dropna()

# Main Execution
nifty_data = load_data()

if not nifty_data.empty:
    with st.spinner("Processing market data..."):
        try:
            # Feature Engineering
            df = add_features(nifty_data)
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            df = df.dropna()
            
            # Prepare Data
            features = [col for col in df.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
            X = df[features]
            y = df['Target']
            
            # Model Training
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            model = GradientBoostingClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)
            
            # Current Prediction
            latest_features = X.iloc[[-1]]
            current_pred = model.predict(latest_features)[0]
            current_proba = model.predict_proba(latest_features)[0][current_pred]
            
            # Display Results
            col1, col2 = st.columns(2)
            col1.metric("Current Prediction", 
                       "BULLISH ↗️" if current_pred == 1 else "BEARISH ↘️",
                       f"{current_proba:.1%} confidence")
            col2.metric("Model Accuracy", 
                       f"{accuracy_score(y_test, model.predict(X_test)):.1%}")
            
            # Price Chart
            fig = px.line(df[-100:], x=df.index[-100:], y='Close')
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"⚠️ Processing Error: {str(e)}")
else:
    st.warning("""
    No market data available. Possible reasons:
    1. Outside NSE trading hours (9:15 AM - 3:30 PM IST)
    2. Connection issues with Yahoo Finance
    3. Market holiday
    """)

st.caption("Last Updated: " + datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S IST"))
