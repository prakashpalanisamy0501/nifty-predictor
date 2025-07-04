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
st.set_page_config(layout="wide", page_title="NIFTY AI Predictor")

# Title
st.title("🚀 NIFTY 50 5-Minute Predictor")
st.markdown("Real-time predictions during NSE market hours (9:15 AM - 3:30 PM IST)")

# Improved Data Loading with Retry Mechanism
@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def load_data():
    try:
        # Try multiple time periods if needed
        for days in [1, 3, 7]:
            data = yf.download(
                "^NSEI",
                interval="5m",
                period=f"{days}d",
                progress=False,
                timeout=10  # Faster timeout
            )
            
            if not data.empty:
                # Clean data
                data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
                data = data.dropna().ffill()
                
                # Convert timezone (UTC → IST)
                data.index = pd.to_datetime(data.index)
                if data.index.tz is None:
                    data.index = data.index.tz_localize('UTC')
                data.index = data.index.tz_convert('Asia/Kolkata')
                
                # Filter market hours
                data = data.between_time('09:15', '15:30')
                
                if len(data) > 10:  # Minimum viable data
                    return data
                
        raise ValueError("No valid data found after multiple attempts")
        
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

# Feature Engineering (unchanged)
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
    
    return df.dropna()

# Main App Logic
def main():
    nifty_data = load_data()
    
    if nifty_data.empty:
        ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
        market_open = ist_now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = ist_now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        st.warning(f"""
        ## 📉 No Live Data Available
        - Current IST Time: {ist_now.strftime('%Y-%m-%d %H:%M:%S')}
        - Possible Reasons:
          1. Outside market hours (9:15 AM - 3:30 PM IST)
          2. Weekend/holiday
          3. Yahoo Finance API issue
        - Next market open: {market_open.strftime('%Y-%m-%d %H:%M')} (in {market_open - ist_now} hours)
        """)
        return
    
    # Process data
    df = add_features(nifty_data)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    
    # Train model
    features = [col for col in df.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
    X_train, X_test, y_train, y_test = train_test_split(df[features], df['Target'], test_size=0.2, shuffle=False)
    model = GradientBoostingClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Display results
    st.success(f"✅ Live data loaded ({len(df)} candles up to {df.index[-1].strftime('%H:%M IST')})")
    
    col1, col2 = st.columns(2)
    with col1:
        latest_pred = model.predict(df[features].iloc[[-1]])[0]
        st.metric("Next 5-min Prediction", 
                 "↑ UP" if latest_pred == 1 else "↓ DOWN",
                 model.predict_proba(df[features].iloc[[-1]])[0][latest_pred].round(2))
    
    with col2:
        st.metric("Model Accuracy", f"{accuracy_score(y_test, model.predict(X_test)):.1%}")
    
    # Price chart
    st.plotly_chart(px.line(df[-100:], x=df.index[-100:], y='Close', title="NIFTY 50 Price"), use_container_width=True)

if __name__ == "__main__":
    main()
