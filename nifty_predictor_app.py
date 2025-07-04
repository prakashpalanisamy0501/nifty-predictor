import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolume
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pytz
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Title
st.title("🚀 NIFTY 5-Minute Scalping Predictor")
st.markdown("""
**Enhanced AI model** predicting NIFTY direction for upcoming 5-minute candles with:
- Advanced technical indicators
- Gradient Boosting classifier
- Historical accuracy metrics
- Multi-timeframe features
""")

# Manual Refresh Button
if st.button("🔄 Refresh Data Now"):
    st.cache_data.clear()

# Load 5-minute data
@st.cache_data(ttl=300)  # 5-minute cache
def load_data():
    try:
        # Get 15 days of 5-minute data
        data = yf.download("^NSEI", interval="5m", period="15d", progress=False)
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

# Feature Engineering Function
def add_features(df):
    # Price Momentum
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    df['Stoch_%K'] = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14).stoch()
    
    # Trend Indicators
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['ADX'] = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).adx()
    
    # Volatility
    boll = BollingerBands(close=df['Close'])
    df['BB_Upper'] = boll.bollinger_hband()
    df['BB_Lower'] = boll.bollinger_lband()
    df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    
    # Volume-based
    df['OBV'] = OnBalanceVolume(close=df['Close'], volume=df['Volume']).on_balance_volume()
    df['VWAP'] = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).volume_weighted_average_price()
    
    # Price Derivatives
    df['Returns_5'] = df['Close'].pct_change(5)
    df['Returns_15'] = df['Close'].pct_change(15)
    
    # Lagged Features
    for lag in [1, 2, 3]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
    
    return df.dropna()

# Load Data
nifty = load_data()

if not nifty.empty:
    st.subheader("📊 Latest NIFTY 5-min Data")
    st.dataframe(nifty.tail(3))

    try:
        # Feature Engineering
        nifty = add_features(nifty)
        
        # Create target (next candle direction)
        nifty['Target'] = (nifty['Close'].shift(-1) > nifty['Close']).astype(int)
        nifty.dropna(inplace=True)
        
        # Prepare training data
        features = [col for col in nifty.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
        X = nifty[features]
        y = nifty['Target']
        
        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        # Train model
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            subsample=0.8
        )
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        st.subheader("🧠 Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("Training Accuracy", f"{train_acc:.2%}")
        col2.metric("Test Accuracy", f"{test_acc:.2%}", 
                   delta_color="inverse" if test_acc < 0.55 else "normal")
        
        # Predict next 1 hour (12 steps x 5-min)
        st.subheader("🔮 Next 1 Hour Forecast (5-min Intervals)")
        
        future_steps = 12
        future_data = []
        current_data = nifty.copy()
        
        for step in range(future_steps):
            # Prepare current feature set
            current_features = current_data[features].iloc[[-1]]
            
            # Predict next movement
            pred = model.predict(current_features)[0]
            proba = model.predict_proba(current_features)[0][pred]
            
            # Generate next candle (simple projection)
            last_row = current_data.iloc[-1].copy()
            next_time = last_row.name + pd.Timedelta(minutes=5)
            
            # Simple price projection based on prediction
            price_multiplier = 1.0003 if pred == 1 else 0.9997
            next_close = last_row['Close'] * price_multiplier
            
            # Create next candle (simplified OHLC)
            next_candle = pd.Series({
                'Open': last_row['Close'],
                'High': max(last_row['Close'], next_close),
                'Low': min(last_row['Close'], next_close),
                'Close': next_close,
                'Volume': last_row['Volume']  # Maintain similar volume
            }, name=next_time)
            
            # Add to future predictions
            future_data.append({
                "Time (IST)": next_time.strftime('%H:%M'),
                "Predicted Move": "📈 UP" if pred == 1 else "📉 DOWN",
                "Confidence": f"{proba:.2%}",
                "Projected Close": f"{next_close:.2f}"
            })
            
            # Add to current data for recursive feature calculation
            current_data = pd.concat([current_data, next_candle.to_frame().T])
            current_data = add_features(current_data)
        
        # Display forecast
        forecast_df = pd.DataFrame(future_data)
        st.dataframe(forecast_df.style.applymap(
            lambda x: 'background-color: #e6f7e6' if 'UP' in str(x) else 'background-color: #ffe6e6', 
            subset=['Predicted Move']
        ), use_container_width=True)
        
        # Add disclaimer
        st.caption("⚠️ Note: Projections are AI-generated estimates for educational purposes only. "
                  "Actual market movements may vary significantly. Always use proper risk management.")

    except Exception as e:
        st.error(f"System error: {str(e)}")
        st.stop()

else:
    st.warning("No data available. Please check your internet connection or try again later.")
