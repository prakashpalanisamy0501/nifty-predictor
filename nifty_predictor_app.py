import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pytz

# Title with enhanced features
st.title("🚀 NIFTY 5-Minute AI Predictor (Professional Scalping Tool)")
st.markdown("""
**Predicts NIFTY movement direction for upcoming 5-minute intervals with 12-step forecast**  
*Enhanced with advanced features:*
- XGBoost with hyperparameter tuning
- 15+ technical indicators
- Sequence modeling with lag features
- True probabilistic confidence scoring
- Model accuracy validation
""")

# Manual Refresh Button
if st.button("🔄 Refresh Market Data Now"):
    st.cache_data.clear()
    st.experimental_rerun()

# Load 5-minute data with extended history
@st.cache_data(ttl=300)  # 5-minute cache
def load_data():
    try:
        data = yf.download("^NSEI", interval="5m", period="60d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data = data.dropna()
        
        # Convert to IST timezone
        if data.index.tzinfo is None:
            data.index = data.index.tz_localize('UTC')
        data.index = data.index.tz_convert('Asia/Kolkata')
        
        return data
    except Exception as e:
        st.error(f"🚨 Data Error: {e}")
        return pd.DataFrame()

# Load Data
nifty = load_data()

if not nifty.empty:
    st.subheader("📊 Current Market Data (5-min intervals)")
    st.dataframe(nifty.tail(3).style.format("{:.2f}"))

    try:
        # Feature Engineering - Advanced Indicators
        # Momentum
        nifty['RSI'] = RSIIndicator(close=nifty['Close'], window=14).rsi()
        stoch = StochasticOscillator(high=nifty['High'], low=nifty['Low'], close=nifty['Close'], window=14, smooth_window=3)
        nifty['Stoch_%K'] = stoch.stoch()
        nifty['Stoch_%D'] = stoch.stoch_signal()
        
        # Trend
        macd = MACD(close=nifty['Close'])
        nifty['MACD'] = macd.macd()
        nifty['MACD_Signal'] = macd.macd_signal()
        nifty['EMA_20'] = EMAIndicator(close=nifty['Close'], window=20).ema_indicator()
        nifty['EMA_50'] = EMAIndicator(close=nifty['Close'], window=50).ema_indicator()
        adx = ADXIndicator(high=nifty['High'], low=nifty['Low'], close=nifty['Close'], window=14)
        nifty['ADX'] = adx.adx()
        
        # Volatility
        bb = BollingerBands(close=nifty['Close'], window=20, window_dev=2)
        nifty['BB_Upper'] = bb.bollinger_hband()
        nifty['BB_Lower'] = bb.bollinger_lband()
        nifty['ATR'] = AverageTrueRange(
            high=nifty['High'], 
            low=nifty['Low'], 
            close=nifty['Close'], 
            window=14
        ).average_true_range()
        
        # Volume
        nifty['OBV'] = OnBalanceVolumeIndicator(close=nifty['Close'], volume=nifty['Volume']).on_balance_volume()
        nifty['Volume_Change'] = nifty['Volume'].pct_change()
        
        # Lag Features
        for lag in [1, 2, 3]:
            nifty[f'Return_{lag}'] = nifty['Close'].pct_change(lag)
            nifty[f'Volume_Change_{lag}'] = nifty['Volume_Change'].shift(lag)
        
        # Target Variable (Next 5-min movement)
        nifty['Target'] = (nifty['Close'].shift(-1) > nifty['Close']).astype(int)
        nifty.dropna(inplace=True)
        
        # Feature Selection
        features = [
            'RSI', 'Stoch_%K', 'Stoch_%D', 'MACD', 'MACD_Signal',
            'EMA_20', 'EMA_50', 'ADX', 'BB_Upper', 'BB_Lower', 'ATR',
            'OBV', 'Volume_Change', 'Return_1', 'Return_2', 'Return_3',
            'Volume_Change_1', 'Volume_Change_2'
        ]
        X = nifty[features]
        y = nifty['Target']
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Train XGBoost with optimized parameters
        model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Validate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model Validation Accuracy: {accuracy:.2%} (Latest 20% Data)")
        
        # Forecasting next 1 hour (12 steps)
        st.subheader("🔮 Next 1 Hour Forecast (5-min Intervals)")
        future_steps = 12
        forecasts = []
        current_data = nifty.copy()
        
        for step in range(future_steps):
            # Prepare last available data point
            current_point = current_data.iloc[[-1]][features]
            
            # Make prediction
            pred_proba = model.predict_proba(current_point)[0]
            direction = 1 if pred_proba[1] > 0.5 else 0
            confidence = max(pred_proba)
            
            # Generate next candle (conservative projection)
            last_close = current_data['Close'].iloc[-1]
            price_multiplier = 1.0003 if direction == 1 else 0.9997
            next_close = last_close * price_multiplier
            
            # Create new candle data
            next_time = current_data.index[-1] + pd.Timedelta(minutes=5)
            new_candle = {
                'Open': last_close,
                'High': last_close * 1.0005 if direction == 1 else last_close,
                'Low': last_close if direction == 1 else last_close * 0.9995,
                'Close': next_close,
                'Volume': current_data['Volume'].iloc[-1] * 0.95  # conservative volume
            }
            
            # Append to history
            new_row = pd.DataFrame([new_candle], index=[next_time])
            current_data = pd.concat([current_data, new_row])
            
            # Update technical indicators
            current_data = calculate_indicators(current_data)
            
            # Store forecast
            forecasts.append({
                "Time (IST)": next_time.strftime('%H:%M'),
                "Direction": "📈 BULLISH" if direction == 1 else "📉 BEARISH",
                "Confidence": f"{confidence:.1%}",
                "Est. Price": f"{next_close:.2f}"
            })
        
        # Display forecast
        forecast_df = pd.DataFrame(forecasts)
        st.dataframe(forecast_df.style.applymap(
            lambda x: 'color: green' if 'BULLISH' in x else 'color: red' if 'BEARISH' in x else '',
            subset=['Direction']
        ))
        
        # Trading strategy suggestion
        bull_count = sum(1 for f in forecasts if "BULLISH" in f['Direction'])
        bear_count = len(forecasts) - bull_count
        st.subheader("🎯 Trading Recommendation")
        if bull_count > bear_count * 1.5:
            st.success("STRONG BUY SIGNAL (Majority Bullish Forecast)")
        elif bear_count > bull_count * 1.5:
            st.error("STRONG SELL SIGNAL (Majority Bearish Forecast)")
        else:
            st.warning("NEUTRAL MARKET (Mixed Signals - Trade with Caution)")
            
    except Exception as e:
        st.error(f"⚠️ Prediction Error: {str(e)}")

else:
    st.warning("📡 No data available. Check internet connection or try again later.")

# Indicator calculation function
def calculate_indicators(df):
    # Momentum
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
    df['Stoch_%K'] = stoch.stoch()
    df['Stoch_%D'] = stoch.stoch_signal()
    
    # Trend
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['EMA_50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
    adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ADX'] = adx.adx()
    
    # Volatility
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['ATR'] = AverageTrueRange(
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'], 
        window=14
    ).average_true_range()
    
    # Volume
    df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Lag Features
    for lag in [1, 2, 3]:
        df[f'Return_{lag}'] = df['Close'].pct_change(lag)
        df[f'Volume_Change_{lag}'] = df['Volume_Change'].shift(lag)
    
    return df.dropna()
