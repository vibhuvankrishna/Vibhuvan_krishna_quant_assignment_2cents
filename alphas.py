import numpy as np
from sklearn.ensemble import RandomForestClassifier
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import pandas as pd

import sklearn


import os
import warnings
import logging

# Suppress Python warnings
warnings.filterwarnings("ignore")

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # only errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress absl logging (includes the compiled metrics warning)
logging.getLogger('absl').setLevel(logging.ERROR)


import sys
print(sys.executable)


print(sklearn.__version__)
# ----------------- Existing Momentum & Mean Reversion -----------------

def alpha_momentum(symbol, timeframe, df):
    return 1
    if len(df) < 11:
        return 0
    last = df['close'].iloc[-1]
    mean10 = df['close'].iloc[-11:-1].mean()
    if last > mean10 * 1.001:
        return 1
    if last < mean10 * 0.999:
        return -1
    return 0

def alpha_mean_revert(symbol, timeframe, df):
    if len(df) < 21:
        return 0
    short = df['close'].rolling(5).mean().iloc[-1]
    long = df['close'].rolling(20).mean().iloc[-1]
    if short > long * 1.002:
        return -1
    if short < long * 0.998:
        return 1
    return 0

# def alpha_random(symbol, timeframe, df):
#     return np.random.randint(-1,2)

def alpha_bollinger(symbol, timeframe, df, window=20, std_multiplier=2):
    """
    Returns signal based on Bollinger Bands:
    1 = Buy (price below lower band)
    -1 = Sell (price above upper band)
    0 = Hold
    """
    if len(df) < window:
        return 0

    close = df['close']
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    
    upper_band = sma + std_multiplier * std
    lower_band = sma - std_multiplier * std
    
    last_close = close.iloc[-1]
    
    if last_close < lower_band.iloc[-1]:
        return 1
    elif last_close > upper_band.iloc[-1]:
        return -1
    else:
        return 0



def alpha_supertrend(symbol, timeframe, df, period=10, multiplier=3):
    """
    Returns signal based on SuperTrend:
    1 = Buy, -1 = Sell, 0 = Hold
    """
    if len(df) < period:
        return 0

    high = df['high']
    low = df['low']
    close = df['close']

    # ATR calculation
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    hl2 = (high + low) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    
    # Supertrend direction
    supertrend = pd.Series(index=df.index)
    direction = 1  # 1 = bullish, -1 = bearish
    
    for i in range(1, len(df)):
        if close.iloc[i-1] <= upperband.iloc[i-1]:
            direction = 1
        elif close.iloc[i-1] >= lowerband.iloc[i-1]:
            direction = -1
        supertrend.iloc[i] = direction
    
    last_direction = supertrend.iloc[-1]
    
    return last_direction





import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# Load models
# -------------------------------
# Random Forest
with open("rf_model_300trees.pkl", "rb") as f:
    rf_model = pickle.load(f)

# LSTM
lstm_model = load_model("lstm_model.h5")

# -------------------------------
# Alpha function using both models
# -------------------------------
def alpha_pred_models(symbol, timeframe, df):
    """
    Returns 1 (buy), -1 (sell), 0 (hold)
    based on predictions from Random Forest and LSTM.
    """
    if len(df) < 61:  # need at least 60 bars for LSTM sequence
        return 0
    
    # --- Prepare Random Forest input ---
    # use last 5 closes
    lag_features = df['close'].iloc[-6:-1].values.reshape(1, -1)  # 5 lag features
    rf_pred = rf_model.predict(lag_features)[0]
    
    # --- Prepare LSTM input ---
    lstm_scaler = MinMaxScaler(feature_range=(0,1))
    closes_scaled = lstm_scaler.fit_transform(df['close'].values.reshape(-1,1))
    lstm_input = closes_scaled[-60:].reshape(1, 60, 1)
    lstm_pred_scaled = lstm_model.predict(lstm_input, verbose=0)[0][0]
    # inverse scale to original
    lstm_pred = lstm_scaler.inverse_transform(np.array([[lstm_pred_scaled]]))[0][0]
    
    # --- Compare predictions to last price ---
    last_close = df['close'].iloc[-1]
    
    rf_signal = 1 if rf_pred > last_close * 1.0005 else -1 if rf_pred < last_close * 0.9995 else 0
    lstm_signal = 1 if lstm_pred > last_close * 1.0005 else -1 if lstm_pred < last_close * 0.9995 else 0
    
    # Combine signals (simple majority)
    signal = rf_signal + lstm_signal
    if signal > 0:
        return 1
    elif signal < 0:
        return -1
    else:
        return 0


def alpha_lstm(symbol, timeframe, df):
    """
    Alpha using only the LSTM model.
    """
    if len(df) < 61:  # need last 60 bars for LSTM
        return 0
    
    # Scale data
    lstm_scaler = MinMaxScaler(feature_range=(0,1))
    closes_scaled = lstm_scaler.fit_transform(df['close'].values.reshape(-1,1))
    
    # Prepare LSTM input
    lstm_input = closes_scaled[-60:].reshape(1, 60, 1)
    lstm_pred_scaled = lstm_model.predict(lstm_input, verbose=0)[0][0]
    
    # Inverse scale
    lstm_pred = lstm_scaler.inverse_transform(np.array([[lstm_pred_scaled]]))[0][0]
    
    last_close = df['close'].iloc[-1]
    
    # Generate signal
    if lstm_pred > last_close * 1.0005:
        return 1
    elif lstm_pred < last_close * 0.9995:
        return -1
    else:
        return 0
    
    
def alpha_rf(symbol, timeframe, df):
    """
    Alpha using only the Random Forest model.
    """
    if len(df) < 6:  # need last 5 bars for RF
        return 0
    
    # Prepare input (last 5 closes)
    lag_features = df['close'].iloc[-6:-1].values.reshape(1, -1)
    rf_pred = rf_model.predict(lag_features)[0]
    
    last_close = df['close'].iloc[-1]
    
    # Generate signal
    if rf_pred > last_close * 1.0005:
        return 1
    elif rf_pred < last_close * 0.9995:
        return -1
    else:
        return 0

# ----------------- Update ALPHAS dict -----------------
# ALPHAS = {
#     "momentum": alpha_momentum,
#     "lstm": alpha_lstm,
#     "mean_revert": alpha_mean_revert,
#     "random_forest": alpha_rf,
#     "lstm_rf": alpha_pred_models
# }

