import math
import requests
import numpy as np
import logging
import pandas as pd
from flask import Blueprint, jsonify, request
from datetime import datetime
from utils.validation import validate_symbol
from config import CRYPTOCOMPARE_API_KEY, TA_PARAMS

logger = logging.getLogger(__name__)

# Create blueprint for DI index routes
di_index_blueprint = Blueprint('di_index', __name__)

def ema(series, periods):
    """Calculate EMA"""
    return series.ewm(span=periods, adjust=False).mean()

def calculate_ma_index(df):
    """Calculate Moving Average index"""
    df["micro"] = ema(df["close"], 6)
    df["short"] = ema(df["close"], 13)
    df["medium"] = df["close"].rolling(window=30).mean()
    df["long"] = df["close"].rolling(window=200).mean()

    MA_bull = (df["micro"] > df["short"]).astype(int)
    MA_bull1 = (df["short"] > df["medium"]).astype(int)
    MA_bull2 = (df["short"] > df["long"]).astype(int)
    MA_bull3 = (df["medium"] > df["long"]).astype(int)
    df["MA_index"] = MA_bull + MA_bull1 + MA_bull2 + MA_bull3
    return df

def calculate_willy_index(df):
    """Calculate Willy index"""
    period = 21
    df["upper"] = df["high"].rolling(window=period).max()
    df["lower"] = df["low"].rolling(window=period).min()
    df["range"] = df["upper"] - df["lower"]
    df["range"].replace(0, 1e-10, inplace=True)
    df["out"] = 100 * (df["close"] - df["upper"]) / df["range"]
    df["out2"] = ema(df["out"], 13)
    df["Willy_index"] = ((df["out2"] < -80).astype(int) + 
                        (df["out"] > df["out2"]).astype(int) + 
                        (df["out"] > -50).astype(int) - 
                        (df["out2"] > -20).astype(int))
    return df

def calculate_macd_index(df):
    """Calculate MACD index"""
    df["fastMA"] = ema(df["close"], 12)
    df["slowMA"] = ema(df["close"], 26)
    df["macd"] = df["fastMA"] - df["slowMA"]
    df["signal"] = df["macd"].rolling(window=9).mean()
    df["macd_index"] = ((df["macd"] > df["signal"]).astype(int) + 
                       (df["macd"] > 0).astype(int))
    return df

def calculate_obv_index(df):
    """Calculate OBV index"""
    df["change"] = df["close"].diff()
    df["direction"] = np.where(df["change"] > 0, 1, np.where(df["change"] < 0, -1, 0))
    df["obv"] = (df["volumefrom"] * df["direction"]).fillna(0).cumsum()
    df["obv_ema"] = ema(df["obv"], 13)
    df["OBV_index"] = ((df["obv"] > df["obv_ema"]).astype(int) + 
                      (df["obv"] > 0).astype(int))
    return df

def get_daily_data(symbol="BTC", tsym="USD", limit=2000):
    """Get daily OHLCV data"""
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym={tsym}&limit={limit}&api_key={CRYPTOCOMPARE_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data.get("Response") != "Success":
            logger.error(f"API Error: {data.get('Message')}")
            raise Exception(f"Error getting daily data: {data}")

        df = pd.DataFrame(data['Data']['Data'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    except Exception as e:
        logger.error(f"Error in get_daily_data: {str(e)}")
        raise

def calculate_di_index(df):
    """Calculate DI index"""
    try:
        df = calculate_ma_index(df)
        df = calculate_willy_index(df)
        df = calculate_macd_index(df)
        df = calculate_obv_index(df)

        df["DI_index"] = (df["MA_index"] + df["Willy_index"] + 
                         df["macd_index"] + df["OBV_index"])

        df["DI_index_EMA"] = ema(df["DI_index"], 13)
        df["DI_index_SMA"] = df["DI_index"].rolling(window=30).mean()

        result = []
        for _, row in df.iterrows():
            result.append({
                "time": row["time"].strftime("%Y-%m-%d"),
                "DI_index": None if pd.isna(row["DI_index"]) else float(row["DI_index"]),
                "DI_index_EMA": None if pd.isna(row["DI_index_EMA"]) else float(row["DI_index_EMA"]),
                "DI_index_SMA": None if pd.isna(row["DI_index_SMA"]) else float(row["DI_index_SMA"]),
                "close": None if pd.isna(row["close"]) else float(row["close"])
            })
        return result
    except Exception as e:
        logger.error(f"Error in calculate_di_index: {str(e)}")
        return []

@di_index_blueprint.route('/')
def index():
    """Root endpoint"""
    return "DI Index API is running! Use /api/di_index endpoint with optional parameter: symbol"

@di_index_blueprint.route('/api/di_index')
def di_index():
    """Get DI index data"""
    try:
        symbol = request.args.get("symbol", "BTC").upper()

        if not validate_symbol(symbol):
            return jsonify({"error": f"Invalid cryptocurrency symbol: {symbol}"}), 400

        df = get_daily_data(symbol=symbol)
        results = calculate_di_index(df)

        return jsonify({
            "symbol": symbol,
            "data": results
        })
    except Exception as e:
        logger.error(f"Error in di_index endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500