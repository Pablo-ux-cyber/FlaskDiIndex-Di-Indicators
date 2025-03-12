import math
import requests
import numpy as np
import logging
import pandas as pd
from flask import Blueprint, jsonify, request, render_template
from datetime import datetime
from utils.validation import validate_symbol
from config import CRYPTOCOMPARE_API_KEY

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

def get_4h_data(symbol="BTC", tsym="USD", limit=2000):
    """Get 4-hour OHLCV data"""
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={symbol}&tsym={tsym}&limit={limit}&aggregate=4&api_key={CRYPTOCOMPARE_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data.get("Response") != "Success":
            logger.error(f"API Error: {data.get('Message')}")
            raise Exception(f"Error getting 4h data: {data}")

        df = pd.DataFrame(data['Data']['Data'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    except Exception as e:
        logger.error(f"Error in get_4h_data: {str(e)}")
        raise

def get_weekly_data(symbol="BTC", tsym="USD", limit=2000):
    """Get weekly OHLCV data"""
    try:
        df_daily = get_daily_data(symbol, tsym, limit)
        df_daily.set_index('time', inplace=True)
        df_weekly = df_daily.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volumefrom': 'sum',
            'volumeto': 'sum'
        }).dropna()
        df_weekly.reset_index(inplace=True)
        return df_weekly
    except Exception as e:
        logger.error(f"Error in get_weekly_data: {str(e)}")
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
        return df
    except Exception as e:
        logger.error(f"Error in calculate_di_index: {str(e)}")
        raise

def calculate_combined_indices(symbol="BTC"):
    """Calculate and combine indices from different timeframes"""
    try:
        # Get data for all timeframes
        df_daily = get_daily_data(symbol=symbol)
        df_4h = get_4h_data(symbol=symbol)
        df_weekly = get_weekly_data(symbol=symbol)

        # Calculate DI Index for each timeframe
        df_daily = calculate_di_index(df_daily)
        df_4h = calculate_di_index(df_4h)
        df_weekly = calculate_di_index(df_weekly)

        # Create date-indexed dictionaries
        results_by_date = {}

        # Process daily data
        for _, row in df_daily.iterrows():
            date = row["time"].strftime("%Y-%m-%d")
            results_by_date[date] = {
                "time": date,
                "daily_di": row["DI_index"],
                "4h_di": None,
                "weekly_di": None,
                "close": row["close"]
            }

        # Process 4h data
        for _, row in df_4h.iterrows():
            date = row["time"].strftime("%Y-%m-%d")
            if date in results_by_date:
                results_by_date[date]["4h_di"] = row["DI_index"]

        # Process weekly data
        for _, row in df_weekly.iterrows():
            date = row["time"].strftime("%Y-%m-%d")
            if date in results_by_date:
                results_by_date[date]["weekly_di"] = row["DI_index"]

        # Convert to DataFrame for calculations
        df = pd.DataFrame(results_by_date.values())

        # Calculate Total DI using mean
        df["total_di"] = df[["daily_di", "4h_di", "weekly_di"]].mean(axis=1)

        # Calculate trend indicators
        df["di_ema_13"] = ema(df["total_di"], 13)
        df["di_sma_30"] = df["total_di"].rolling(window=30).mean()

        # Calculate trend
        df["trend"] = np.where(
            (df["di_ema_13"].notna() & df["di_sma_30"].notna()),
            np.where(df["di_ema_13"] > df["di_sma_30"], "bull", "bear"),
            None
        )

        # Convert NaN to None for JSON serialization
        def clean_value(val):
            if pd.isna(val) or (isinstance(val, float) and math.isnan(val)):
                return None
            return float(val) if isinstance(val, (float, np.float64)) else val

        # Convert to list of dictionaries
        result = []
        for _, row in df.iterrows():
            entry = {
                "time": row["time"],
                "weekly_di": clean_value(row["weekly_di"]),
                "daily_di": clean_value(row["daily_di"]),
                "4h_di": clean_value(row["4h_di"]),
                "total_di": clean_value(row["total_di"]),
                "di_ema_13": clean_value(row["di_ema_13"]),
                "di_sma_30": clean_value(row["di_sma_30"]),
                "trend": row["trend"],
                "close": clean_value(row["close"])
            }
            result.append(entry)

        return sorted(result, key=lambda x: x["time"])

    except Exception as e:
        logger.error(f"Error in calculate_combined_indices: {str(e)}")
        return []

@di_index_blueprint.route('/')
def index():
    """Root endpoint serving the HTML page"""
    return render_template('index.html')

@di_index_blueprint.route('/api/di_index')
def di_index():
    """API endpoint for getting DI index data"""
    try:
        symbol = request.args.get("symbol", "BTC").upper()

        if not validate_symbol(symbol):
            return jsonify({"error": f"Invalid cryptocurrency symbol: {symbol}"}), 400

        results = calculate_combined_indices(symbol=symbol)

        return jsonify({
            "symbol": symbol,
            "data": results
        })
    except Exception as e:
        logger.error(f"Error in di_index endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500