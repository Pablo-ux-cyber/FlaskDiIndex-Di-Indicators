import math
import requests
import numpy as np
import logging
import pandas as pd
from flask import Blueprint, jsonify, request
from datetime import datetime
from utils.validation import validate_symbol, validate_params
from config import CRYPTOCOMPARE_API_KEY, TA_PARAMS

logger = logging.getLogger(__name__)

# Create blueprint for DI index routes
di_index_blueprint = Blueprint('di_index', __name__)

def ema(series, length):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=length, adjust=False).mean()

def sma(series, length):
    """Calculate Simple Moving Average"""
    return series.rolling(window=length).mean()

def calculate_ma_index(df):
    """Calculate Moving Average index"""
    ma_params = TA_PARAMS['MA']
    df["micro"] = ema(df["close"], length=ma_params['micro'])
    df["short"] = ema(df["close"], length=ma_params['short'])
    df["medium"] = sma(df["close"], length=ma_params['medium'])
    df["long"] = sma(df["close"], length=ma_params['long'])
    MA_bull = (df["micro"] > df["short"]).astype(int)
    MA_bull1 = (df["short"] > df["medium"]).astype(int)
    MA_bull2 = (df["short"] > df["long"]).astype(int)
    MA_bull3 = (df["medium"] > df["long"]).astype(int)
    df["MA_index"] = MA_bull + MA_bull1 + MA_bull2 + MA_bull3
    return df

def calculate_willy_index(df):
    """Calculate Willy index"""
    willy_params = TA_PARAMS['WILLY']
    df["upper"] = df["high"].rolling(window=willy_params['period'], min_periods=willy_params['period']).max()
    df["lower"] = df["low"].rolling(window=willy_params['period'], min_periods=willy_params['period']).min()
    df["range"] = df["upper"] - df["lower"]
    df["range"].replace(0, 1e-10, inplace=True)
    df["out"] = 100 * (df["close"] - df["upper"]) / df["range"]
    df["out2"] = ema(df["out"], length=willy_params['smooth'])
    df["Willy_stupid_os"] = (df["out2"] < -80).astype(int)
    df["Willy_stupid_ob"] = (df["out2"] > -20).astype(int)
    df["Willy_bullbear"] = (df["out"] > df["out2"]).astype(int)
    df["Willy_bias"] = (df["out"] > -50).astype(int)
    df["Willy_index"] = df["Willy_stupid_os"] + df["Willy_bullbear"] + df["Willy_bias"] - df["Willy_stupid_ob"]
    return df

def calculate_macd_index(df):
    """Calculate MACD index"""
    macd_params = TA_PARAMS['MACD']
    df["fastMA"] = ema(df["close"], length=macd_params['fast'])
    df["slowMA"] = ema(df["close"], length=macd_params['slow'])
    df["macd"] = df["fastMA"] - df["slowMA"]
    df["signal"] = sma(df["macd"], length=macd_params['signal'])
    df["macd_bullbear"] = (df["macd"] > df["signal"]).astype(int)
    df["macd_bias"] = (df["macd"] > 0).astype(int)
    df["macd_index"] = df["macd_bullbear"] + df["macd_bias"]
    return df

def calculate_obv_index(df):
    """Calculate OBV index"""
    df["change"] = df["close"].diff()
    df["direction"] = 0
    df.loc[df["change"] > 0, "direction"] = 1
    df.loc[df["change"] < 0, "direction"] = -1
    df["obv"] = (df["volumefrom"] * df["direction"]).fillna(0).cumsum()
    df["obv_ema"] = ema(df["obv"], length=13)
    df["OBV_bullbear"] = (df["obv"] > df["obv_ema"]).astype(int)
    df["OBV_bias"] = (df["obv"] > 0).astype(int)
    df["OBV_index"] = df["OBV_bullbear"] + df["OBV_bias"]
    return df

def calculate_mfi_index(df):
    """Calculate MFI index"""
    mfi_length = 14
    mfi_len = 13
    df["mfi_src"] = (df["high"] + df["low"] + df["close"]) / 3
    df["mfi_change"] = df["mfi_src"].diff()

    def safe_mfi_upper(row):
        if pd.isna(row["mfi_change"]):
            return np.nan
        return row["volumefrom"] * row["mfi_src"] if row["mfi_change"] > 0 else 0

    def safe_mfi_lower(row):
        if pd.isna(row["mfi_change"]):
            return np.nan
        return row["volumefrom"] * row["mfi_src"] if row["mfi_change"] < 0 else 0

    df["mfi_upper_calc"] = df.apply(safe_mfi_upper, axis=1)
    df["mfi_lower_calc"] = df.apply(safe_mfi_lower, axis=1)

    df["mfi_upper_sum"] = df["mfi_upper_calc"].rolling(window=mfi_length, min_periods=mfi_length).sum()
    df["mfi_lower_sum"] = df["mfi_lower_calc"].rolling(window=mfi_length, min_periods=mfi_length).sum()
    df["mfi_lower_sum"].replace(0, np.nan, inplace=True)

    df["mfi_ratio"] = df["mfi_upper_sum"] / df["mfi_lower_sum"]
    df["mfi_mf"] = 100 - (100 / (1 + df["mfi_ratio"]))
    df["mfi_mf2"] = ema(df["mfi_mf"], length=mfi_len)

    df["mfi_stupid_os"] = df["mfi_mf"].fillna(0).lt(20).astype(int)
    df["mfi_stupid_ob"] = df["mfi_mf"].fillna(0).gt(80).astype(int)
    df["mfi_bullbear"] = (df["mfi_mf"].fillna(0) > df["mfi_mf2"].fillna(0)).astype(int)
    df["mfi_bias"] = df["mfi_mf"].fillna(0).gt(50).astype(int)
    df["mfi_index"] = df["mfi_bullbear"] + df["mfi_bias"] + df["mfi_stupid_os"] - df["mfi_stupid_ob"]
    return df

def calculate_ad_index(df):
    """Calculate AD index"""
    condition = ((df["close"] == df["high"]) & (df["close"] == df["low"])) | (df["high"] == df["low"])
    df["ad_calc"] = ((2 * df["close"] - df["low"] - df["high"]) / (df["high"] - df["low"])).where(~condition, 0) * df["volumefrom"]
    df["ad"] = df["ad_calc"].cumsum()
    ad2 = ema(df["ad"], length=13)
    ad3 = df["ad"].rolling(window=30, min_periods=30).mean()
    ad4 = df["ad"].rolling(window=200, min_periods=200).mean()
    df["AD_bullbear_short"] = (df["ad"] > ad2).astype(int)
    df["AD_bullbear_med"] = (df["ad"] > ad3).astype(int)
    df["AD_bullbear_long"] = (ad2 > ad3).astype(int)
    df["AD_bias"] = (df["ad"] > 0).astype(int)
    df["AD_bias_long"] = (ad3 > ad4).astype(int)
    df["AD_index"] = df["AD_bullbear_short"] + df["AD_bullbear_med"] + df["AD_bullbear_long"] + df["AD_bias"] + df["AD_bias_long"]
    return df

def handle_nan_value(value):
    """Helper function to handle NaN values consistently"""
    return None if pd.isna(value) or (isinstance(value, float) and math.isnan(value)) else value

def get_daily_data(symbol="BTC", tsym="USD", limit=2000):
    """Get daily OHLCV data for given cryptocurrency"""
    url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym={tsym}&limit={limit}&api_key={CRYPTOCOMPARE_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get("Response") != "Success":
            logger.error(f"API Error: {data.get('Message')}")
            raise Exception(f"Error getting daily data: {data}")
        df = pd.DataFrame(data['Data']['Data'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        raise

def get_4h_data(symbol="BTC", tsym="USD", limit=2000):
    """Get 4-hour OHLCV data for given cryptocurrency"""
    url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={symbol}&tsym={tsym}&limit={limit}&aggregate=4&api_key={CRYPTOCOMPARE_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get("Response") != "Success":
            logger.error(f"Error getting 4-hour data: {data}")
            raise Exception(f"Error getting 4-hour data: {data}")
        df = pd.DataFrame(data['Data']['Data'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        raise

def get_weekly_data(symbol="BTC", tsym="USD", limit=2000):
    """Get weekly OHLCV data for given cryptocurrency"""
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

def calculate_di_index(df, debug=False):
    """Calculate DI index from multiple technical indicators"""
    df = calculate_ma_index(df)
    if debug:
        logger.debug("MA_index calculated")

    df = calculate_willy_index(df)
    if debug:
        logger.debug("Willy_index calculated")

    df = calculate_macd_index(df)
    if debug:
        logger.debug("MACD_index calculated")

    df = calculate_obv_index(df)
    if debug:
        logger.debug("OBV_index calculated")

    df = calculate_mfi_index(df)
    if debug:
        logger.debug("MFI_index calculated")

    df = calculate_ad_index(df)
    if debug:
        logger.debug("AD_index calculated")

    df["DI_index"] = (df["MA_index"] + df["Willy_index"] + df["macd_index"] +
                      df["OBV_index"] + df["mfi_index"] + df["AD_index"])
    df["DI_index_EMA"] = ema(df["DI_index"], length=13)
    df["DI_index_SMA"] = df["DI_index"].rolling(window=30, min_periods=30).mean()
    df["weekly_DI_index"] = df["DI_index"].rolling(window=7, min_periods=7).mean()

    result = []
    for _, row in df.iterrows():
        result.append({
            "time": row["time"].strftime("%Y-%m-%d") if isinstance(row["time"], pd.Timestamp) else str(row["time"]),
            "DI_index": handle_nan_value(row["DI_index"]),
            "DI_index_EMA": handle_nan_value(row["DI_index_EMA"]),
            "DI_index_SMA": handle_nan_value(row["DI_index_SMA"]),
            "weekly_DI_index": handle_nan_value(row["weekly_DI_index"]),
            "close": handle_nan_value(row["close"])
        })
    return result

def calculate_combined_indices(symbol="BTC", debug=False):
    """Calculate and combine indices from different timeframes"""
    try:
        # Get data for all timeframes
        df_daily = get_daily_data(symbol=symbol)
        df_4h = get_4h_data(symbol=symbol)
        df_weekly = get_weekly_data(symbol=symbol)

        # Calculate DI Index for each timeframe
        daily_di = calculate_di_index(df_daily, debug)
        fourh_di = calculate_di_index(df_4h, debug)
        weekly_di = calculate_di_index(df_weekly, debug)

        # Create date-indexed dictionaries
        results_by_date = {}

        # Process all timeframes
        for data in [daily_di, fourh_di, weekly_di]:
            for entry in data:
                date = entry["time"][:10]  # Get YYYY-MM-DD part
                if date not in results_by_date:
                    results_by_date[date] = {
                        "time": date,
                        "daily_di": None,
                        "4h_di": None,
                        "weekly_di": None,
                        "total_di": None,
                        "di_ema_13": None,
                        "di_sma_30": None,
                        "trend": None,
                        "close": entry["close"]
                    }

                # Determine which timeframe this entry belongs to
                if entry in daily_di:
                    results_by_date[date]["daily_di"] = entry["DI_index"]
                elif entry in fourh_di:
                    results_by_date[date]["4h_di"] = entry["DI_index"]
                elif entry in weekly_di:
                    results_by_date[date]["weekly_di"] = entry["DI_index"]

        # Convert to list and sort by date
        results_list = list(results_by_date.values())
        results_list.sort(key=lambda x: x["time"])

        # Calculate additional metrics
        df = pd.DataFrame(results_list)

        # Calculate Total DI
        df["total_di"] = df.apply(
            lambda row: (
                sum(filter(None, [row["daily_di"], row["4h_di"], row["weekly_di"]]))
                if any(filter(None, [row["daily_di"], row["4h_di"], row["weekly_di"]]))
                else None
            ),
            axis=1
        )

        # Calculate EMA and SMA on total_di
        df["di_ema_13"] = ema(df["total_di"], length=13)
        df["di_sma_30"] = df["total_di"].rolling(window=30, min_periods=30).mean()

        # Calculate trend
        df["trend"] = np.where(
            (df["di_ema_13"].notna() & df["di_sma_30"].notna()),
            np.where(df["di_ema_13"] > df["di_sma_30"], "bull", "bear"),
            None
        )

        # Convert back to dictionary format
        final_results = []
        for _, row in df.iterrows():
            entry = {
                "time": row["time"],
                "daily_di": handle_nan_value(row["daily_di"]),
                "4h_di": handle_nan_value(row["4h_di"]),
                "weekly_di": handle_nan_value(row["weekly_di"]),
                "total_di": handle_nan_value(row["total_di"]),
                "di_ema_13": handle_nan_value(row["di_ema_13"]),
                "di_sma_30": handle_nan_value(row["di_sma_30"]),
                "trend": row["trend"],
                "close": handle_nan_value(row["close"])
            }
            final_results.append(entry)

        return final_results

    except Exception as e:
        logger.error(f"Error in calculate_combined_indices: {str(e)}")
        return []

@di_index_blueprint.route('/')
def index():
    """Root endpoint showing API status"""
    return "DI Index API is running! Use /api/di_index endpoint with optional parameters: symbol, debug"

@di_index_blueprint.route('/api/di_index')
def di_index():
    """Main endpoint for getting DI index data"""
    try:
        symbol = request.args.get("symbol", "BTC").upper()
        debug_mode = request.args.get("debug", "false").lower() == "true"

        # Validate cryptocurrency symbol
        if not validate_symbol(symbol):
            return jsonify({"error": f"Invalid cryptocurrency symbol: {symbol}"}), 400

        # Calculate combined indices
        results = calculate_combined_indices(symbol=symbol, debug=debug_mode)

        return jsonify({
            "symbol": symbol,
            "data": results
        })
    except Exception as e:
        logger.error(f"Error in di_index endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500