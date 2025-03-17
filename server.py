import math
import requests
import numpy as np
from datetime import datetime, timedelta
import time
from functools import lru_cache
import logging
import concurrent.futures

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Monkey-patch for numpy's NaN compatibility with pandas-ta
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas as pd
import pandas_ta as ta
from flask import Blueprint, jsonify, request, render_template
import os

# Create blueprint for DI index routes
di_index_blueprint = Blueprint('di_index', __name__)

# Get API key from environment variable with fallback
API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY", "2193d3ce789e90e474570058a3a96caa0d585ca0d0d0e62687a295c8402d29e9")

# Cache for storing cryptocurrency data
CACHE = {}
CACHE_DURATION = 7200  # 2 hours in seconds
MAX_WORKERS = 5  # Increase maximum number of concurrent workers

def get_cache_key(symbol, data_type):
    """Generate cache key for given symbol and data type"""
    return f"{symbol}_{data_type}"

def is_cache_valid(cache_time):
    """Check if cached data is still valid"""
    return time.time() - cache_time < CACHE_DURATION

def get_cached_data(symbol, data_type):
    """Get data from cache if valid"""
    cache_key = get_cache_key(symbol, data_type)
    cached = CACHE.get(cache_key)
    if cached and is_cache_valid(cached['time']):
        return cached['data']
    return None

def set_cached_data(symbol, data_type, data):
    """Store data in cache"""
    cache_key = get_cache_key(symbol, data_type)
    CACHE[cache_key] = {
        'data': data,
        'time': time.time()
    }

def process_symbol(symbol, debug=False):
    """Process a single symbol"""
    try:
        logger.debug(f"Processing symbol: {symbol}")

        # Check cache first
        cached_result = get_cached_data(symbol, 'combined_indices')
        if cached_result is not None:
            logger.debug(f"Cache hit for {symbol}")
            return symbol, cached_result

        # Get data for all timeframes with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df_daily = get_daily_data(symbol=symbol)
                df_4h = get_4h_data(symbol=symbol)
                df_weekly = get_weekly_data(symbol=symbol)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)  # Wait before retry

        # Calculate indices
        daily_di = calculate_di_index(df_daily, debug)
        fourh_di = calculate_di_index(df_4h, debug)
        weekly_di = calculate_di_index(df_weekly, debug)

        # Process data - UPDATED to handle new structure
        results_by_date = {}

        # First, process daily and weekly data
        for data_type, data_list in zip(["daily", "weekly"], [daily_di, weekly_di]):
            for entry in data_list:
                date = entry["time"][:10]  # Get just the date part
                if date not in results_by_date:
                    results_by_date[date] = {
                        "time": date,
                        "daily_di_old": None,
                        "daily_di_new": None,
                        "weekly_di_old": None,
                        "weekly_di_new": None,
                        "4h_values_old": [],  # List to store all 4h values
                        "4h_values_new": [],  # List to store all 4h values
                        "4h_di_old": None,    # Latest 4h value
                        "4h_di_new": None,    # Latest 4h value
                        "DI_index_old": None,
                        "DI_index_new": None,
                        "di_ema_13_old": None,
                        "di_ema_13_new": None,
                        "di_sma_30_old": None,
                        "di_sma_30_new": None,
                        "trend_old": None,
                        "trend_new": None,
                        "close": entry["close"]
                    }

                # Update daily/weekly values
                if data_type == "daily":
                    results_by_date[date].update({
                        "daily_di_old": entry["daily_di_old"],
                        "daily_di_new": entry["daily_di_new"],
                        "DI_index_old": entry["DI_index_old"],
                        "DI_index_new": entry["DI_index_new"],
                        "di_ema_13_old": entry["di_ema_13_old"],
                        "di_ema_13_new": entry["di_ema_13_new"],
                        "di_sma_30_old": entry["di_sma_30_old"],
                        "di_sma_30_new": entry["di_sma_30_new"],
                        "trend_old": entry["trend_old"],
                        "trend_new": entry["trend_new"]
                    })
                elif data_type == "weekly":
                    results_by_date[date].update({
                        "weekly_di_old": entry["weekly_di_old"],
                        "weekly_di_new": entry["weekly_di_new"]
                    })

        # Then process 4h data
        for entry in fourh_di:
            date = entry["time"][:10]
            if date in results_by_date:
                # Store the full 4h value
                results_by_date[date]["4h_values_old"].append({
                    "time": entry["time"],
                    "value": entry["4h_di_old"]
                })
                results_by_date[date]["4h_values_new"].append({
                    "time": entry["time"],
                    "value": entry["4h_di_new"]
                })

                # Update the latest 4h value for the day
                results_by_date[date]["4h_di_old"] = entry["4h_di_old"]
                results_by_date[date]["4h_di_new"] = entry["4h_di_new"]

        # Sort 4h values by time for each day
        for date_data in results_by_date.values():
            date_data["4h_values_old"].sort(key=lambda x: x["time"])
            date_data["4h_values_new"].sort(key=lambda x: x["time"])

        results_list = list(results_by_date.values())
        results_list.sort(key=lambda x: x["time"])

        # Cache results
        set_cached_data(symbol, 'combined_indices', results_list)
        return symbol, results_list

    except Exception as e:
        logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
        return symbol, {"error": str(e)}

def validate_symbol(symbol):
    """Validate if the cryptocurrency symbol exists on CryptoCompare"""
    url = f"https://min-api.cryptocompare.com/data/price?fsym={symbol}&tsyms=USD&api_key={API_KEY}"
    response = requests.get(url)
    data = response.json()
    if "Response" in data and data["Response"] == "Error":
        return False
    return True

def get_daily_data(symbol="BTC", tsym="USD", limit=2000):
    """Get daily OHLCV data for given cryptocurrency"""
    cached_data = get_cached_data(symbol, "daily_data")
    if cached_data is not None and not cached_data.empty:
        return cached_data

    url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym={tsym}&limit={limit}&api_key={API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data.get("Response") != "Success":
        raise Exception(f"Error getting daily data: {data}")

    # Convert timestamp to datetime and adjust to end of day
    df = pd.DataFrame(data['Data']['Data'])
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Отфильтровываем будущие даты и сегодняшний день, так как он еще не закончился
    today = pd.Timestamp.now().normalize()
    df = df[df['time'] < today]

    set_cached_data(symbol, "daily_data", df)
    return df

def get_4h_data(symbol="BTC", tsym="USD", limit=2000):
    """Get 4-hour OHLCV data for given cryptocurrency"""
    cached_data = get_cached_data(symbol, "4h_data")
    if cached_data is not None and not cached_data.empty:
        return cached_data

    url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={symbol}&tsym={tsym}&limit={limit}&aggregate=4&api_key={API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data.get("Response") != "Success":
        raise Exception(f"Error getting 4-hour data: {data}")

    # Convert timestamp to datetime with timezone info
    df = pd.DataFrame(data['Data']['Data'])
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Group by date to ensure we have all 4h intervals
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour

    # Log the data distribution
    logger.debug(f"4h data distribution for {symbol}:")
    logger.debug(df.groupby(['date', 'hour']).size().reset_index(name='count'))

    set_cached_data(symbol, "4h_data", df)
    return df

def get_weekly_data(symbol="BTC", tsym="USD", limit=2000):
    """Get weekly OHLCV data for given cryptocurrency"""
    cached_data = get_cached_data(symbol, "weekly_data")
    if cached_data is not None and not cached_data.empty:
        return cached_data

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
    set_cached_data(symbol, "weekly_data", df_weekly)
    return df_weekly

def calculate_ma_index(df):
    df["micro"] = ta.ema(df["close"], length=6)
    df["short"] = ta.ema(df["close"], length=13)
    df["medium"] = ta.sma(df["close"], length=30)
    df["long"] = ta.sma(df["close"], length=200)

    MA_bull = (df["micro"] > df["short"]).astype(int)
    MA_bull1 = (df["short"] > df["medium"]).astype(int)
    MA_bull2 = (df["short"] > df["long"]).astype(int)
    MA_bull3 = (df["medium"] > df["long"]).astype(int)
    df["MA_index"] = MA_bull + MA_bull1 + MA_bull2 + MA_bull3
    return df

def calculate_willy_index(df):
    df["upper"] = df["high"].rolling(window=21, min_periods=21).max()
    df["lower"] = df["low"].rolling(window=21, min_periods=21).min()
    df["range"] = df["upper"] - df["lower"]
    df["range"].replace(0, 1e-10, inplace=True)
    df["out"] = 100 * (df["close"] - df["upper"]) / df["range"]
    df["out2"] = ta.ema(df["out"], length=13)
    df["Willy_stupid_os"] = (df["out2"] < -80).astype(int)
    df["Willy_stupid_ob"] = (df["out2"] > -20).astype(int)
    df["Willy_bullbear"] = (df["out"] > df["out2"]).astype(int)
    df["Willy_bias"] = (df["out"] > -50).astype(int)
    df["Willy_index"] = df["Willy_stupid_os"] + df["Willy_bullbear"] + df["Willy_bias"] - df["Willy_stupid_ob"]
    return df

def calculate_macd_index(df):
    df["fastMA"] = ta.ema(df["close"], length=12)
    df["slowMA"] = ta.ema(df["close"], length=26)
    df["macd"] = df["fastMA"] - df["slowMA"]
    df["signal"] = ta.sma(df["macd"], length=9)
    df["macd_bullbear"] = (df["macd"] > df["signal"]).astype(int)
    df["macd_bias"] = (df["macd"] > 0).astype(int)
    df["macd_index"] = df["macd_bullbear"] + df["macd_bias"]
    return df

def calculate_obv_index(df):
    """Calculate OBV Index using both old and new methods"""
    # Old method
    df["change"] = df["close"].diff()
    df["direction"] = 0
    df.loc[df["change"] > 0, "direction"] = 1
    df.loc[df["change"] < 0, "direction"] = -1
    df["obv_old"] = (df["volumefrom"] * df["direction"]).fillna(0).cumsum()
    df["obv_ema_old"] = ta.ema(df["obv_old"], length=13)
    df["OBV_bullbear_old"] = (df["obv_old"] > df["obv_ema_old"]).astype(int)
    df["OBV_bias_old"] = (df["obv_old"] > 0).astype(int)
    df["OBV_index_old"] = df["OBV_bullbear_old"] + df["OBV_bias_old"]

    # New method (Pine Script style)
    df["obv_new"] = df.apply(
        lambda row: row["volumefrom"] if row["change"] > 0 else (-row["volumefrom"] if row["change"] < 0 else 0),
        axis=1
    ).fillna(0).cumsum()
    df["obv_ema_new"] = ta.ema(df["obv_new"], length=13)
    df["OBV_bullbear_new"] = (df["obv_new"] > df["obv_ema_new"]).astype(int)
    df["OBV_bias_new"] = (df["obv_new"] > 0).astype(int)
    df["OBV_index_new"] = df["OBV_bullbear_new"] + df["OBV_bias_new"]

    # Use new method for final calculations
    df["OBV_index"] = df["OBV_index_new"]
    return df

def calculate_mfi_index(df):
    """Calculate MFI Index using both old and new methods"""
    mfi_length = 14
    mfi_len = 13
    df["mfi_src"] = (df["high"] + df["low"] + df["close"]) / 3
    df["mfi_change"] = df["mfi_src"].diff()

    # Old method
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
    df["mfi_mf_old"] = 100 - (100 / (1 + df["mfi_ratio"]))
    df["mfi_mf2_old"] = ta.ema(df["mfi_mf_old"], length=mfi_len)

    # Old method components
    df["mfi_stupid_os_old"] = df["mfi_mf_old"].fillna(0).lt(20).astype(int)
    df["mfi_stupid_ob_old"] = df["mfi_mf_old"].fillna(0).gt(80).astype(int)
    df["mfi_bullbear_old"] = (df["mfi_mf_old"].fillna(0) > df["mfi_mf2_old"].fillna(0)).astype(int)
    df["mfi_bias_old"] = df["mfi_mf_old"].fillna(0).gt(50).astype(int)
    df["mfi_index_old"] = df["mfi_bullbear_old"] + df["mfi_bias_old"] + df["mfi_stupid_os_old"] - df["mfi_stupid_ob_old"]

    # New method (Pine Script style using RSI)
    # Fix: calculate RSI correctly on the money flow ratio
    df["mfi_mf_new"] = ta.rsi(close=df["mfi_upper_sum"] / df["mfi_lower_sum"], length=mfi_length)
    df["mfi_mf2_new"] = ta.ema(df["mfi_mf_new"], length=mfi_len)

    # New method components
    df["mfi_stupid_os_new"] = df["mfi_mf_new"].fillna(0).lt(20).astype(int)
    df["mfi_stupid_ob_new"] = df["mfi_mf_new"].fillna(0).gt(80).astype(int)
    df["mfi_bullbear_new"] = (df["mfi_mf_new"].fillna(0) > df["mfi_mf2_new"].fillna(0)).astype(int)
    df["mfi_bias_new"] = df["mfi_mf_new"].fillna(0).gt(50).astype(int)
    df["mfi_index_new"] = df["mfi_bullbear_new"] + df["mfi_bias_new"] + df["mfi_stupid_os_new"] - df["mfi_stupid_ob_new"]

    # Use new method for final calculations
    df["mfi_index"] = df["mfi_index_new"]
    return df

def calculate_ad_index(df):
    condition = ((df["close"] == df["high"]) & (df["close"] == df["low"])) | (df["high"] == df["low"])
    df["ad_calc"] = ((2 * df["close"] - df["low"] - df["high"]) / (df["high"] - df["low"])).where(~condition, 0) * df["volumefrom"]
    df["ad"] = df["ad_calc"].cumsum()
    ad2 = ta.ema(df["ad"], length=13)
    ad3 = df["ad"].rolling(window=30, min_periods=30).mean()
    ad4 = df["ad"].rolling(window=200, min_periods=200).mean()
    df["AD_bullbear_short"] = (df["ad"] > ad2).astype(int)
    df["AD_bullbear_med"] = (df["ad"] > ad3).astype(int)
    df["AD_bullbear_long"] = (ad2 > ad3).astype(int)
    df["AD_bias"] = (df["ad"] > 0).astype(int)
    df["AD_bias_long"] = (ad3 > ad4).astype(int)
    df["AD_index"] = df["AD_bullbear_short"] + df["AD_bullbear_med"] + df["AD_bullbear_long"] + df["AD_bias"] + df["AD_bias_long"]
    return df

def calculate_di_index(df, debug=False):
    """Calculate DI index components and final value"""
    if 'time' not in df.columns and df.index.name == 'time':
        df = df.reset_index()

    # Calculate all components
    df = calculate_ma_index(df)
    df = calculate_willy_index(df)
    df = calculate_macd_index(df)
    df = calculate_obv_index(df)
    df = calculate_mfi_index(df)
    df = calculate_ad_index(df)

    # Calculate old and new DI indices
    df["DI_index_old"] = (df["MA_index"] + df["Willy_index"] + df["macd_index"] +
                         df["OBV_index_old"] + df["mfi_index_old"] + df["AD_index"])

    df["DI_index_new"] = (df["MA_index"] + df["Willy_index"] + df["macd_index"] +
                         df["OBV_index_new"] + df["mfi_index_new"] + df["AD_index"])

    # Calculate EMAs and SMAs for both methods
    df["di_ema_13_old"] = ta.ema(df["DI_index_old"], length=13)
    df["di_sma_30_old"] = df["DI_index_old"].rolling(window=30, min_periods=30).mean()
    df["trend_old"] = np.where(
        (df["di_ema_13_old"].notna() & df["di_sma_30_old"].notna()),
        np.where(df["di_ema_13_old"] > df["di_sma_30_old"], "bull", "bear"),
        None
    )

    df["di_ema_13_new"] = ta.ema(df["DI_index_new"], length=13)
    df["di_sma_30_new"] = df["DI_index_new"].rolling(window=30, min_periods=30).mean()
    df["trend_new"] = np.where(
        (df["di_ema_13_new"].notna() & df["di_sma_30_new"].notna()),
        np.where(df["di_ema_13_new"] > df["di_sma_30_new"], "bull", "bear"),
        None
    )

    # Calculate Weekly, Daily, and 4h DI for both methods
    if len(df) >= 7:  # For weekly calculation
        df["weekly_di_old"] = df["DI_index_old"].rolling(window=7, min_periods=7).mean()
        df["weekly_di_new"] = df["DI_index_new"].rolling(window=7, min_periods=7).mean()
    else:
        df["weekly_di_old"] = df["DI_index_old"]
        df["weekly_di_new"] = df["DI_index_new"]

    df["daily_di_old"] = df["DI_index_old"]
    df["daily_di_new"] = df["DI_index_new"]

    # 4h DI is just the current DI value for that timeframe
    df["4h_di_old"] = df["DI_index_old"]
    df["4h_di_new"] = df["DI_index_new"]

    if debug:
        logger.debug("Sample of 4h values:")
        logger.debug(df[["time", "4h_di_old", "4h_di_new"]].head())

    def format_time(dt):
        return dt.strftime("%Y-%m-%d %H:%M:%S") if isinstance(dt, pd.Timestamp) else str(dt)

    result = []
    for _, row in df.iterrows():
        time_val = row["time"] if "time" in row.index else row.name
        time_str = format_time(time_val)

        result.append({
            "time": time_str,
            "weekly_di_old": nan_to_none(row["weekly_di_old"]),
            "weekly_di_new": nan_to_none(row["weekly_di_new"]),
            "daily_di_old": nan_to_none(row["daily_di_old"]),
            "daily_di_new": nan_to_none(row["daily_di_new"]),
            "4h_di_old": nan_to_none(row["4h_di_old"]),
            "4h_di_new": nan_to_none(row["4h_di_new"]),
            "DI_index_old": nan_to_none(row["DI_index_old"]),
            "DI_index_new": nan_to_none(row["DI_index_new"]),
            "di_ema_13_old": nan_to_none(row["di_ema_13_old"]),
            "di_ema_13_new": nan_to_none(row["di_ema_13_new"]),
            "di_sma_30_old": nan_to_none(row["di_sma_30_old"]),
            "di_sma_30_new": nan_to_none(row["di_sma_30_new"]),
            "trend_old": row["trend_old"],
            "trend_new": row["trend_new"],
            "close": nan_to_none(row["close"])
        })
    return result

def nan_to_none(val):
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def process_symbol_batch(symbols, debug=False):
    """Process a batch of symbols efficiently"""
    results = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Добавляем задержку между запросами к API
            def process_with_delay(symbol):
                try:
                    time.sleep(0.5)  # 500ms задержка между запросами
                    return process_symbol(symbol, debug)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
                    return symbol, {"error": str(e)}

            future_to_symbol = {
                executor.submit(process_with_delay, symbol): symbol
                for symbol in symbols
            }

            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    symbol, result = future.result()
                    # Сортируем результаты по дате перед отправкой
                    if isinstance(result, list):
                        result.sort(key=lambda x: x["time"], reverse=True)
                    results[symbol] = result
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
                    results[symbol] = {"error": str(e)}

    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}", exc_info=True)
        for symbol in symbols:
            results[symbol] = {"error": f"Batch processing error: {str(e)}"}

    return results

@di_index_blueprint.route('/api/di_index')
def di_index():
    try:
        symbols = request.args.get("symbols", "BTC")
        debug_mode = request.args.get("debug", "false").lower() == "true"

        logger.debug(f"Received request for symbols: {symbols}")
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        logger.debug(f"Parsed symbol list: {symbol_list}")

        if not symbol_list:
            return jsonify({"error": "No valid symbols provided"}), 400

        # Validate symbols first
        for symbol in symbol_list:
            if not validate_symbol(symbol):
                logger.error(f"Invalid cryptocurrency symbol: {symbol}")
                return jsonify({"error": f"Invalid cryptocurrency symbol: {symbol}"}), 400

        results = process_symbol_batch(symbol_list, debug_mode)
        logger.debug(f"Calculation completed, results keys: {list(results.keys())}")

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error in di_index endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@di_index_blueprint.route('/')
def index():
    return render_template('index.html')