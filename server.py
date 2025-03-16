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

        # Process data
        results_by_date = {}

        for data in [daily_di, fourh_di, weekly_di]:
            for entry in data:
                date = entry["time"][:10]
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

                if entry in daily_di:
                    results_by_date[date]["daily_di"] = entry["DI_index"]
                elif entry in fourh_di:
                    results_by_date[date]["4h_di"] = entry["DI_index"]
                elif entry in weekly_di:
                    results_by_date[date]["weekly_di"] = entry["DI_index"]

        results_list = list(results_by_date.values())
        results_list.sort(key=lambda x: x["time"])

        # Fill Weekly DI Index gaps
        last_weekly = None
        for item in results_list:
            if item["weekly_di"] is None or math.isnan(item["weekly_di"]):
                item["weekly_di"] = last_weekly
            else:
                last_weekly = item["weekly_di"]

        # Calculate metrics
        df = pd.DataFrame(results_list)

        # Calculate Total DI with filled weekly values
        df["total_di"] = df.apply(
            lambda row: (
                sum(filter(None, [
                    row["weekly_di"],
                    row["daily_di"],
                    # Only include 4h_di if it exists
                    row["4h_di"] if pd.notna(row["4h_di"]) else None
                ]))
            ),
            axis=1
        )

        # Debug logs for calculations
        logger.debug("Sample of Total DI calculations:")
        logger.debug(df[["time", "weekly_di", "daily_di", "4h_di", "total_di"]].head())

        # Calculate indicators
        df["di_ema_13"] = ta.ema(df["total_di"], length=13)
        df["di_sma_30"] = df["total_di"].rolling(window=30, min_periods=30).mean()

        # Debug logs for indicators
        logger.debug("Sample of indicators:")
        logger.debug(df[["time", "total_di", "di_ema_13", "di_sma_30"]].head())

        # Calculate trend
        df["trend"] = np.where(
            (df["di_ema_13"].notna() & df["di_sma_30"].notna()),
            np.where(df["di_ema_13"] > df["di_sma_30"], "bull", "bear"),
            None
        )

        # Format results
        final_results = []
        for _, row in df.iterrows():
            entry = {
                "time": row["time"],
                "daily_di": row["daily_di"],
                "4h_di": row["4h_di"],
                "weekly_di": row["weekly_di"],
                "total_di": row["total_di"],
                "di_ema_13": row["di_ema_13"],
                "di_sma_30": row["di_sma_30"],
                "trend": row["trend"],
                "close": row["close"],
                "has_4h": pd.notna(row["4h_di"])  # Add flag for 4h data presence
            }
            # Convert NaN to None
            for key, value in entry.items():
                if isinstance(value, float) and math.isnan(value):
                    entry[key] = None
            final_results.append(entry)

        # Cache results
        set_cached_data(symbol, 'combined_indices', final_results)
        return symbol, final_results

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
    df = pd.DataFrame(data['Data']['Data'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
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
    df["change"] = df["close"].diff()
    df["direction"] = 0
    df.loc[df["change"] > 0, "direction"] = 1
    df.loc[df["change"] < 0, "direction"] = -1
    df["obv"] = (df["volumefrom"] * df["direction"]).fillna(0).cumsum()
    df["obv_ema"] = ta.ema(df["obv"], length=13)
    df["OBV_bullbear"] = (df["obv"] > df["obv_ema"]).astype(int)
    df["OBV_bias"] = (df["obv"] > 0).astype(int)
    df["OBV_index"] = df["OBV_bullbear"] + df["OBV_bias"]
    return df

def calculate_mfi_index(df):
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
    df["mfi_mf2"] = ta.ema(df["mfi_mf"], length=mfi_len)

    df["mfi_stupid_os"] = df["mfi_mf"].fillna(0).lt(20).astype(int)
    df["mfi_stupid_ob"] = df["mfi_mf"].fillna(0).gt(80).astype(int)
    df["mfi_bullbear"] = (df["mfi_mf"].fillna(0) > df["mfi_mf2"].fillna(0)).astype(int)
    df["mfi_bias"] = df["mfi_mf"].fillna(0).gt(50).astype(int)
    df["mfi_index"] = df["mfi_bullbear"] + df["mfi_bias"] + df["mfi_stupid_os"] - df["mfi_stupid_ob"]
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
    # Убедимся, что у нас есть столбец time и он доступен
    if 'time' not in df.columns and df.index.name == 'time':
        df = df.reset_index()

    # MA Index calculation (identical to Pine Script)
    df = calculate_ma_index(df)
    if debug:
        logger.debug("MA_index components:")
        logger.debug("MA_bull (micro > short):", df["micro"] > df["short"])
        logger.debug("MA_bull1 (short > medium):", df["short"] > df["medium"])
        logger.debug("MA_bull2 (short > long):", df["short"] > df["long"])
        logger.debug("MA_bull3 (medium > long):", df["medium"] > df["long"])
        logger.debug("Final MA_index:", df["MA_index"])

    # Willy Index
    df = calculate_willy_index(df)
    if debug:
        logger.debug("Willy_index components:")
        logger.debug("Willy_stupid_os (out2 < -80):", df["out2"] < -80)
        logger.debug("Willy_stupid_ob (out2 > -20):", df["out2"] > -20)
        logger.debug("Willy_bullbear (out > out2):", df["out"] > df["out2"])
        logger.debug("Willy_bias (out > -50):", df["out"] > -50)
        logger.debug("Final Willy_index:", df["Willy_index"])

    # MACD Index
    df = calculate_macd_index(df)
    if debug:
        logger.debug("MACD_index components:")
        logger.debug("macd_bullbear (macd > signal):", df["macd"] > df["signal"])
        logger.debug("macd_bias (macd > 0):", df["macd"] > 0)
        logger.debug("Final macd_index:", df["macd_index"])

    # OBV Index
    df = calculate_obv_index(df)
    if debug:
        logger.debug("OBV_index components:")
        logger.debug("OBV_bullbear (obv > obv_ema):", df["obv"] > df["obv_ema"])
        logger.debug("OBV_bias (obv > 0):", df["obv"] > 0)
        logger.debug("Final OBV_index:", df["OBV_index"])

    # MFI Index
    df = calculate_mfi_index(df)
    if debug:
        logger.debug("MFI_index components:")
        logger.debug("mfi_stupid_os (mfi_mf < 20):", df["mfi_mf"] < 20)
        logger.debug("mfi_stupid_ob (mfi_mf > 80):", df["mfi_mf"] > 80)
        logger.debug("mfi_bullbear (mfi_mf > mfi_mf2):", df["mfi_mf"] > df["mfi_mf2"])
        logger.debug("mfi_bias (mfi_mf > 50):", df["mfi_mf"] > 50)
        logger.debug("Final mfi_index:", df["mfi_index"])

    # AD Index
    df = calculate_ad_index(df)
    if debug:
        logger.debug("AD_index components:")
        logger.debug("AD_bullbear_short (ad > ad2):", df["ad"] > df["ad2"])
        logger.debug("AD_bullbear_med (ad > ad3):", df["ad"] > df["ad3"])
        logger.debug("AD_bullbear_long (ad2 > ad3):", df["ad2"] > df["ad3"])
        logger.debug("AD_bias (ad > 0):", df["ad"] > 0)
        logger.debug("AD_bias_long (ad3 > ad4):", df["ad3"] > df["ad4"])
        logger.debug("Final AD_index:", df["AD_index"])

    # Final DI calculation (identical to Pine Script)
    df["DI_index"] = (df["MA_index"] + df["Willy_index"] + df["macd_index"] +
                      df["OBV_index"] + df["mfi_index"] + df["AD_index"])

    # Calculate EMA and SMA
    df["DI_index_EMA"] = ta.ema(df["DI_index"], length=13)
    df["DI_index_SMA"] = df["DI_index"].rolling(window=30, min_periods=30).mean()
    df["weekly_DI_index"] = df["DI_index"].rolling(window=7, min_periods=7).mean()

    if debug:
        logger.debug("Final calculations:")
        logger.debug("DI_index =", df["DI_index"])
        logger.debug("DI_index_EMA =", df["DI_index_EMA"])
        logger.debug("DI_index_SMA =", df["DI_index_SMA"])
        logger.debug("weekly_DI_index =", df["weekly_DI_index"])

    def nan_to_none(val):
        if isinstance(val, float) and math.isnan(val):
            return None
        return val

    result = []
    for _, row in df.iterrows():
        time_val = row["time"] if "time" in row.index else row.name
        if isinstance(time_val, pd.Timestamp):
            time_str = time_val.strftime("%Y-%m-%d")
        else:
            time_str = str(time_val)

        result.append({
            "time": time_str,
            "DI_index": nan_to_none(row["DI_index"]),
            "DI_index_EMA": nan_to_none(row["DI_index_EMA"]),
            "DI_index_SMA": nan_to_none(row["DI_index_SMA"]),
            "weekly_DI_index": nan_to_none(row["weekly_DI_index"]),
            "close": nan_to_none(row["close"])
        })
    return result


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