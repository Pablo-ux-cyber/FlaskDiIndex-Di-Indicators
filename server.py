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
MAX_WORKERS = 5  # Maximum number of concurrent workers

def calculate_ma_index(df):
    """Calculate MA Index components exactly as in Pine Script"""
    # MAs
    df["micro"] = ta.ema(df["close"], length=6)
    df["short"] = ta.ema(df["close"], length=13)
    df["medium"] = ta.sma(df["close"], length=30)
    df["long"] = ta.sma(df["close"], length=200)

    # MA Bull conditions exactly as in Pine Script
    df["MA_bull"] = (df["micro"] > df["short"]).astype(int)
    df["MA_bull1"] = (df["short"] > df["medium"]).astype(int)
    df["MA_bull2"] = (df["short"] > df["long"]).astype(int)
    df["MA_bull3"] = (df["medium"] > df["long"]).astype(int)

    df["MA_index"] = df["MA_bull"] + df["MA_bull1"] + df["MA_bull2"] + df["MA_bull3"]

    # Test case logging for 01.01.2024
    test_date = pd.Timestamp('2024-01-01')
    if test_date in df.index:
        logger.debug("\nMA Index Test case values for 2024-01-01:")
        test_data = df.loc[test_date]
        logger.debug(f"MA conditions: bull={test_data['MA_bull']}, bull1={test_data['MA_bull1']}, bull2={test_data['MA_bull2']}, bull3={test_data['MA_bull3']}")
        logger.debug(f"MA_index final value: {test_data['MA_index']}")

    return df

def calculate_willy_index(df):
    """Calculate Willy Index components exactly as in Pine Script"""
    # Willy calculation
    length = 21  # As defined in Pine Script
    len_out = 13  # For EMA calculation of out

    # Highest and lowest calculations over length period
    df["upper"] = df["high"].rolling(window=length, min_periods=1).max()
    df["lower"] = df["low"].rolling(window=length, min_periods=1).min()
    df["range"] = df["upper"] - df["lower"]
    df["range"].replace(0, 1e-10, inplace=True)  # Avoid division by zero

    # Calculate Williams %R exactly as in Pine Script
    df["out"] = 100 * (df["close"] - df["upper"]) / df["range"]
    df["out2"] = ta.ema(df["out"], length=len_out)

    # Calculate components exactly as in Pine Script
    df["Willy_stupid_os"] = (df["out2"] < -80).astype(int)
    df["Willy_stupid_ob"] = (df["out2"] > -20).astype(int)
    df["Willy_bullbear"] = (df["out"] > df["out2"]).astype(int)
    df["Willy_bias"] = (df["out"] > -50).astype(int)

    df["Willy_index"] = df["Willy_stupid_os"] + df["Willy_bullbear"] + df["Willy_bias"] - df["Willy_stupid_ob"]

    # Test case logging for 01.01.2024
    test_date = pd.Timestamp('2024-01-01')
    if test_date in df.index:
        logger.debug("\nWilly Index Test case values for 2024-01-01:")
        test_data = df.loc[test_date]
        logger.debug(f"Willy components: os={test_data['Willy_stupid_os']}, ob={test_data['Willy_stupid_ob']}")
        logger.debug(f"Willy bullbear={test_data['Willy_bullbear']}, bias={test_data['Willy_bias']}")
        logger.debug(f"Willy_index final value: {test_data['Willy_index']}")

    return df

def calculate_macd_index(df):
    """Calculate MACD Index components exactly as in Pine Script"""
    # MACD parameters as defined in Pine Script
    fastLength = 12
    slowLength = 26
    signalLength = 9

    # MACD calculation exactly as in Pine Script
    df["fastMA"] = ta.ema(df["close"], length=fastLength)
    df["slowMA"] = ta.ema(df["close"], length=slowLength)
    df["macd"] = df["fastMA"] - df["slowMA"]
    df["signal"] = ta.sma(df["macd"], length=signalLength)

    # Calculate bull/bear conditions exactly as in Pine Script
    df["macd_bullbear"] = (df["macd"] > df["signal"]).astype(int)
    df["macd_bias"] = (df["macd"] > 0).astype(int)
    df["macd_index"] = df["macd_bullbear"] + df["macd_bias"]

    # Test case logging for 01.01.2024
    test_date = pd.Timestamp('2024-01-01')
    if test_date in df.index:
        logger.debug("\nMACD Index Test case values for 2024-01-01:")
        test_data = df.loc[test_date]
        logger.debug(f"MACD components: bullbear={test_data['macd_bullbear']}, bias={test_data['macd_bias']}")
        logger.debug(f"MACD_index final value: {test_data['macd_index']}")

    return df

def calculate_obv_index(df):
    """Calculate OBV Index components exactly as in Pine Script"""
    # OBV
    df["change"] = df["close"].diff()

    # Exact Pine Script logic implementation
    df["obv_volume"] = np.where(
        df["change"] > 0,
        df["volumefrom"],
        np.where(df["change"] < 0, -df["volumefrom"], 0)
    )

    df["obv_out"] = df["obv_volume"].cumsum()
    df["obv_out2"] = ta.ema(df["obv_out"], length=13)

    df["OBV_bullbear"] = (df["obv_out"] > df["obv_out2"]).astype(int)
    df["OBV_bias"] = (df["obv_out"] > 0).astype(int)
    df["OBV_index"] = df["OBV_bullbear"] + df["OBV_bias"]
    return df

def calculate_mfi_index(df):
    """Calculate MFI Index components exactly as in Pine Script"""
    # MFI parameters as defined in Pine Script
    mfi_length = 14

    # Source calculation - hlc3 (high, low, close average)
    df["mfi_src"] = (df["high"] + df["low"] + df["close"]) / 3
    df["mfi_change"] = df["mfi_src"].diff()

    # Money flow calculations exactly as in Pine Script
    df["mfi_upper"] = df["volumefrom"] * np.where(df["mfi_change"] > 0, df["mfi_src"], 0)
    df["mfi_lower"] = df["volumefrom"] * np.where(df["mfi_change"] < 0, df["mfi_src"], 0)

    # Calculate sums with minimum periods of 1 to match Pine Script behavior
    df["mfi_upper_sum"] = df["mfi_upper"].rolling(window=mfi_length, min_periods=1).sum()
    df["mfi_lower_sum"] = df["mfi_lower"].rolling(window=mfi_length, min_periods=1).sum()
    df["mfi_lower_sum"].replace(0, np.nan, inplace=True)

    # MFI calculation using RSI formula
    df["mfi_mf"] = ta.rsi(df["mfi_upper_sum"] / df["mfi_lower_sum"], length=mfi_length)
    df["mfi_mf2"] = ta.ema(df["mfi_mf"], length=13)

    df["mfi_stupid_os"] = (df["mfi_mf"] < 20).astype(int)
    df["mfi_stupid_ob"] = (df["mfi_mf"] > 80).astype(int)
    df["mfi_bullbear"] = (df["mfi_mf"] > df["mfi_mf2"]).astype(int)
    df["mfi_bias"] = (df["mfi_mf"] > 50).astype(int)

    df["mfi_index"] = df["mfi_bullbear"] + df["mfi_bias"] + df["mfi_stupid_os"] - df["mfi_stupid_ob"]
    return df

def calculate_ad_index(df):
    """Calculate AD Index components exactly as in Pine Script"""
    # AD
    condition = ((df["close"] == df["high"]) & (df["close"] == df["low"])) | (df["high"] == df["low"])
    df["ad_calc"] = ((2 * df["close"] - df["low"] - df["high"]) / (df["high"] - df["low"])).where(~condition, 0) * df["volumefrom"]
    df["ad"] = df["ad_calc"].cumsum()

    df["ad2"] = ta.ema(df["ad"], length=13)
    df["ad3"] = ta.sma(df["ad"], length=30)
    df["ad4"] = ta.sma(df["ad"], length=200)

    df["AD_bullbear_short"] = (df["ad"] > df["ad2"]).astype(int)
    df["AD_bullbear_med"] = (df["ad"] > df["ad3"]).astype(int)
    df["AD_bullbear_long"] = (df["ad2"] > df["ad3"]).astype(int)
    df["AD_bias"] = (df["ad"] > 0).astype(int)
    df["AD_bias_long"] = (df["ad3"] > df["ad4"]).astype(int)

    df["AD_index"] = df["AD_bullbear_short"] + df["AD_bullbear_med"] + df["AD_bullbear_long"] + df["AD_bias"] + df["AD_bias_long"]
    return df

def calculate_di_index(df):
    """Calculate DI index components and final value"""
    # Calculate all components
    df = calculate_ma_index(df)
    df = calculate_willy_index(df)
    df = calculate_macd_index(df)
    df = calculate_obv_index(df)
    df = calculate_mfi_index(df)
    df = calculate_ad_index(df)

    # Calculate DI Value (sum of all components)
    df["di_value"] = (df["MA_index"] + df["Willy_index"] + df["macd_index"] +
                      df["OBV_index"] + df["mfi_index"] + df["AD_index"])

    # Round DI values to integers
    df["di_value"] = df["di_value"].round()

    # Initialize timeframe columns
    df["weekly_di_new"] = None
    df["daily_di_new"] = None
    df["4h_di_new"] = None

    # Assign DI value to appropriate timeframe column
    timeframe = df.attrs.get("timeframe")
    if timeframe == "weekly":
        df["weekly_di_new"] = df["di_value"]
    elif timeframe == "daily":
        df["daily_di_new"] = df["di_value"]
    else:  # 4h
        df["4h_di_new"] = df["di_value"]

    result = []
    for _, row in df.iterrows():
        time_val = row["time"] if "time" in row.index else row.name
        time_str = time_val.strftime("%Y-%m-%d %H:%M:%S") if isinstance(time_val, pd.Timestamp) else str(time_val)

        result.append({
            "time": time_str,
            "weekly_di_new": nan_to_none(row["weekly_di_new"]),
            "daily_di_new": nan_to_none(row["daily_di_new"]),
            "4h_di_new": nan_to_none(row["4h_di_new"]),
            "open": nan_to_none(row["open"]),  # Added open price
            "close": nan_to_none(row["close"])  # Keep close for reference
        })

    return result

def get_weekly_data(symbol="BTC", tsym="USD", limit=2000):
    """Get weekly OHLCV data for given cryptocurrency"""
    df_daily = get_daily_data(symbol, tsym, limit)

    # Convert index to datetime if it's not already
    if not isinstance(df_daily.index, pd.DatetimeIndex):
        df_daily.set_index('time', inplace=True)

    # Use W-MON for Monday-based weekly grouping
    df_weekly = df_daily.resample('W-MON').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volumefrom': 'sum',
        'volumeto': 'sum'
    })

    # Reset index to get time as a column
    df_weekly.reset_index(inplace=True)

    # Set timeframe attribute
    df_weekly.attrs['timeframe'] = 'weekly'

    return df_weekly

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

        # Get data for all timeframes
        df_daily = get_daily_data(symbol=symbol)
        df_4h = get_4h_data(symbol=symbol)
        df_weekly = get_weekly_data(symbol=symbol)

        # Calculate indices
        daily_di = calculate_di_index(df_daily)
        fourh_di = calculate_di_index(df_4h)
        weekly_di = calculate_di_index(df_weekly)

        # Process data
        results_by_date = {}
        weekly_values = {}  # Store weekly values for lookup

        # First, process weekly data to build lookup table
        for entry in weekly_di:
            date = entry["time"][:10]  # Get just the date part
            weekly_values[date] = entry["weekly_di_new"]

        # Process daily data to get structure and daily values
        df_daily_dict = df_daily.to_dict('records')
        daily_di_dict = {entry["time"][:10]: entry for entry in daily_di}

        for daily_row in df_daily_dict:
            date = pd.Timestamp(daily_row["time"]).strftime("%Y-%m-%d")
            daily_entry = daily_di_dict.get(date, {})

            if date not in results_by_date:
                results_by_date[date] = {
                    "time": date,
                    "daily_di_new": None,
                    "weekly_di_new": None,
                    "4h_values_new": [],  # List to store all 4h values
                    "4h_di_new": None,    # Latest 4h value
                    "total_new": None,
                    "di_ema_13_new": None,
                    "di_sma_30_new": None,
                    "trend_new": None,
                    "open": daily_row.get("open"),  # Get open price from daily data
                    "close": daily_row.get("close")  # Keep close for reference
                }
            results_by_date[date]["daily_di_new"] = daily_entry.get("daily_di_new")

        # Fill in weekly values, using previous week's value if missing
        dates = sorted(results_by_date.keys())
        prev_weekly = None
        for date in dates:
            if date in weekly_values:
                results_by_date[date]["weekly_di_new"] = weekly_values[date]
                prev_weekly = weekly_values[date]
            else:
                results_by_date[date]["weekly_di_new"] = prev_weekly

        # Process 4h data
        for entry in fourh_di:
            date = entry["time"][:10]
            if date in results_by_date:
                results_by_date[date]["4h_values_new"].append({
                    "time": entry["time"],
                    "value_new": entry["4h_di_new"]
                })
                # Update the latest 4h value for the day
                results_by_date[date]["4h_di_new"] = entry["4h_di_new"]

        # Convert to list and calculate additional fields
        results_list = []
        for date in sorted(results_by_date.keys(), reverse=True):
            data = results_by_date[date]
            # Calculate total
            components = [
                data["weekly_di_new"],
                data["daily_di_new"],
                data["4h_di_new"]
            ]
            total = sum(x for x in components if x is not None)
            data["total_new"] = total
            results_list.append(data)

        # Calculate EMAs and SMAs on the total values
        if results_list:
            totals = pd.Series([d["total_new"] for d in results_list])
            ema13 = ta.ema(totals, length=13)
            sma30 = totals.rolling(window=30, min_periods=1).mean()

            # Update results with EMAs, SMAs and trends
            for i, data in enumerate(results_list):
                data["di_ema_13_new"] = None if pd.isna(ema13.iloc[i]) else round(ema13.iloc[i], 2)
                data["di_sma_30_new"] = None if pd.isna(sma30.iloc[i]) else round(sma30.iloc[i], 2)

                if data["di_ema_13_new"] is not None and data["di_sma_30_new"] is not None:
                    data["trend_new"] = "bull" if data["di_ema_13_new"] > data["di_sma_30_new"] else "bear"
                else:
                    data["trend_new"] = None

        # Test case logging for 01.01.2024
        test_date = "2024-01-01"
        if test_date in results_by_date:
            logger.debug("\nFinal Test case values for 2024-01-01:")
            test_data = results_by_date[test_date]
            logger.debug(f"Weekly DI: {test_data['weekly_di_new']}")
            logger.debug(f"Daily DI: {test_data['daily_di_new']}")
            logger.debug(f"4h DI: {test_data['4h_di_new']}")
            logger.debug(f"Total: {test_data['total_new']}")

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

    # Convert timestamp to datetime and adjust to end of day (00:00:00 UTC следующего дня)
    df = pd.DataFrame(data['Data']['Data'])
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Логируем исходное время для проверки
    if len(df) > 0:
        logger.debug(f"Original timestamps for {symbol} daily data:")
        logger.debug(df['time'].head())

    # Отфильтровываем будущие даты и сегодняшний день
    today = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    df = df[df['time'] < today]

    # Логируем время свечей для проверки
    logger.debug(f"Sample of daily candle times for {symbol}:")
    logger.debug(df['time'].head())

    # Устанавливаем атрибут timeframe
    df.attrs['timeframe'] = 'daily'

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

    # Устанавливаем атрибут timeframe
    df.attrs['timeframe'] = '4h'

    set_cached_data(symbol, "4h_data", df)
    return df


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
        logger.error(f"Error in diindex endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@di_index_blueprint.route('/')
def index():
    return render_template('index.html')