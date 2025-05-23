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

# Импортируем модуль для работы с историческими данными
from utils.history_manager import (
    load_historical_data, 
    save_historical_data, 
    merge_with_historical_data
)

# Импортируем модуль для работы с историей DI индекса
from utils.di_history_manager import (
    save_di_history,
    load_di_history
)

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
    # Вычисляем все компоненты DI индекса для всех данных без исключений
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
        # Для 4h данных просто устанавливаем значения для всех точек
        # Фильтрация точек из первых 7 дней будет произведена позже
        df["4h_di_new"] = df["di_value"]
        
        # Проверим, есть ли помеченные для скрытия точки
        if "_hide_in_frontend" in df.columns:
            hide_mask = df["_hide_in_frontend"] == True
            if hide_mask.any():
                logger.debug(f"Найдено {hide_mask.sum()} точек, которые будут скрыты при передаче на фронтенд")

    result = []
    for _, row in df.iterrows():
        # Проверяем, нужно ли скрыть эту точку на фронтенде
        if "_hide_in_frontend" in row and row["_hide_in_frontend"] == True:
            # Пропускаем эту точку - она не попадет в результат
            continue
            
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

    # Get the current date and find the last completed Sunday
    current_date = pd.Timestamp.now().normalize()
    days_since_sunday = current_date.dayofweek + 1
    last_completed_sunday = current_date - pd.Timedelta(days=days_since_sunday)

    # Filter data to include only completed weeks
    df_daily = df_daily[df_daily.index <= last_completed_sunday]

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

    # Log sample of weekly data for verification
    logger.debug(f"Sample of weekly data timestamps for {symbol}:")
    if not df_weekly.empty:
        logger.debug(df_weekly['time'].head())

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

def get_4h_data(symbol="BTC", tsym="USD", limit=2000):
    """Get 4-hour OHLCV data for given cryptocurrency"""
    logger.debug(f"DEBUG: get_4h_data started for {symbol}")
    
    # Сначала проверяем кеш
    cached_data = get_cached_data(symbol, "4h_data")
    if cached_data is not None and not cached_data.empty:
        logger.debug(f"DEBUG: Using cached data for {symbol}")
        return cached_data
    
    # Затем проверяем, есть ли сохраненные исторические данные
    historical_data = load_historical_data(symbol, "4h_data")
    
    all_data = []

    # First request - current period
    url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={symbol}&tsym={tsym}&limit={limit}&aggregate=4&api_key={API_KEY}"
    logger.debug(f"DEBUG: URL for 4h data request: {url}")
    
    try:
        logger.debug(f"DEBUG: Making first request to API")
        response = requests.get(url)
        logger.debug(f"DEBUG: Response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"DEBUG: API Error: {response.text}")
            raise Exception(f"Error getting 4-hour data: HTTP error {response.status_code}")
        
        data = response.json()
        logger.debug(f"DEBUG: API Response: {data.get('Response')}")
        
        if data.get("Response") != "Success":
            logger.error(f"DEBUG: Unsuccessful API response: {data}")
            raise Exception(f"Error getting 4-hour data: {data}")

        logger.debug("\nFirst request details:")
        logger.debug(f"TimeFrom: {datetime.fromtimestamp(data['Data']['TimeFrom'])}")
        logger.debug(f"TimeTo: {datetime.fromtimestamp(data['Data']['TimeTo'])}")
        logger.debug(f"Data points: {len(data['Data']['Data'])}")
        logger.debug(f"DEBUG: Received {len(data['Data']['Data'])} data points")
        
        # Добавляем все данные из первого запроса (это более новые данные)
        all_data.extend(data['Data']['Data'])
        
        # Second request - previous period
        toTs = data['Data']['TimeFrom']  # Use TimeFrom from first request as end time for second request
        second_url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={symbol}&tsym={tsym}&limit={limit}&aggregate=4&toTs={toTs}&api_key={API_KEY}"
        logger.debug(f"DEBUG: URL for second 4h data request: {second_url}")
        
        logger.debug(f"DEBUG: Making second request to API")
        response = requests.get(second_url)
        logger.debug(f"DEBUG: Second response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"DEBUG: Second API Error: {response.text}")
            # Не прерываем выполнение, просто логируем ошибку
        else:
            data = response.json()
            
            if data.get("Response") == "Success":
                logger.debug("\nSecond request details:")
                logger.debug(f"TimeFrom: {datetime.fromtimestamp(data['Data']['TimeFrom'])}")
                logger.debug(f"TimeTo: {datetime.fromtimestamp(data['Data']['TimeTo'])}")
                logger.debug(f"Data points: {len(data['Data']['Data'])}")
                
                # Вычисляем время, которое соответствует 7 дням от начала второго блока данных
                # Это нужно для пропуска первых 7 дней из второго запроса (более старые данные)
                # так как они могут быть менее стабильны
                if len(data['Data']['Data']) > 0:
                    # Получаем начальное время из данных
                    start_time = data['Data']['TimeFrom']
                    # Вычисляем время, после которого будем использовать данные (7 дней = 7 * 24 * 60 * 60 секунд)
                    skip_until_time = start_time + (7 * 24 * 60 * 60)
                    
                    # Преобразуем в даты для вывода в лог
                    start_date = datetime.fromtimestamp(start_time)
                    skip_until_date = datetime.fromtimestamp(skip_until_time)
                    end_date = datetime.fromtimestamp(toTs)
                    
                    logger.debug(f"Second request range: {start_date} to {end_date}")
                    logger.debug(f"Skipping data until: {skip_until_date}")
                    
                    # Проверим, есть ли проблемные даты в запросе 
                    problem_dates = ["2024-11-09", "2024-11-11"]
                    for date_str in problem_dates:
                        # Конвертируем дату в timestamp начала дня
                        date_ts = int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())
                        # Проверяем, попадает ли в диапазон текущего запроса
                        if date_ts >= start_time and date_ts < toTs:
                            logger.debug(f"Problem date {date_str} is in current request range")
                            if date_ts < skip_until_time:
                                logger.debug(f"WARNING: Date {date_str} would be skipped by 7-day filter!")
                    
                    # Используем все точки для расчетов, но помечаем первые 7 дней
                    # специальным флагом, чтобы не показывать их на фронтенде
                    filtered_data = []
                    
                    for point in data['Data']['Data']:
                        if point['time'] < toTs:  # Убедимся, что точка в нужном временном интервале
                            # Если точка в первые 7 дней, отмечаем ее специальным ключом
                            # Этот флаг будет использоваться только при формировании ответа для фронтенда
                            if point['time'] < skip_until_time:
                                point['_hide_in_frontend'] = True
                            filtered_data.append(point)
                    
                    logger.debug(f"Всего точек из второго запроса: {len(filtered_data)}, из них будет скрыто на фронтенде: {sum(1 for p in filtered_data if p.get('_hide_in_frontend', False))}")
                    
                    # Проверяем наличие проблемных дат в данных до фильтрации
                    for date_str in problem_dates:
                        date_ts = int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())
                        # Находим точки данных для этой даты (начало дня до начало следующего дня)
                        next_day_ts = date_ts + (24 * 60 * 60)
                        points_for_date = [p for p in data['Data']['Data'] 
                                        if p['time'] >= date_ts and p['time'] < next_day_ts]
                        if points_for_date:
                            logger.debug(f"Found {len(points_for_date)} 4h points for {date_str} before filtering")
                        else:
                            logger.debug(f"No 4h points found for {date_str} in second request")
                    
                    all_data.extend(filtered_data)
                    logger.debug(f"Added {len(filtered_data)} filtered points from second request (skipped first 7 days)")
    
    except Exception as e:
        logger.error(f"DEBUG: Exception in get_4h_data: {str(e)}")
        import traceback
        logger.error(f"DEBUG: {traceback.format_exc()}")
        
        # Если API запрос не удался, но у нас есть исторические данные - используем их
        if historical_data is not None and not historical_data.empty:
            logger.info(f"Using historical data as fallback for {symbol}")
            return historical_data
        raise

    # Convert to DataFrame and process
    df = pd.DataFrame(all_data)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Log data boundaries
    logger.debug("\nFinal data boundaries:")
    logger.debug(f"Earliest data: {df['time'].min()}")
    logger.debug(f"Latest data: {df['time'].max()}")
    logger.debug(f"Total points: {len(df)}")

    # Sort and remove duplicates
    df = df.sort_values('time', ascending=True)
    df = df.drop_duplicates(subset=['time'], keep='last')

    # Group by date and hour
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour

    # Log final distribution
    distribution = df.groupby(['date', 'hour']).size().reset_index(name='count')
    logger.debug("\nFinal data distribution:")
    logger.debug(distribution)

    # Set timeframe attribute
    df.attrs['timeframe'] = '4h'
    
    # Сохраняем в кеш
    set_cached_data(symbol, "4h_data", df)
    
    # Объединяем новые данные с историческими и сохраняем
    try:
        # Если у нас есть исторические данные, объединяем с новыми
        if historical_data is not None and not historical_data.empty:
            merge_with_historical_data(df, symbol, "4h_data")
        else:
            # Иначе просто сохраняем текущие данные как исторические
            save_historical_data(df, symbol, "4h_data")
    except Exception as e:
        logger.error(f"Error saving historical data: {str(e)}")
        # Ошибка сохранения исторических данных не должна прерывать выполнение
    
    return df

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

        # Process daily data
        daily_data = df_daily.copy()
        if isinstance(daily_data.index, pd.DatetimeIndex):
            daily_data = daily_data.reset_index()

        daily_di_dict = {entry["time"][:10]: entry for entry in daily_di}

        # Process 4h data first to organize by date
        fourh_by_date = {}
        for entry in fourh_di:
            date = entry["time"][:10]
            if date not in fourh_by_date:
                fourh_by_date[date] = []
            fourh_by_date[date].append({
                "time": entry["time"],
                "value_new": entry["4h_di_new"]
            })

        for _, row in daily_data.iterrows():
            date = pd.Timestamp(row["time"]).strftime("%Y-%m-%d")
            daily_entry = daily_di_dict.get(date, {})

            if date not in results_by_date:
                # Get 4h values for this date, sorted by time
                fourh_values = sorted(fourh_by_date.get(date, []), key=lambda x: x["time"])

                # Use the last value (20:00:00) for the main table display
                fourh_display_value = fourh_values[-1]["value_new"] if fourh_values else None

                results_by_date[date] = {
                    "time": date,
                    "daily_di_new": None,
                    "weekly_di_new": None,
                    "4h_values_new": fourh_values,  # Store all 4h values for the day
                    "4h_di_new": fourh_display_value,  # Use 20:00:00 value for display
                    "total_new": None,
                    "di_ema_13_new": None,
                    "di_sma_30_new": None,
                    "trend_new": None,
                    "open": float(row["open"]) if pd.notnull(row["open"]) else None,
                    "close": float(row["close"]) if pd.notnull(row["close"]) else None
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

        # Convert to list and calculate additional fields
        results_list = []
        for date in sorted(results_by_date.keys(), reverse=True):
            data = results_by_date[date]
            # Calculate total using the 00:00:00 4h value
            components = [
                data["weekly_di_new"],
                data["daily_di_new"],
                data["4h_di_new"]  # Now using 20:00:00 value
            ]
            total = sum(x for x in components if x is not None)
            data["total_new"] = total
            results_list.append(data)

        # Calculate EMAs and SMAs on the total values
        if results_list:
            # Create a new DataFrame for calculations
            df_calcs = pd.DataFrame([{
                'date': d['time'],
                'total': d['total_new']
            } for d in results_list]).set_index('date')

            # Sort by date for correct calculation
            df_calcs = df_calcs.sort_index()

            # Calculate EMAs and SMAs
            df_calcs['ema13'] = ta.ema(df_calcs['total'], length=13)
            df_calcs['sma30'] = df_calcs['total'].rolling(window=30, min_periods=1).mean()

            # Create a lookup dictionary
            calcs_dict = df_calcs.to_dict('index')

            # Update results with EMAs, SMAs and trends
            for data in results_list:
                date = data['time']
                if date in calcs_dict:
                    values = calcs_dict[date]
                    data['di_ema_13_new'] = None if pd.isna(values['ema13']) else round(float(values['ema13']), 2)
                    data['di_sma_30_new'] = None if pd.isna(values['sma30']) else round(float(values['sma30']), 2)

                    if data['di_ema_13_new'] is not None and data['di_sma_30_new'] is not None:
                        data['trend_new'] = 'bull' if data['di_ema_13_new'] > data['di_sma_30_new'] else 'bear'
                    else:
                        data['trend_new'] = None

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

    # Convert timestamp to datetime
    df = pd.DataFrame(data['Data']['Data'])
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Логируем исходное время для проверки
    if len(df) > 0:
        logger.debug(f"Original timestamps for {symbol} daily data:")
        logger.debug(df['time'].head())

    # Отфильтровываем текущий день и будущие даты, но включаем вчерашний день
    today = pd.Timestamp.now().normalize()  # Получаем начало текущего дня в UTC
    yesterday = today - pd.Timedelta(days=1)  # Вчерашний день уже завершён
    df = df[df['time'] <= yesterday]  # Оставляем завершённые дни, включая вчерашний

    # Логируем время свечей для проверки
    logger.debug(f"Sample of daily candle times for {symbol}:")
    logger.debug(df['time'].head())

    # Устанавливаем атрибут timeframe
    df.attrs['timeframe'] = 'daily'

    set_cached_data(symbol, "daily_data", df)
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
                        
                        # Сохраняем историю DI индекса
                        try:
                            save_di_history(symbol, result)
                        except Exception as e:
                            logger.error(f"Error saving DI history for {symbol}: {str(e)}")
                            
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
        # Добавляем параметр использования исторических данных
        use_history = request.args.get("use_history", "true").lower() == "true"

        logger.debug(f"Received request for symbols: {symbols}, use_history: {use_history}")
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        logger.debug(f"Parsed symbol list: {symbol_list}")

        if not symbol_list:
            return jsonify({"error": "No valid symbols provided"}), 400

        # Validate symbols first
        for symbol in symbol_list:
            if not validate_symbol(symbol):
                logger.error(f"Invalid cryptocurrency symbol: {symbol}")
                return jsonify({"error": f"Invalid cryptocurrency symbol: {symbol}"}), 400

        # Получаем данные от API CryptoCompare
        results = process_symbol_batch(symbol_list, debug_mode)
        
        # Если нужно, дополняем результаты историческими данными из файлов
        if use_history:
            from utils.di_history_manager import load_di_history
            
            for symbol in symbol_list:
                # Загружаем историю DI индекса
                history = load_di_history(symbol)
                
                if history and symbol in results:
                    # Словарь текущих записей для быстрого доступа и обновления
                    current_entries = {}
                    for i, entry in enumerate(results[symbol]):
                        if isinstance(entry, dict) and "time" in entry and "error" not in entry:
                            current_entries[entry["time"]] = (i, entry)
                    
                    # Добавляем данные из истории
                    historical_entries = []
                    
                    for date, hist_data in history.items():
                        # Убеждаемся, что запись - словарь
                        if isinstance(hist_data, dict) and "daily_di_new" in hist_data:
                            daily_value = hist_data.get("daily_di_new")
                            
                            # Если 4h_di_new отсутствует или null, заполняем его значением из daily
                            if "4h_di_new" not in hist_data or hist_data["4h_di_new"] is None:
                                hist_data["4h_di_new"] = daily_value
                            
                            # Мы больше не генерируем синтетические 4-часовые данные
                            # Если 4h_values_new отсутствует, создаем пустой массив
                            if "4h_values_new" not in hist_data:
                                hist_data["4h_values_new"] = []
                            
                            # Проверяем, есть ли эта дата в текущих результатах
                            if date in current_entries:
                                # Если дата уже есть, обновляем только 4h данные из истории
                                idx, current_data = current_entries[date]
                                # Если в истории есть 4h_values_new, используем их
                                if hist_data.get("4h_values_new"):
                                    current_data["4h_values_new"] = hist_data["4h_values_new"]
                                    current_data["4h_di_new"] = hist_data["4h_di_new"]
                                    # Обновляем запись в результатах
                                    results[symbol][idx] = current_data
                            else:
                                # Если даты нет, добавляем всю запись из истории
                                historical_entries.append(hist_data)
                    
                    # Добавляем исторические данные
                    if isinstance(results[symbol], list) and historical_entries:
                        # Добавляем исторические записи, которых нет в текущих результатах
                        results[symbol].extend(historical_entries)
                        
                        # Мы не генерируем синтетические данные для 4h_values_new,
                        # а только используем реальные исторические данные
                        
                        # Сортируем результаты по дате в обратном порядке (сначала новые)
                        results[symbol] = sorted(results[symbol], key=lambda x: x.get("time", ""), reverse=True)
                        
                        logger.debug(f"Обработаны исторические данные для {symbol}, всего записей: {len(results[symbol])}")
        
        logger.debug(f"Calculation completed, results keys: {list(results.keys())}")
        
        return jsonify(results)

    except Exception as e:
        logger.error(f"Error in diindex endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@di_index_blueprint.route('/')
def index():
    return render_template('index.html')