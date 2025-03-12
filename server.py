import math
import requests
import numpy as np
import time
import logging

# Monkey-patch for numpy's NaN compatibility with pandas-ta
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas as pd
import pandas_ta as ta
from flask import Blueprint, jsonify, request, render_template
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create blueprint for DI index routes
di_index_blueprint = Blueprint('di_index', __name__)

# Get API key from environment variable with fallback
API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY", "2193d3ce789e90e474570058a3a96caa0d585ca0d0d0e62687a295c8402d29e9")

# Define available cryptocurrencies
AVAILABLE_CRYPTOCURRENCIES = [
    {"symbol": "BTC", "name": "Bitcoin"},
    {"symbol": "ETH", "name": "Ethereum"},
    {"symbol": "XRP", "name": "XRP"},
    {"symbol": "BNB", "name": "BNB"},
    {"symbol": "SOL", "name": "Solana"},
    {"symbol": "ADA", "name": "Cardano"},
    {"symbol": "DOGE", "name": "Dogecoin"},
    {"symbol": "TRX", "name": "TRON"},
    {"symbol": "PI", "name": "Pi Network"},
    {"symbol": "LEO", "name": "LEO Token"},
    {"symbol": "HBAR", "name": "Hedera"},
    {"symbol": "LINK", "name": "Chainlink"},
    {"symbol": "XLM", "name": "Stellar"},
    {"symbol": "AVAX", "name": "Avalanche"},
    {"symbol": "SUI", "name": "Sui"},
    {"symbol": "SHIB", "name": "Shiba Inu"},
    {"symbol": "LTC", "name": "Litecoin"},
    {"symbol": "BCH", "name": "Bitcoin Cash"},
    {"symbol": "TON", "name": "Toncoin"},
    {"symbol": "OM", "name": "MANTRA OM"},
    {"symbol": "DOT", "name": "Polkadot"},
    {"symbol": "BGB", "name": "Bitget Token"},
    {"symbol": "HYPE", "name": "Hyperliquid"},
    {"symbol": "WBT", "name": "WhiteBIT Coin"},
    {"symbol": "XMR", "name": "Monero"},
    {"symbol": "UNI", "name": "Uniswap"},
    {"symbol": "APT", "name": "Aptos"},
    {"symbol": "NEAR", "name": "NEAR Protocol"},
    {"symbol": "AAVE", "name": "Aave"},
    {"symbol": "ETC", "name": "Ethereum Classic"}
]

def validate_symbol(symbol):
    """Validate if the cryptocurrency symbol exists on CryptoCompare"""
    try:
        url = f"https://min-api.cryptocompare.com/data/price?fsym={symbol}&tsyms=USD&api_key={API_KEY}"
        response = requests.get(url)
        data = response.json()
        if "Response" in data and data["Response"] == "Error":
            logger.warning(f"Invalid symbol {symbol}: {data.get('Message')}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating symbol {symbol}: {str(e)}")
        return False

def get_daily_data(symbol="BTC", tsym="USD", limit=2000):
    """Get daily OHLCV data for given cryptocurrency"""
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym={tsym}&limit={limit}&api_key={API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data.get("Response") != "Success":
            logger.error(f"API Error for {symbol}: {data.get('Message', 'Unknown error')}")
            return pd.DataFrame()

        df = pd.DataFrame(data['Data']['Data'])
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return df

        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    except Exception as e:
        logger.error(f"Error getting daily data for {symbol}: {str(e)}")
        return pd.DataFrame()

def get_4h_data(symbol="BTC", tsym="USD", limit=2000):
    """Get 4-hour OHLCV data for given cryptocurrency"""
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={symbol}&tsym={tsym}&limit={limit}&aggregate=4&api_key={API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data.get("Response") != "Success":
            logger.error(f"API Error for {symbol}: {data.get('Message', 'Unknown error')}")
            return pd.DataFrame()

        df = pd.DataFrame(data['Data']['Data'])
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return df

        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    except Exception as e:
        logger.error(f"Error getting 4h data for {symbol}: {str(e)}")
        return pd.DataFrame()

def get_weekly_data(symbol="BTC", tsym="USD", limit=2000):
    """Get weekly OHLCV data for given cryptocurrency"""
    try:
        df_daily = get_daily_data(symbol, tsym, limit)
        if df_daily.empty:
            return pd.DataFrame()

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
        logger.error(f"Error getting weekly data for {symbol}: {str(e)}")
        return pd.DataFrame()

def calculate_ma_index(df):
    """Calculate Moving Average Index based on PineScript logic"""
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
    """Calculate Willy Index based on PineScript logic"""
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
    """Calculate MACD Index based on PineScript logic"""
    df["fastMA"] = ta.ema(df["close"], length=12)
    df["slowMA"] = ta.ema(df["close"], length=26)
    df["macd"] = df["fastMA"] - df["slowMA"]
    df["signal"] = ta.sma(df["macd"], length=9)
    df["macd_bullbear"] = (df["macd"] > df["signal"]).astype(int)
    df["macd_bias"] = (df["macd"] > 0).astype(int)
    df["macd_index"] = df["macd_bullbear"] + df["macd_bias"]
    return df

def calculate_obv_index(df):
    """Calculate OBV Index based on PineScript logic"""
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
    """Calculate MFI Index based on PineScript logic"""
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
    """Calculate AD Index based on PineScript logic"""
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
    """Calculate DI Index based on PineScript logic"""
    try:
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

        # Calculate final DI_index
        df["DI_index"] = (df["MA_index"] + df["Willy_index"] + df["macd_index"] +
                         df["OBV_index"] + df["mfi_index"] + df["AD_index"])
        df["DI_index_EMA"] = ta.ema(df["DI_index"], length=13)
        df["DI_index_SMA"] = df["DI_index"].rolling(window=30, min_periods=30).mean()

        # Calculate trend
        df["trend"] = np.where(
            (df["DI_index_EMA"].notna() & df["DI_index_SMA"].notna()),
            np.where(df["DI_index_EMA"] > df["DI_index_SMA"], "bull", "bear"),
            None
        )

        if debug:
            trend_counts = df["trend"].value_counts()
            logger.debug(f"Trend distribution: {trend_counts.to_dict()}")

        result = []
        for _, row in df.iterrows():
            entry = {
                "time": row["time"].strftime("%Y-%m-%d"),
                "DI_index": nan_to_none(row["DI_index"]),
                "close": nan_to_none(row["close"]),
                "trend": row["trend"]
            }
            result.append(entry)
        return result
    except Exception as e:
        logger.error(f"Error calculating DI index: {str(e)}")
        return []

def nan_to_none(val):
    """Convert NaN values to None"""
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val

def process_coins_in_batches(coins, batch_size=3):
    """Process coins in smaller batches to avoid API rate limits"""
    results = []
    for i in range(0, len(coins), batch_size):
        batch = coins[i:i + batch_size]
        logger.info(f"Processing batch of {len(batch)} coins: {', '.join([c['symbol'] for c in batch])}")

        for coin in batch:
            try:
                logger.info(f"Processing {coin['symbol']}")
                if validate_symbol(coin["symbol"]):
                    coin_data = calculate_combined_indices(symbol=coin["symbol"])
                    if coin_data:
                        results.append({
                            "symbol": coin["symbol"],
                            "name": coin["name"],
                            "data": coin_data
                        })
                        logger.info(f"Successfully processed {coin['symbol']}")
                time.sleep(2)  # Delay between coins
            except Exception as coin_error:
                logger.error(f"Error processing {coin['symbol']}: {str(coin_error)}")
                continue

        time.sleep(3)  # Delay between batches
    return results

def calculate_combined_indices(symbol="BTC", debug=False):
    """Calculate and combine indices from different timeframes"""
    try:
        logger.info(f"Starting calculations for {symbol}")

        # Get data for all timeframes
        df_daily = get_daily_data(symbol=symbol)
        df_4h = get_4h_data(symbol=symbol)
        df_weekly = get_weekly_data(symbol=symbol)

        if df_daily.empty or df_4h.empty or df_weekly.empty:
            logger.error(f"Empty data received for {symbol}")
            return []

        # Calculate DI Index for each timeframe
        daily_di = calculate_di_index(df_daily, debug)
        fourh_di = calculate_di_index(df_4h, debug)
        weekly_di = calculate_di_index(df_weekly, debug)

        # Create date-indexed dictionaries
        results_by_date = {}

        # First process daily data
        for entry in daily_di:
            date = entry["time"]
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
            results_by_date[date]["daily_di"] = entry["DI_index"]
            results_by_date[date]["trend"] = entry["trend"]

        # Then process 4h data
        for entry in fourh_di:
            date = entry["time"]
            if date in results_by_date:
                results_by_date[date]["4h_di"] = entry["DI_index"]

        # Finally process weekly data
        for entry in weekly_di:
            date = entry["time"]
            if date in results_by_date:
                results_by_date[date]["weekly_di"] = entry["DI_index"]

        # Convert to list and sort by date
        results_list = list(results_by_date.values())
        results_list.sort(key=lambda x: x["time"], reverse=True)

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

        # Calculate EMA and SMA
        df["di_ema_13"] = ta.ema(df["total_di"], length=13)
        df["di_sma_30"] = df["total_di"].rolling(window=30, min_periods=30).mean()

        # Convert back to list format
        final_results = []
        for _, row in df.iterrows():
            entry = row.to_dict()
            # Convert NaN to None
            for key, value in entry.items():
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    entry[key] = None
            final_results.append(entry)

        logger.info(f"Successfully processed {symbol}")
        return final_results

    except Exception as e:
        logger.error(f"Error in calculate_combined_indices for {symbol}: {str(e)}")
        return []

@di_index_blueprint.route('/')
def index():
    return render_template('index.html', cryptocurrencies=AVAILABLE_CRYPTOCURRENCIES)

@di_index_blueprint.route('/api/di_index')
def di_index():
    try:
        symbol = request.args.get("symbol", "BTC").upper()
        debug_mode = request.args.get("debug", "false").lower() == "true"

        if symbol == "ALL":
            try:
                logger.info(f"Processing test set of {len(AVAILABLE_CRYPTOCURRENCIES)} coins")
                results = process_coins_in_batches(AVAILABLE_CRYPTOCURRENCIES[:30])  # Process first 30 coins

                if not results:
                    logger.error("No valid data received for any cryptocurrency")
                    return jsonify({"error": "No valid data received for any cryptocurrency"}), 500

                logger.info(f"Successfully processed {len(results)} coins")
                return jsonify({
                    "coins": results
                })

            except Exception as e:
                logger.error(f"Error processing ALL request: {str(e)}")
                return jsonify({"error": str(e)}), 500
        else:
            # Single coin request
            if not validate_symbol(symbol):
                return jsonify({"error": f"Invalid cryptocurrency symbol: {symbol}"}), 400

            coin_data = calculate_combined_indices(symbol=symbol, debug=debug_mode)
            if not coin_data:
                return jsonify({"error": f"No data available for {symbol}"}), 404

            # Make response format consistent with ALL endpoint
            return jsonify({
                "coins": [{
                    "symbol": symbol,
                    "name": next((c["name"] for c in AVAILABLE_CRYPTOCURRENCIES if c["symbol"] == symbol), symbol),
                    "data": coin_data
                }]
            })

    except Exception as e:
        logger.error(f"Error in di_index endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500