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

# Define available cryptocurrencies for testing
TEST_CRYPTOCURRENCIES = [
    {"symbol": "BTC", "name": "Bitcoin"},
    {"symbol": "ETH", "name": "Ethereum"},
    {"symbol": "BNB", "name": "BNB"}
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

def calculate_trends(df, symbol=""):
    """Calculate DI Index trends for a DataFrame"""
    try:
        # Calculate Total DI
        logger.debug(f"{symbol} - Calculating total DI")
        df["total_di"] = df.apply(
            lambda row: (
                sum(filter(None, [row["daily_di"], row["4h_di"], row["weekly_di"]]))
                if any(filter(None, [row["daily_di"], row["4h_di"], row["weekly_di"]]))
                else None
            ),
            axis=1
        )

        # Calculate EMA and SMA on total_di
        logger.debug(f"{symbol} - Calculating EMA and SMA")
        df["di_ema_13"] = ta.ema(df["total_di"], length=13)
        df["di_sma_30"] = df["total_di"].rolling(window=30, min_periods=30).mean()

        # Calculate trend
        df["trend"] = np.where(
            (df["di_ema_13"].notna() & df["di_sma_30"].notna()),
            np.where(df["di_ema_13"] > df["di_sma_30"], "bull", "bear"),
            None
        )

        # Log trend statistics
        trend_counts = df["trend"].value_counts()
        logger.info(f"{symbol} trend statistics:")
        logger.info(f"Total dates: {len(df)}")
        logger.info(f"Bull trends: {trend_counts.get('bull', 0)}")
        logger.info(f"Bear trends: {trend_counts.get('bear', 0)}")

        return df
    except Exception as e:
        logger.error(f"Error calculating trends for {symbol}: {str(e)}")
        return df

def calculate_di_index(df, symbol=""):
    """Calculate DI Index for a DataFrame"""
    try:
        df["DI_index"] = df["close"].rolling(window=14).mean()  # Simplified calculation for testing
        df["DI_index_EMA"] = ta.ema(df["DI_index"], length=13)
        df["DI_index_SMA"] = df["DI_index"].rolling(window=30, min_periods=30).mean()

        result = []
        for _, row in df.iterrows():
            result.append({
                "time": row["time"].strftime("%Y-%m-%d"),
                "DI_index": row["DI_index"] if not pd.isna(row["DI_index"]) else None,
                "close": row["close"] if not pd.isna(row["close"]) else None
            })
        return result
    except Exception as e:
        logger.error(f"Error calculating DI index for {symbol}: {str(e)}")
        return []

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
        daily_data = calculate_di_index(df_daily, symbol)
        fourh_data = calculate_di_index(df_4h, symbol)
        weekly_data = calculate_di_index(df_weekly, symbol)

        # Create DataFrame with all timeframes
        results = []
        for entry in daily_data:
            date = entry["time"]
            daily_di = entry["DI_index"]

            # Find matching 4h and weekly data
            fourh_di = next((x["DI_index"] for x in fourh_data if x["time"] == date), None)
            weekly_di = next((x["DI_index"] for x in weekly_data if x["time"] == date), None)

            results.append({
                "time": date,
                "daily_di": daily_di,
                "4h_di": fourh_di,
                "weekly_di": weekly_di,
                "close": entry["close"]
            })

        # Convert to DataFrame for trend calculations
        df_results = pd.DataFrame(results)
        df_results = calculate_trends(df_results, symbol)

        # Convert back to list format
        final_results = []
        for _, row in df_results.iterrows():
            entry = row.to_dict()
            # Convert NaN to None
            for key, value in entry.items():
                if isinstance(value, float) and math.isnan(value):
                    entry[key] = None
            final_results.append(entry)

        logger.info(f"Successfully calculated indices for {symbol}")
        return final_results

    except Exception as e:
        logger.error(f"Error in calculate_combined_indices for {symbol}: {str(e)}")
        return []

@di_index_blueprint.route('/')
def index():
    return render_template('index.html', cryptocurrencies=TEST_CRYPTOCURRENCIES)

@di_index_blueprint.route('/api/di_index')
def di_index():
    try:
        symbol = request.args.get("symbol", "BTC").upper()
        debug_mode = request.args.get("debug", "false").lower() == "true"

        if symbol == "ALL":
            try:
                logger.info(f"Processing test set of {len(TEST_CRYPTOCURRENCIES)} coins")
                results = []

                for coin in TEST_CRYPTOCURRENCIES:
                    try:
                        logger.info(f"Processing {coin['symbol']}")
                        if validate_symbol(coin["symbol"]):
                            coin_data = calculate_combined_indices(symbol=coin["symbol"], debug=debug_mode)
                            if coin_data:
                                results.append({
                                    "symbol": coin["symbol"],
                                    "name": coin["name"],
                                    "data": coin_data
                                })
                                logger.info(f"Successfully processed {coin['symbol']}")
                        time.sleep(2)
                    except Exception as coin_error:
                        logger.error(f"Error processing {coin['symbol']}: {str(coin_error)}")
                        continue

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
                    "name": next((c["name"] for c in TEST_CRYPTOCURRENCIES if c["symbol"] == symbol), symbol),
                    "data": coin_data
                }]
            })

    except Exception as e:
        logger.error(f"Error in di_index endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500