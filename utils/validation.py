import requests
from config import CRYPTOCOMPARE_API_KEY
import logging

logger = logging.getLogger(__name__)

def validate_symbol(symbol):
    """
    Validate if the cryptocurrency symbol exists on CryptoCompare
    
    Args:
        symbol (str): Cryptocurrency symbol to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        url = f"https://min-api.cryptocompare.com/data/price?fsym={symbol}&tsyms=USD&api_key={CRYPTOCOMPARE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        if "Response" in data and data["Response"] == "Error":
            logger.warning(f"Invalid symbol: {symbol}")
            return False
        return True
    except Exception as e:
        logger.error(f"Symbol validation error: {str(e)}")
        return False

def validate_params(params):
    """
    Validate API request parameters
    
    Args:
        params (dict): Request parameters
        
    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    if not params.get("symbol"):
        return False, "Symbol parameter is required"
    
    symbol = params["symbol"].upper()
    if not symbol.isalnum():
        return False, "Invalid symbol format"
        
    return True, None
