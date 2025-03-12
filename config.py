import os

# API Configuration
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY", "2193d3ce789e90e474570058a3a96caa0d585ca0d0d0e62687a295c8402d29e9")

# Technical Analysis Parameters
TA_PARAMS = {
    'MA': {
        'micro': 6,
        'short': 13,
        'medium': 30,
        'long': 200
    },
    'MACD': {
        'fast': 12,
        'slow': 26,
        'signal': 9
    },
    'WILLY': {
        'period': 21,
        'smooth': 13
    }
}
