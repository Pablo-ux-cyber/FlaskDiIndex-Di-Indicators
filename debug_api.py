import requests
import logging
import sys
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_cryptocompare_api():
    """Тестирование различных эндпоинтов CryptoCompare API"""
    symbol = "BTC"
    tsym = "USD"
    limit = 10
    API_KEY = "2193d3ce789e90e474570058a3a96caa0d585ca0d0d0e62687a295c8402d29e9"  # Используем тестовый ключ
    
    # Список URL для тестирования
    urls = [
        # 1. Обычный URL для daily данных
        f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym={tsym}&limit={limit}&api_key={API_KEY}",
        
        # 2. URL для histohour с параметром aggregate=4
        f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={symbol}&tsym={tsym}&limit={limit}&aggregate=4&api_key={API_KEY}",
        
        # 3. Альтернативный URL для 4h данных (histo4h)
        f"https://min-api.cryptocompare.com/data/v2/histo4h?fsym={symbol}&tsym={tsym}&limit={limit}&api_key={API_KEY}",
        
        # 4. URL с параметром toTs
        f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={symbol}&tsym={tsym}&limit={limit}&aggregate=4&toTs={int(datetime.now().timestamp())}&api_key={API_KEY}"
    ]
    
    # Тестируем каждый URL
    for i, url in enumerate(urls, 1):
        logger.info(f"Тест #{i}: {url}")
        try:
            response = requests.get(url)
            data = response.json()
            
            logger.info(f"Статус: {response.status_code}, Успех: {data.get('Response') == 'Success'}")
            
            if data.get('Response') == 'Success':
                if 'Data' in data and 'Data' in data['Data']:
                    logger.info(f"Получено точек данных: {len(data['Data']['Data'])}")
                    if len(data['Data']['Data']) > 0:
                        logger.info(f"Первая точка: {data['Data']['Data'][0]}")
            else:
                logger.error(f"Ошибка API: {data}")
                
        except Exception as e:
            logger.error(f"Исключение при тестировании URL #{i}: {str(e)}")
            
    logger.info("Тестирование завершено")

if __name__ == "__main__":
    test_cryptocompare_api()