import os
import json
import logging
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

# Директория для хранения исторических данных
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

def get_historical_file_path(symbol, timeframe):
    """Получает путь к файлу с историческими данными для указанного символа и таймфрейма"""
    return os.path.join(DATA_DIR, f"{symbol.lower()}_{timeframe.lower()}_history.json")

def save_historical_data(symbol, timeframe, data):
    """
    Сохраняет исторические данные в JSON файл
    
    Args:
        symbol (str): Символ криптовалюты
        timeframe (str): Таймфрейм ('4h', 'daily')
        data (list): Список данных OHLCV
    """
    if not data:
        logger.warning(f"Нет данных для сохранения: {symbol} {timeframe}")
        return
    
    file_path = get_historical_file_path(symbol, timeframe)
    
    # Если файл существует, загружаем данные и объединяем с новыми
    existing_data = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
        except Exception as e:
            logger.error(f"Ошибка при чтении исторического файла {file_path}: {str(e)}")
    
    # Объединяем данные и удаляем дубликаты на основе временной метки
    all_data = existing_data + data
    unique_data = {}
    for item in all_data:
        unique_data[item['time']] = item
    
    # Сортируем по времени
    sorted_data = [unique_data[timestamp] for timestamp in sorted(unique_data.keys())]
    
    # Сохраняем в файл
    try:
        with open(file_path, 'w') as f:
            json.dump(sorted_data, f)
        logger.info(f"Сохранено {len(sorted_data)} точек данных для {symbol} {timeframe}. Добавлено {len(sorted_data) - len(existing_data)} новых точек.")
    except Exception as e:
        logger.error(f"Ошибка при сохранении исторического файла {file_path}: {str(e)}")

def load_historical_data(symbol, timeframe):
    """
    Загружает исторические данные из JSON файла
    
    Args:
        symbol (str): Символ криптовалюты
        timeframe (str): Таймфрейм ('4h', 'daily')
        
    Returns:
        list: Список данных OHLCV или пустой список, если данные не найдены
    """
    file_path = get_historical_file_path(symbol, timeframe)
    
    if not os.path.exists(file_path):
        logger.debug(f"Файл с историческими данными не найден: {file_path}")
        return []
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            logger.info(f"Загружено {len(data)} точек исторических данных для {symbol} {timeframe}")
            return data
    except Exception as e:
        logger.error(f"Ошибка при загрузке исторического файла {file_path}: {str(e)}")
        return []

def merge_historical_and_api_data(historical_data, api_data):
    """
    Объединяет исторические данные с данными из API
    
    Args:
        historical_data (list): Исторические данные из JSON
        api_data (list): Данные полученные от API
        
    Returns:
        list: Объединенные данные без дубликатов
    """
    # Преобразуем все в словарь для быстрого доступа и удаления дубликатов
    merged_data = {}
    
    # Добавляем исторические данные
    for item in historical_data:
        merged_data[item['time']] = item
    
    # Добавляем данные API (с приоритетом)
    for item in api_data:
        merged_data[item['time']] = item
    
    # Преобразуем обратно в список и сортируем по времени
    sorted_data = [merged_data[timestamp] for timestamp in sorted(merged_data.keys())]
    
    return sorted_data

def convert_to_dataframe(data):
    """
    Преобразует данные из JSON в DataFrame pandas
    
    Args:
        data (list): Список данных OHLCV
        
    Returns:
        pandas.DataFrame: Данные в формате DataFrame
    """
    if not data:
        logger.warning("Нет данных для преобразования в DataFrame")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Преобразуем timestamp в datetime
    df['time_dt'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time_dt', inplace=True)
    
    return df