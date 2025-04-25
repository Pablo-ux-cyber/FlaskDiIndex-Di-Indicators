import os
import json
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger('server')

# Каталог для хранения исторических данных
HISTORY_DIR = "historical_data"

def ensure_history_dir():
    """Убедиться, что каталог для хранения истории существует"""
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)
        logger.info(f"Создан каталог для хранения исторических данных: {HISTORY_DIR}")

def get_di_history_file_path(symbol):
    """Получить путь к файлу истории DI индекса для указанного символа"""
    ensure_history_dir()
    return os.path.join(HISTORY_DIR, f"{symbol}_di_combined_history.json")

def save_di_history(symbol, results_list):
    """Сохранить историю DI индекса в файл
    
    Args:
        symbol (str): Символ криптовалюты (например, 'BTC')
        results_list (list): Список результатов расчета DI индекса
    """
    file_path = get_di_history_file_path(symbol)
    
    # Преобразуем список в формат, оптимальный для сохранения
    # Используем дату как ключ
    current_data = {}
    for entry in results_list:
        # Создаем копию данных, чтобы не изменять оригинал
        data_to_save = entry.copy()
        
        # Извлекаем дату из результата
        date = data_to_save.get('time')
        if date:
            # Удаляем поле '4h_values_new', так как оно может содержать сложные структуры
            if '4h_values_new' in data_to_save:
                del data_to_save['4h_values_new']
            
            # Сохраняем запись
            current_data[date] = data_to_save
    
    try:
        # Загружаем существующие данные, если они есть
        existing_data = {}
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    logger.error(f"Ошибка чтения существующего файла истории: {file_path}")
        
        # Объединяем текущие и существующие данные
        # Текущие данные имеют приоритет
        existing_data.update(current_data)
        
        # Сохраняем объединенные данные
        with open(file_path, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
        logger.info(f"История DI индекса сохранена: {file_path}, записей: {len(existing_data)}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при сохранении истории DI индекса: {str(e)}")
        return False

def load_di_history(symbol):
    """Загрузить историю DI индекса из файла
    
    Args:
        symbol (str): Символ криптовалюты (например, 'BTC')
        
    Returns:
        dict: Словарь с историей DI индекса или None при ошибке
    """
    file_path = get_di_history_file_path(symbol)
    
    if not os.path.exists(file_path):
        logger.info(f"Файл с историей DI индекса не найден: {file_path}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Загружена история DI индекса: {file_path}, записей: {len(data)}")
        return data
    except Exception as e:
        logger.error(f"Ошибка при загрузке истории DI индекса: {str(e)}")
        return None