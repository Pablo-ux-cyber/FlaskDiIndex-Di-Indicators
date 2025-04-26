import os
import json
import pandas as pd
import logging
from datetime import datetime, date

logger = logging.getLogger('server')

# Каталог для хранения исторических данных
HISTORY_DIR = "historical_data"

# Класс для сериализации дат в JSON
class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return super(DateTimeEncoder, self).default(o)

def ensure_history_dir():
    """Убедиться, что каталог для хранения истории существует"""
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)
        logger.info(f"Создан каталог для хранения исторических данных: {HISTORY_DIR}")

def get_history_file_path(symbol, data_type):
    """Получить путь к файлу истории для указанного символа и типа данных"""
    ensure_history_dir()
    return os.path.join(HISTORY_DIR, f"{symbol}_{data_type}_history.json")

def save_historical_data(df, symbol, data_type):
    """Сохранить данные DataFrame в файл истории"""
    file_path = get_history_file_path(symbol, data_type)
    
    # Преобразуем датафрейм в формат для сохранения
    # Преобразуем timestamp в строки для корректной сериализации в JSON
    records = df.copy()
    records['time'] = records['time'].astype(str)
    
    # Преобразуем DataFrame в список словарей
    data_to_save = records.to_dict(orient='records')
    
    try:
        with open(file_path, 'w') as f:
            json.dump(data_to_save, f, cls=DateTimeEncoder)
        logger.info(f"Исторические данные сохранены: {file_path}, записей: {len(data_to_save)}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при сохранении исторических данных: {str(e)}")
        return False

def load_historical_data(symbol, data_type):
    """Загрузить исторические данные из файла"""
    file_path = get_history_file_path(symbol, data_type)
    
    if not os.path.exists(file_path):
        logger.info(f"Файл с историческими данными не найден: {file_path}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Преобразуем список словарей обратно в DataFrame
        df = pd.DataFrame(data)
        
        # Преобразуем строки обратно в timestamp
        df['time'] = pd.to_datetime(df['time'])
        
        logger.info(f"Загружены исторические данные: {file_path}, записей: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Ошибка при загрузке исторических данных: {str(e)}")
        return None

def merge_with_historical_data(new_df, symbol, data_type):
    """Объединить новые данные с историческими и сохранить"""
    logger.info(f"Объединение новых данных с историческими для {symbol}_{data_type}")
    
    # Загрузить существующие исторические данные
    hist_df = load_historical_data(symbol, data_type)
    
    if hist_df is None or hist_df.empty:
        logger.info(f"Исторические данные не найдены, сохраняем только новые данные")
        return save_historical_data(new_df, symbol, data_type)
    
    # Объединить данные и удалить дубликаты
    logger.info(f"Исторических записей: {len(hist_df)}, новых записей: {len(new_df)}")
    combined_df = pd.concat([hist_df, new_df], ignore_index=True)
    
    # Удаляем дубликаты, сохраняя самые новые значения для каждой временной точки
    combined_df = combined_df.sort_values('time', ascending=True)
    combined_df = combined_df.drop_duplicates(subset=['time'], keep='last')
    
    # Сортируем по времени перед сохранением
    combined_df = combined_df.sort_values('time', ascending=True)
    
    logger.info(f"После объединения и удаления дубликатов: {len(combined_df)} записей")
    
    # Сохраняем обновленные исторические данные
    return save_historical_data(combined_df, symbol, data_type)