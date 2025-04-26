#!/usr/bin/env python3
"""
Скрипт для исправления JSON-файлов с историческими данными.
Ищет поврежденные файлы и очищает их, чтобы они могли быть 
заполнены заново при следующем запуске скрипта обновления.
"""

import os
import json
import logging
import glob
from datetime import datetime, date

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fix_json')

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

def fix_json_files():
    """Проверить все JSON-файлы на наличие ошибок и исправить их"""
    ensure_history_dir()
    
    # Получаем список всех JSON-файлов в каталоге
    json_files = glob.glob(os.path.join(HISTORY_DIR, "*.json"))
    logger.info(f"Найдено {len(json_files)} JSON-файлов для проверки")
    
    fixed_files = 0
    broken_files = 0
    
    for json_file in json_files:
        try:
            # Пытаемся загрузить JSON-файл
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Если удалось загрузить, сохраняем в правильном формате
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2, cls=DateTimeEncoder)
            
            logger.info(f"Файл {json_file} проверен и переформатирован")
            fixed_files += 1
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка в файле {json_file}: {e}")
            broken_files += 1
            
            # Создаем пустой файл, заменяя поврежденный
            with open(json_file, 'w') as f:
                if "_di_combined_history.json" in json_file:
                    # Создаем пустой словарь для истории DI индекса
                    json.dump({}, f, indent=2)
                else:
                    # Создаем пустой список для исторических данных
                    json.dump([], f, indent=2)
            
            logger.info(f"Файл {json_file} очищен для последующего перезаполнения")
    
    logger.info(f"Проверка завершена: {fixed_files} файлов в порядке, {broken_files} файлов исправлено")

if __name__ == "__main__":
    logger.info("Запуск скрипта исправления JSON-файлов")
    fix_json_files()
    logger.info("Скрипт завершил работу")