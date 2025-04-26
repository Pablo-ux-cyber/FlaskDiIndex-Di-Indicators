#!/usr/bin/env python3
"""
Скрипт для удаления поврежденных JSON-файлов с историческими данными.
Удаляет файлы, которые не могут быть загружены, чтобы они были 
пересозданы при следующем запуске скрипта обновления.
"""

import os
import json
import logging
import glob
import sys

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fix_json')

# Каталог для хранения исторических данных
HISTORY_DIR = "historical_data"

def ensure_history_dir():
    """Убедиться, что каталог для хранения истории существует"""
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)
        logger.info(f"Создан каталог для хранения исторических данных: {HISTORY_DIR}")

def delete_broken_json_files():
    """Удалить все поврежденные JSON-файлы"""
    ensure_history_dir()
    
    # Получаем список всех JSON-файлов в каталоге
    json_files = glob.glob(os.path.join(HISTORY_DIR, "*.json"))
    logger.info(f"Найдено {len(json_files)} JSON-файлов для проверки")
    
    valid_files = 0
    deleted_files = 0
    
    for json_file in json_files:
        try:
            # Пытаемся загрузить JSON-файл
            with open(json_file, 'r') as f:
                json.load(f)
            
            logger.info(f"Файл {json_file} корректный, оставляем")
            valid_files += 1
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка в файле {json_file}: {e}")
            
            # Спрашиваем подтверждение, если не указан флаг --force
            if '--force' not in sys.argv and input(f"Удалить файл {json_file}? (y/n): ").lower() != 'y':
                logger.info(f"Пропускаем файл {json_file}")
                continue
            
            # Удаляем поврежденный файл
            os.remove(json_file)
            deleted_files += 1
            logger.info(f"Файл {json_file} удален")
    
    logger.info(f"Проверка завершена: {valid_files} корректных файлов, {deleted_files} файлов удалено")

def delete_all_json_files():
    """Удалить все JSON-файлы в каталоге исторических данных"""
    ensure_history_dir()
    
    # Получаем список всех JSON-файлов в каталоге
    json_files = glob.glob(os.path.join(HISTORY_DIR, "*.json"))
    logger.info(f"Найдено {len(json_files)} JSON-файлов для удаления")
    
    # Спрашиваем подтверждение, если не указан флаг --force
    if '--force' not in sys.argv and input(f"Удалить все {len(json_files)} JSON-файлов? (y/n): ").lower() != 'y':
        logger.info("Операция отменена")
        return
    
    # Удаляем все файлы
    for json_file in json_files:
        os.remove(json_file)
        logger.info(f"Файл {json_file} удален")
    
    logger.info(f"Все {len(json_files)} файлов успешно удалены")

if __name__ == "__main__":
    logger.info("Запуск скрипта очистки JSON-файлов")
    
    if '--all' in sys.argv:
        # Удаляем все JSON-файлы
        delete_all_json_files()
    else:
        # Удаляем только поврежденные файлы
        delete_broken_json_files()
    
    logger.info("Скрипт завершил работу")