#!/usr/bin/env python
"""
Тестовый скрипт для обновления небольшого списка монет.
"""

import os
import sys
import time
import logging
import concurrent.futures
import json
from datetime import datetime

# Настройка путей для запуска из cron
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_update")

# Импорт функции для обработки символа
from server import process_symbol

def load_tracked_symbols():
    """Загружает список монет для отслеживания из конфигурационного файла"""
    try:
        # Для тестирования берем только SOL для проверки
        test_symbols = ["SOL"]
        logger.info(f"Загружен тестовый список из {len(test_symbols)} монет для проверки: {test_symbols}")
        return test_symbols
    except Exception as e:
        logger.error(f"Ошибка при загрузке списка монет: {str(e)}")
        return []

def update_symbols_data(symbols):
    """Обновляет данные для списка символов"""
    logger.info(f"Запуск обновления данных для {len(symbols)} монет")
    
    # Создаем пул потоков для параллельной обработки
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Запускаем обработку всех символов
        futures = {executor.submit(process_symbol, symbol, True): symbol for symbol in symbols}
        
        # Ожидаем завершения всех задач
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            try:
                # Получаем результат выполнения
                result = future.result()
                logger.info(f"Обработка {symbol} завершена")
            except Exception as e:
                logger.error(f"Ошибка при обработке {symbol}: {str(e)}")

def main():
    """Основная функция скрипта"""
    logger.info("Запуск тестового обновления данных DI индекса")
    
    # Загружаем список монет для отслеживания
    symbols = load_tracked_symbols()
    
    if not symbols:
        logger.error("Не удалось загрузить список монет для отслеживания")
        return
    
    # Обновляем данные для всех монет
    update_symbols_data(symbols)
    
    logger.info("Обновление данных завершено")

if __name__ == "__main__":
    main()