#!/usr/bin/env python
"""
Скрипт для ежедневного автоматического обновления данных DI индекса.
Запускается через cron как:
0 1 * * * /путь/к/проекту/daily_update.py > /путь/к/проекту/daily_update.log 2>&1
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
        logging.FileHandler('daily_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("daily_update")

# Импортируем нужные модули из приложения
# Импортируем после установки путей
from server import (
    process_symbol_batch, 
    save_di_history,
    validate_symbol
)

# Список монет по умолчанию для отслеживания
DEFAULT_SYMBOLS = ["BTC", "ETH", "XRP", "LTC", "ADA", "DOT", "SOL", "DOGE", "LINK", "AVAX"]

def load_tracked_symbols():
    """Загружает список монет для отслеживания из конфигурационного файла"""
    config_path = os.path.join(script_dir, "tracked_symbols.json")
    
    # Если файл существует, загружаем список из него
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                symbols = json.load(f)
                logger.info(f"Загружены {len(symbols)} монет из конфигурационного файла")
                return symbols
        except Exception as e:
            logger.error(f"Ошибка при чтении файла tracked_symbols.json: {str(e)}")
    
    # Иначе используем список по умолчанию и создаем файл
    try:
        with open(config_path, 'w') as f:
            json.dump(DEFAULT_SYMBOLS, f, indent=2)
            logger.info(f"Создан файл со списком отслеживаемых монет по умолчанию")
    except Exception as e:
        logger.error(f"Ошибка при создании файла tracked_symbols.json: {str(e)}")
    
    return DEFAULT_SYMBOLS

def update_symbols_data(symbols):
    """Обновляет данные для списка символов"""
    logger.info(f"Запуск обновления данных для {len(symbols)} монет")
    
    # Проверяем, что все символы валидны
    valid_symbols = []
    for symbol in symbols:
        if validate_symbol(symbol):
            valid_symbols.append(symbol)
        else:
            logger.warning(f"Невалидный символ: {symbol}, пропускаем")
    
    # Обрабатываем символы пакетами
    batch_size = 5
    for i in range(0, len(valid_symbols), batch_size):
        batch = valid_symbols[i:i+batch_size]
        logger.info(f"Обработка пакета {i//batch_size + 1}/{(len(valid_symbols) + batch_size - 1)//batch_size}: {', '.join(batch)}")
        
        try:
            # Обработка с отладочной информацией отключена
            results = process_symbol_batch(batch, debug=False)
            
            for symbol, result in results.items():
                if isinstance(result, dict) and "error" in result:
                    logger.error(f"Ошибка при обработке {symbol}: {result['error']}")
                else:
                    logger.info(f"Успешно обновлены данные для {symbol}")
        except Exception as e:
            logger.error(f"Ошибка при обработке пакета: {str(e)}")
        
        # Небольшая пауза между пакетами, чтобы не перегружать API
        time.sleep(2)

def main():
    """Основная функция скрипта"""
    start_time = time.time()
    logger.info("Запуск ежедневного обновления данных DI индекса")
    
    # Загружаем список отслеживаемых монет
    symbols = load_tracked_symbols()
    logger.info(f"Загружен список из {len(symbols)} монет для отслеживания")
    
    # Обновляем данные
    update_symbols_data(symbols)
    
    # Записываем информацию о завершении
    elapsed_time = time.time() - start_time
    logger.info(f"Обновление данных завершено за {elapsed_time:.2f} секунд")

if __name__ == "__main__":
    main()