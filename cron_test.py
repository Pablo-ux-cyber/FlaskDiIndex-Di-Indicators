#!/usr/bin/env python
"""
Тестовый скрипт для проверки загрузки монет через крон
"""

import os
import sys
import json
import logging

# Настройка путей
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
logger = logging.getLogger("cron_test")

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

def main():
    """Основная функция скрипта"""
    logger.info("Тестовая проверка крон-скрипта")
    
    # Загружаем список монет для отслеживания
    symbols = load_tracked_symbols()
    
    if not symbols:
        logger.error("Не удалось загрузить список монет для отслеживания")
        return
    
    logger.info(f"Загружен список из {len(symbols)} монет для отслеживания")
    
    # Выводим список всех монет для проверки
    for i, symbol in enumerate(symbols):
        logger.info(f"Монета {i+1}: {symbol}")
    
    logger.info("Тест крон-скрипта завершен")

if __name__ == "__main__":
    main()