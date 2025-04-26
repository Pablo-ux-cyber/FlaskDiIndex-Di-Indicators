#!/usr/bin/env python3
"""
Скрипт для обновления формата исторических данных DI индекса.
Добавляет поле 4h_values_new для всех записей, где оно отсутствует.
"""

import os
import json
import logging
import sys
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('update_history')

# Директория для хранения исторических данных
HISTORY_DIR = "historical_data"

class DateTimeEncoder(json.JSONEncoder):
    """Кастомный энкодер для сериализации объектов datetime"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def get_history_files():
    """Получить список всех файлов истории DI индекса"""
    if not os.path.exists(HISTORY_DIR):
        logger.error(f"Директория {HISTORY_DIR} не существует")
        return []
    
    # Находим все файлы истории DI индекса
    files = []
    for filename in os.listdir(HISTORY_DIR):
        if filename.endswith("_di_combined_history.json"):
            files.append(os.path.join(HISTORY_DIR, filename))
    
    return files


def update_history_file(file_path, generate_missing=True):
    """Обновить формат истории в указанном файле
    
    Args:
        file_path (str): Путь к файлу истории
        generate_missing (bool, optional): Генерировать значения для отсутствующих 4h данных. По умолчанию True.
    """
    try:
        # Загрузка данных
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        symbol = os.path.basename(file_path).split('_')[0]
        total_records = len(data)
        updated_records = 0
        generated_records = 0
        
        # Обновляем каждую запись
        for date, record in data.items():
            if isinstance(record, dict):
                updated = False
                
                # Проверяем нужно ли генерировать 4h данные
                need_generation = generate_missing and "daily_di_new" in record and (
                    # Нет 4h_di_new или оно null
                    "4h_di_new" not in record or record["4h_di_new"] is None or
                    # Есть 4h_values_new, но это пустой массив
                    ("4h_values_new" in record and isinstance(record["4h_values_new"], list) and len(record["4h_values_new"]) == 0)
                )
                
                if need_generation:
                    daily_value = record.get("daily_di_new")
                    
                    # Используем daily значение как 4h (так как настоящих данных у нас нет)
                    record["4h_di_new"] = daily_value
                    
                    # Создаем записи для всех 6 4-часовых интервалов
                    record["4h_values_new"] = []
                    for hour in [0, 4, 8, 12, 16, 20]:
                        time_part = f"{hour:02d}:00:00"
                        full_time = f"{date} {time_part}"
                        
                        # Добавляем немного вариации для более реалистичных данных
                        # Значения растут в течение дня и заканчиваются на daily_value
                        if hour == 0:
                            value = max(0, daily_value - 2) if daily_value is not None else None
                        elif hour == 4:
                            value = max(0, daily_value - 1) if daily_value is not None else None
                        elif hour == 8:
                            value = daily_value
                        elif hour == 12:
                            value = daily_value
                        elif hour == 16:
                            value = daily_value
                        else:  # hour == 20
                            value = daily_value
                            
                        record["4h_values_new"].append({
                            "time": full_time,
                            "value_new": value
                        })
                    
                    generated_records += 1
                    updated = True
                
                # Если поля 4h_values_new нет, но есть 4h_di_new, создаем его
                elif "4h_di_new" in record and "4h_values_new" not in record:
                    # Создаем запись с 20:00:00 (последняя 4h свеча дня)
                    time_part = "20:00:00"
                    full_time = f"{date}T{time_part}"
                    record["4h_values_new"] = [{
                        "time": full_time,
                        "value_new": record["4h_di_new"]
                    }]
                    updated_records += 1
                    updated = True
                
                # Проверяем, что у всех записей в 4h_values_new есть значения
                if "4h_values_new" in record and isinstance(record["4h_values_new"], list):
                    for entry in record["4h_values_new"]:
                        if "value_new" not in entry and "4h_di_new" in record:
                            entry["value_new"] = record["4h_di_new"]
                            updated = True
                
                if updated:
                    updated_records += 1
        
        # Сохраняем обновленные данные
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, cls=DateTimeEncoder)
        
        logger.info(f"Обновлен файл {file_path}: {updated_records} из {total_records} записей (сгенерировано: {generated_records})")
        return updated_records, generated_records
        
    except Exception as e:
        logger.error(f"Ошибка при обновлении файла {file_path}: {str(e)}")
        return 0, 0


def main():
    """Основная функция скрипта"""
    import argparse
    
    # Создаем парсер аргументов
    parser = argparse.ArgumentParser(description="Обновление формата исторических данных DI индекса")
    parser.add_argument("--generate", action="store_true", 
                       help="Генерировать 4h данные на основе daily для записей, где они отсутствуют")
    args = parser.parse_args()
    
    logger.info("Запуск обновления формата исторических данных")
    if args.generate:
        logger.info("Включена генерация 4h данных для записей, где они отсутствуют")
    
    # Получаем список файлов истории
    history_files = get_history_files()
    
    if not history_files:
        logger.error("Файлы истории не найдены")
        return
    
    logger.info(f"Найдено файлов истории: {len(history_files)}")
    
    # Обновляем каждый файл
    total_updated = 0
    total_generated = 0
    for file_path in history_files:
        results = update_history_file(file_path, generate_missing=args.generate)
        if isinstance(results, tuple) and len(results) == 2:
            updated, generated = results
            total_updated += updated
            total_generated += generated
        else:
            # В случае ошибки или старой версии функции
            total_updated += results if isinstance(results, int) else 0
    
    logger.info(f"Обновление завершено. Всего обновлено записей: {total_updated}, сгенерировано: {total_generated}")


if __name__ == "__main__":
    main()