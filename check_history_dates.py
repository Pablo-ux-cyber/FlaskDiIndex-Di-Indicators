#!/usr/bin/env python3
"""
Скрипт для проверки последних дат в исторических файлах
"""

import os
import json
import datetime

def load_json_file(file_path):
    """Загрузить JSON файл"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка при загрузке файла {file_path}: {str(e)}")
        return None

def check_4h_data_history(symbol):
    """Проверить исторические 4-часовые данные для символа"""
    file_path = f"historical_data/{symbol}_4h_data_history.json"
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не существует")
        return
    
    data = load_json_file(file_path)
    if not data:
        return
    
    print(f"\nФайл: {file_path}")
    print(f"Всего записей: {len(data)}")
    
    # Сортируем данные по time_value
    sorted_data = sorted(data, key=lambda x: x.get('time_value', 0))
    
    # Выводим первую запись
    if sorted_data:
        first_item = sorted_data[0]
        time_value = first_item.get('time_value')
        if time_value:
            date_time = datetime.datetime.fromtimestamp(time_value)
            print(f"Первая запись: {date_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Выводим последние 5 записей
    print("\nПоследние записи:")
    for item in sorted_data[-5:]:
        time_value = item.get('time_value')
        if time_value:
            date_time = datetime.datetime.fromtimestamp(time_value)
            print(f"- {date_time.strftime('%Y-%m-%d %H:%M:%S')} (timevalue: {time_value})")

def check_di_history(symbol):
    """Проверить историю DI индекса для символа"""
    file_path = f"historical_data/{symbol}_di_combined_history.json"
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не существует")
        return
    
    data = load_json_file(file_path)
    if not data:
        return
    
    print(f"\nФайл: {file_path}")
    print(f"Всего записей: {len(data)}")
    
    # Сортируем данные по timestamp
    sorted_data = sorted(data, key=lambda x: x.get('timestamp', 0))
    
    # Выводим первую запись
    if sorted_data:
        first_item = sorted_data[0]
        timestamp = first_item.get('timestamp')
        if timestamp:
            try:
                date_time = datetime.datetime.strptime(timestamp, '%Y-%m-%d')
                print(f"Первая запись: {date_time.strftime('%Y-%m-%d')}")
            except:
                print(f"Первая запись: {timestamp}")
    
    # Выводим последние 5 записей
    print("\nПоследние записи:")
    for item in sorted_data[-5:]:
        timestamp = item.get('timestamp')
        if timestamp:
            try:
                date_time = datetime.datetime.strptime(timestamp, '%Y-%m-%d')
                print(f"- {date_time.strftime('%Y-%m-%d')} (timestamp: {timestamp})")
            except:
                print(f"- {timestamp}")

def main():
    """Основная функция"""
    print("Проверка исторических данных DI индекса\n")
    
    symbols = ["BTC", "ETH", "XRP"]
    
    for symbol in symbols:
        check_4h_data_history(symbol)
        check_di_history(symbol)

if __name__ == "__main__":
    main()