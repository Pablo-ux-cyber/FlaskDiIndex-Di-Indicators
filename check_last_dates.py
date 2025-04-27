#!/usr/bin/env python3
"""
Скрипт для проверки последних дат в историческом файле DI индекса
"""

import json
import os
from datetime import datetime

def get_latest_date(symbol="BTC"):
    """Получить последние даты в историческом файле"""
    history_file = f"historical_data/{symbol}_di_combined_history.json"
    
    if not os.path.exists(history_file):
        print(f"Файл {history_file} не найден")
        return
    
    try:
        with open(history_file, 'r') as f:
            data = json.load(f)
        
        # В этом формате dates будут ключами словаря
        dates = list(data.keys())
        
        # Сортировка дат
        dates.sort()
        
        # Вывод последних 10 дат
        print(f"Последние 10 дат в файле {history_file}:")
        for date in dates[-10:]:
            daily_value = data[date].get('daily_di_new')
            print(f" - {date}: DI index = {daily_value}")
            
            # Проверяем 4-часовые значения
            four_hour_values = data[date].get('4h_values_new', [])
            if four_hour_values:
                print(f"   4-часовые значения ({len(four_hour_values)}):")
                for val in four_hour_values:
                    print(f"   -- {val.get('time')}: {val.get('value_new')}")
        
        # Показать сегодняшнюю дату для сравнения
        today = datetime.now().strftime('%Y-%m-%d')
        print(f"\nСегодняшняя дата: {today}")
        
        # Показать последнюю дату в файле
        if dates:
            last_date = dates[-1]
            print(f"Последняя дата в файле: {last_date}")
            
            # Проверка, есть ли вчерашняя дата
            yesterday = (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - 
                        datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            if yesterday in dates:
                print(f"Данные за вчера ({yesterday}) присутствуют в файле")
            else:
                print(f"Данные за вчера ({yesterday}) отсутствуют в файле")
    
    except Exception as e:
        print(f"Ошибка при обработке файла: {str(e)}")

if __name__ == "__main__":
    print("Проверка последних дат в историческом файле DI индекса\n")
    get_latest_date("BTC")