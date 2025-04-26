#!/usr/bin/env python3
"""
Скрипт для тестирования отображения исторических 4-часовых данных.
Проверяет наличие 4h данных для разных временных периодов.
"""

import requests
import json
import argparse
from datetime import datetime, timedelta

def format_date(date_str):
    """Форматирует дату для вывода"""
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        return date.strftime("%d %b %Y")
    except:
        return date_str

def check_period(data, start_date, end_date, period_name):
    """Проверяет наличие 4h данных для указанного периода"""
    print(f"\n\033[1m{period_name} ({start_date} - {end_date}):\033[0m")
    
    # Преобразуем строковые даты в объекты datetime
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Считаем статистику
    total_days = 0
    days_with_4h = 0
    days_without_4h = 0
    missing_days = 0
    
    # Создаем список всех дней в периоде
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        total_days += 1
        
        # Ищем дату в данных
        entry = next((item for item in data if item.get("time") == date_str), None)
        
        if entry:
            # Проверяем наличие 4h данных
            if "4h_values_new" in entry and isinstance(entry["4h_values_new"], list) and len(entry["4h_values_new"]) > 0:
                days_with_4h += 1
            else:
                days_without_4h += 1
                print(f"  - \033[33mНет 4h данных\033[0m для {format_date(date_str)}")
        else:
            missing_days += 1
            print(f"  - \033[31mНет данных\033[0m для {format_date(date_str)}")
        
        current += timedelta(days=1)
    
    # Выводим статистику
    print(f"\nСтатистика периода ({period_name}):")
    print(f"  Всего дней: {total_days}")
    print(f"  Дней с 4h данными: {days_with_4h} ({days_with_4h/total_days*100:.1f}%)")
    print(f"  Дней без 4h данных: {days_without_4h} ({days_without_4h/total_days*100:.1f}%)")
    print(f"  Отсутствующих дней: {missing_days} ({missing_days/total_days*100:.1f}%)")
    
    return days_with_4h, days_without_4h, missing_days

def main():
    """Основная функция скрипта"""
    parser = argparse.ArgumentParser(description="Тестирование исторических 4h данных DI индекса")
    parser.add_argument("--symbol", default="BTC", help="Символ криптовалюты (по умолчанию BTC)")
    parser.add_argument("--port", default=5000, type=int, help="Порт сервера (по умолчанию 5000)")
    parser.add_argument("--use-history", default="true", choices=["true", "false"], 
                        help="Использовать исторические данные (по умолчанию true)")
    args = parser.parse_args()
    
    # Формируем URL для API
    base_url = f"http://localhost:{args.port}/api/di_index"
    url = f"{base_url}?symbols={args.symbol}&use_history={args.use_history}"
    
    print(f"Запрос данных от API: {url}")
    
    try:
        # Делаем запрос к API
        response = requests.get(url)
        response.raise_for_status()
        
        # Разбираем ответ
        data = response.json()
        
        if args.symbol not in data:
            print(f"Ошибка: символ {args.symbol} не найден в ответе API")
            return
        
        symbol_data = data[args.symbol]
        
        # Общая информация
        print(f"\n\033[1mДанные для {args.symbol}:\033[0m")
        print(f"Всего записей: {len(symbol_data)}")
        
        # Проверяем первую и последнюю запись
        if len(symbol_data) > 0:
            first_entry = symbol_data[-1]  # Самая старая запись
            last_entry = symbol_data[0]    # Самая новая запись
            
            print(f"Период данных: с {format_date(first_entry.get('time'))} по {format_date(last_entry.get('time'))}")
            
            # Проверка наличия 4h данных в первой и последней записи
            print("\nПроверка крайних точек:")
            if "4h_values_new" in first_entry and isinstance(first_entry["4h_values_new"], list) and len(first_entry["4h_values_new"]) > 0:
                print(f"Самая старая запись ({format_date(first_entry.get('time'))}): \033[32mЕсть 4h данные\033[0m")
            else:
                print(f"Самая старая запись ({format_date(first_entry.get('time'))}): \033[31mНет 4h данных\033[0m")
                
            if "4h_values_new" in last_entry and isinstance(last_entry["4h_values_new"], list) and len(last_entry["4h_values_new"]) > 0:
                print(f"Самая новая запись ({format_date(last_entry.get('time'))}): \033[32mЕсть 4h данные\033[0m")
            else:
                print(f"Самая новая запись ({format_date(last_entry.get('time'))}): \033[31mНет 4h данных\033[0m")
            
            # Проверяем наличие данных для разных периодов
            # Текущий период
            current_period = check_period(
                symbol_data, 
                (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                datetime.now().strftime("%Y-%m-%d"),
                "Последний месяц"
            )
            
            # Средний период
            middle_period = check_period(
                symbol_data,
                "2024-12-01",
                "2024-12-31",
                "Декабрь 2024"
            )
            
            # Ранний период
            early_period = check_period(
                symbol_data,
                "2023-05-01",
                "2023-05-31",
                "Май 2023"
            )
            
            # Самый ранний период
            earliest_period = check_period(
                symbol_data,
                "2019-11-01",
                "2019-11-30",
                "Ноябрь 2019"
            )
            
            print("\n\033[1mВЫВОД:\033[0m")
            # Проверяем только данные за последние 6 месяцев
            if middle_period[0] > 0 and current_period[0] > 0:
                print("\033[32mУСПЕХ: 4-часовые данные отображаются для недавних периодов!\033[0m")
                if early_period[0] == 0 or earliest_period[0] == 0:
                    print("\033[33mПРИМЕЧАНИЕ: Для старых периодов (до ноября 2024) 4-часовые данные отсутствуют — это нормально, т.к. мы сохраняем только реальные данные из API.\033[0m")
            else:
                print("\033[31mПРОБЛЕМА: 4-часовые данные отсутствуют в недавних периодах.\033[0m")
        
    except requests.RequestException as e:
        print(f"Ошибка при запросе к API: {e}")
    except Exception as e:
        print(f"Ошибка при выполнении скрипта: {e}")

if __name__ == "__main__":
    main()