# Руководство по реализации системы анализа криптовалютного рынка

## 1. Архитектура системы

### 1.1 Основные компоненты
```
Backend (Python/Flask)
├── server.py - основной файл с бизнес-логикой
└── templates/
    └── index.html - пользовательский интерфейс
```

### 1.2 Ключевые технологии
- Backend: Python 3.11, Flask
- Frontend: HTML, JavaScript, Chart.js
- API: CryptoCompare
- Библиотеки: pandas, pandas-ta (технические индикаторы)

## 2. Работа с API и данными

### 2.1 Получение данных
Система работает с тремя временными масштабами:

#### Weekly Data
```python
def get_weekly_data(symbol="BTC", tsym="USD", limit=2000):
    # 1. Получаем daily данные
    # 2. Группируем по неделям (W-MON)
    # 3. Агрегируем данные:
    #    - open: первое значение
    #    - high: максимум
    #    - low: минимум
    #    - close: последнее значение
    #    - volume: сумма
```

#### Daily Data
```python
def get_daily_data(symbol="BTC", tsym="USD", limit=2000):
    # 1. URL: histoday endpoint
    # 2. Параметры: limit=2000
    # 3. Фильтрация: только завершенные дни
    # 4. Кэширование: 2 часа
```

#### 4h Data
```python
def get_4h_data(symbol="BTC", tsym="USD", limit=2000):
    # 1. Первый запрос (текущий период):
    #    URL: histohour endpoint
    #    Параметры: aggregate=4, limit=2000
    
    # 2. Второй запрос (предыдущий период):
    #    Использует toTs из первого запроса
    #    Получение предыдущего периода
    
    # 3. Объединение и обработка данных:
    #    - Объединение данных из обоих запросов
    #    - Сортировка по времени
    #    - Удаление дубликатов
    #    - Группировка по дате и часу
```

### 2.2 Кэширование
```python
CACHE_DURATION = 7200  # 2 часа
cache_key = f"{symbol}_{data_type}"
cached = CACHE.get(cache_key)
```

## 3. Расчет индикаторов

### 3.1 DI Index Components

#### MA Index
```python
def calculate_ma_index(df):
    # Компоненты:
    micro = EMA(close, 6)
    short = EMA(close, 13)
    medium = SMA(close, 30)
    long = SMA(close, 200)
    
    # Условия:
    MA_bull = micro > short
    MA_bull1 = short > medium
    MA_bull2 = short > long
    MA_bull3 = medium > long
    
    # Итоговый индекс:
    MA_index = MA_bull + MA_bull1 + MA_bull2 + MA_bull3
```

#### Willy Index
```python
def calculate_willy_index(df):
    # Параметры:
    length = 21
    len_out = 13
    
    # Расчет:
    upper = rolling_max(high, length)
    lower = rolling_min(low, length)
    out = 100 * (close - upper) / (upper - lower)
    out2 = EMA(out, len_out)
    
    # Компоненты:
    Willy_stupid_os = out2 < -80
    Willy_stupid_ob = out2 > -20
    Willy_bullbear = out > out2
    Willy_bias = out > -50
    
    # Итоговый индекс:
    Willy_index = Willy_stupid_os + Willy_bullbear + Willy_bias - Willy_stupid_ob
```

### 3.2 Итоговый DI Index
```python
def calculate_di_index(df):
    # 1. Расчет всех компонентов
    MA_index = calculate_ma_index(df)
    Willy_index = calculate_willy_index(df)
    MACD_index = calculate_macd_index(df)
    OBV_index = calculate_obv_index(df)
    MFI_index = calculate_mfi_index(df)
    AD_index = calculate_ad_index(df)
    
    # 2. Суммирование
    di_value = MA_index + Willy_index + MACD_index + OBV_index + MFI_index + AD_index
    
    # 3. Распределение по временным масштабам
    if timeframe == "weekly":
        df["weekly_di_new"] = di_value
    elif timeframe == "daily":
        df["daily_di_new"] = di_value
    else:  # 4h
        df["4h_di_new"] = di_value
```

## 4. Frontend Implementation

### 4.1 Компоненты интерфейса
```javascript
// Основные элементы:
1. Селектор криптовалют (мультивыбор)
2. Вкладки для каждой монеты
3. График тренда
4. Таблица с данными
5. Модальное окно для 4h значений

// Структура данных в таблице:
{
    time: "YYYY-MM-DD",
    weekly_di_new: value,
    daily_di_new: value,
    "4h_di_new": value,
    total_new: value,
    di_ema_13_new: value,
    di_sma_30_new: value,
    trend_new: "bull"/"bear",
    close: value
}
```

### 4.2 Визуализация данных
```javascript
// График тренда
new Chart(canvas, {
    type: 'line',
    data: {
        datasets: [
            {
                label: 'Close Price',
                data: closeValues,
                borderColor: 'rgb(75, 192, 192)'
            },
            {
                label: '13 EMA',
                data: ema13Values,
                borderColor: 'rgb(255, 99, 132)'
            },
            {
                label: '30 SMA',
                data: sma30Values,
                borderColor: 'rgb(54, 162, 235)'
            }
        ]
    }
});
```

## 5. Пример использования

### 5.1 Запуск сервера
```python
# main.py
from server import di_index_blueprint
app = Flask(__name__)
app.register_blueprint(di_index_blueprint)
app.run(host="0.0.0.0", port=5000)
```

### 5.2 API endpoints
```
GET /api/di_index?symbols=BTC,ETH
Параметры:
- symbols: список криптовалют через запятую
- debug: флаг для отладочной информации

Ответ:
{
    "BTC": [...],  # Массив данных по каждой монете
    "ETH": [...],
}
```

### 5.3 Параллельная обработка
```python
def process_symbol_batch(symbols, debug=False):
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Добавляем задержку между запросами
        def process_with_delay(symbol):
            time.sleep(0.5)  # 500ms задержка
            return process_symbol(symbol, debug)
```

## 6. Важные заметки

1. Временные масштабы данных:
   - Weekly: агрегированные недельные данные
   - Daily: данные за каждый день
   - 4h: данные за каждые 4 часа (6 значений в день)

2. Кэширование:
   - Длительность кэша: 2 часа
   - Отдельный кэш для каждого типа данных и символа

3. API ограничения:
   - Используйте задержки между запросами (500ms)
   - Limit=2000 для каждого запроса
   - Для 4h данных делайте два последовательных запроса

4. Расчет трендов:
   - bull: 13 EMA > 30 SMA
   - bear: 13 EMA < 30 SMA

5. Отображение данных:
   - 4h значения показываются за 20:00 UTC
   - При клике показываются все 6 значений за день
