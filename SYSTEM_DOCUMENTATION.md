# Документация системы анализа криптовалютного рынка

## 1. Общая архитектура

### 1.1 Основные компоненты
- **Backend (Python/Flask)**
  - Сбор данных с CryptoCompare API
  - Расчет индикаторов
  - API endpoints для фронтенда
- **Frontend (HTML/JavaScript)**
  - Интерактивные графики (Chart.js)
  - Таблицы с данными
  - Экспорт в Excel

### 1.2 Временные масштабы данных
- Weekly (недельные данные)
- Daily (дневные данные)
- 4h (4-часовые данные)

## 2. Сбор и обработка данных

### 2.1 Получение данных из CryptoCompare API

#### Weekly Data (get_weekly_data)
```python
- Получение daily данных
- Группировка по неделям (W-MON)
- Агрегация: open(first), high(max), low(min), close(last), volume(sum)
```

#### Daily Data (get_daily_data)
```python
- URL: histoday endpoint
- Параметры: limit=2000
- Фильтрация: только завершенные дни
- Кэширование: 2 часа (CACHE_DURATION = 7200)
```

#### 4h Data (get_4h_data)
```python
1. Первый запрос:
   - URL: histohour endpoint
   - Параметры: aggregate=4, limit=2000
   - Получение текущего периода

2. Второй запрос:
   - Использует toTs из первого запроса
   - Получение предыдущего периода
   
3. Обработка данных:
   - Объединение данных из обоих запросов
   - Сортировка по времени
   - Удаление дубликатов
   - Группировка по дате и часу
```

## 3. Расчет индикаторов

### 3.1 Компоненты DI Index

#### MA Index
```python
- micro = EMA(close, 6)
- short = EMA(close, 13)
- medium = SMA(close, 30)
- long = SMA(close, 200)

Условия:
- MA_bull = micro > short
- MA_bull1 = short > medium
- MA_bull2 = short > long
- MA_bull3 = medium > long

MA_index = MA_bull + MA_bull1 + MA_bull2 + MA_bull3
```

#### Willy Index
```python
length = 21
len_out = 13

Расчет:
- upper = rolling_max(high, length)
- lower = rolling_min(low, length)
- out = 100 * (close - upper) / (upper - lower)
- out2 = EMA(out, len_out)

Компоненты:
- Willy_stupid_os = out2 < -80
- Willy_stupid_ob = out2 > -20
- Willy_bullbear = out > out2
- Willy_bias = out > -50

Willy_index = Willy_stupid_os + Willy_bullbear + Willy_bias - Willy_stupid_ob
```

#### MACD Index
```python
Параметры:
- fastLength = 12
- slowLength = 26
- signalLength = 9

Расчет:
- fastMA = EMA(close, fastLength)
- slowMA = EMA(close, slowLength)
- macd = fastMA - slowMA
- signal = SMA(macd, signalLength)

Компоненты:
- macd_bullbear = macd > signal
- macd_bias = macd > 0

MACD_index = macd_bullbear + macd_bias
```

### 3.2 Итоговый DI Index
```python
1. Расчет компонентов для каждого таймфрейма
2. Суммирование всех компонентов
3. Округление до целых чисел
4. Распределение по таймфреймам (weekly_di_new, daily_di_new, 4h_di_new)
```

## 4. Фронтенд

### 4.1 Основные компоненты интерфейса
- Селектор криптовалют (мультивыбор)
- Вкладки для каждой монеты
- График тренда
- Таблица с данными
- Модальное окно для 4h значений

### 4.2 Графики
```javascript
- Chart.js для визуализации
- Отображение:
  * Цена закрытия
  * 13 EMA
  * 30 SMA
  * Тренд (bull/bear)
```

### 4.3 Таблицы
```javascript
Колонки:
- Дата
- DI Компоненты (Weekly, Daily, 4h)
- Расчет (Total, 13 EMA, 30 SMA, Trend)
- Close (Daily)
```

## 5. Кэширование и оптимизация

### 5.1 Кэширование данных
```python
CACHE_DURATION = 7200  # 2 часа
MAX_WORKERS = 5  # Максимум параллельных запросов

Структура кэша:
{
    'symbol_datatype': {
        'data': DataFrame,
        'time': timestamp
    }
}
```

### 5.2 Параллельная обработка
```python
- ThreadPoolExecutor для параллельных запросов
- Задержка 500ms между запросами к API
- Обработка ошибок и повторные попытки
```

## 6. API Endpoints

### 6.1 /api/di_index
```python
Параметры:
- symbols: список криптовалют через запятую
- debug: флаг для отладочной информации

Ответ:
{
    "BTC": [...],  # Массив данных по каждой монете
    "ETH": [...],
    ...
}
```

## 7. Обработка ошибок

### 7.1 Основные типы ошибок
```python
- Ошибки API (неверный ответ, таймаут)
- Ошибки валидации данных
- Ошибки расчета индикаторов
```

### 7.2 Стратегия обработки
```python
- Логирование всех ошибок
- Повторные попытки для API запросов
- Возврат информативных сообщений об ошибках
```
