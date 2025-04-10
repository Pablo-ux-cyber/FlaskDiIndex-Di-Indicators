import requests
import json

def main():
    # Проверяем GET запрос для первой монеты
    symbol = "BTC"
    url = f"http://imm2.dinet.fvds.ru/di_index_new/api/coins?symbol={symbol}"
    
    print(f"Делаем запрос к: {url}")
    try:
        response = requests.get(url)
        print(f"Статус ответа: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Ответ: {json.dumps(data, indent=2)}")
        else:
            print(f"Ошибка: {response.text}")
    except Exception as e:
        print(f"Исключение: {str(e)}")

if __name__ == "__main__":
    main()
