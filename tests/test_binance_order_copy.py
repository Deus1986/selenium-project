import datetime
import hashlib
import hmac
import time

import pytest
import requests

@pytest.mark.skip()
def test_order():
    api_key = 'isQR14OjrXeyl8Kzp1IJdXFw0hY7fli1cceJGpWeTAzFATn80hE5LIqe2N3fnU6P'  # Замените на ваш API ключ
    secret_key = 'n9ryVubCLYgBLEK4xdozIW9ShuOPal4v7ZgpabAC7E5vcM0AvLX7xumIIH3XpZwR'  # Ваш секретный ключ

    symbol = 'IOSTUSDT'  # !!!!!
    side = 'SELL'  # !!!!!!
    order_type = 'MARKET'
    quantity = 2700
    recv_window = 5000

    # date = datetime.datetime.now()
    # current_year = date.year
    # current_month = date.month
    # current_day = date.day
    #
    # target_time = datetime.datetime(current_year, current_month, current_day, 7, 6, 58, 500000)
    # target_time_2 = datetime.datetime(current_year, current_month, current_day, 7, 7, 1, 250001)

    # while date <= target_time:
    #     date = datetime.datetime.now()
    # else:
    #     print(f"время срабатывания таргет 1 {date}")
    #
    # timestamp = int(time.time() * 1000)
    #
    # query_string = f'symbol={symbol}&side={side}&type={order_type}&quantity={quantity}&recvWindow={recv_window}&timestamp={timestamp}'
    #
    # signature = hmac.new(secret_key.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    #
    # url = f'https://fapi.binance.com/fapi/v1/order?{query_string}&signature={signature}'

    headers = {
        'X-MBX-APIKEY': api_key
    }

    # response = requests.post(url, headers=headers)
    # print(datetime.datetime.now())
    # print(response.json())
    # order_id = response.json().get("clientOrderId")
    # print(order_id)

    # while date <= target_time_2:
    #     date = datetime.datetime.now()
    # else:
    #     print(f"время срабатывания таргет 2 {date}")

    timestamp = int(time.time() * 1000)
    query_string = f'symbol={symbol}&orderid=8069832376&timestamp={timestamp}'
    signature = hmac.new(secret_key.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

    url = f'https://fapi.binance.com/fapi/v1/order?{query_string}&signature={signature}'

    response = requests.delete(url, headers=headers)
    print(datetime.datetime.now())
    print(response.json())
    print(response.url)
    print(response.status_code)
