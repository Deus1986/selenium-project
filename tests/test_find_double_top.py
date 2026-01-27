import time

import numpy as np
from scipy.signal import argrelextrema


def find_candles(symbol, start_time, end_time):
    params = {
        "interval": "Min1",
        "start": start_time,
        "end": f"{end_time}"
    }
    response = requests.get(f"https://contract.mexc.com/api/v1/contract/kline/{symbol}", params=params)
    assert response.status_code == 200
    print(response.json())
    return response.json()


def get_futures_coins():
    url = "https://contract.mexc.com/api/v1/contract/detail"
    response = requests.get(url)
    data = response.json()
    return data


def find_double_top_pattern(data, window=5, tolerance=0.01, min_distance=3):
    """
    Находит паттерн двойная вершина в данных свечей

    Args:
        data: словарь с данными OHLCV
        window: окно для поиска локальных экстремумов
        tolerance: допуск для сравнения высот вершин (%)
        min_distance: минимальное расстояние между вершинами (баров)

    Returns:
        Список найденных паттернов двойной вершины
    """
    # Извлекаем цены high и временные метки
    highs = np.array(data['data']['high'])
    times = np.array(data['data']['time'])

    # Находим локальные максимумы
    local_maxima = find_local_maxima(highs, window=window)

    patterns = []

    # Ищем пары вершин, образующие двойную вершину
    for i in range(len(local_maxima)):
        for j in range(i + 1, len(local_maxima)):
            idx1 = local_maxima[i]
            idx2 = local_maxima[j]

            # Проверяем расстояние между вершинами
            if abs(idx2 - idx1) < min_distance:
                continue

            # Проверяем схожесть высот вершин
            price1 = highs[idx1]
            price2 = highs[idx2]
            price_diff = abs(price1 - price2) / min(price1, price2)

            if price_diff <= tolerance:
                # Проверяем наличие "шеи" - минимума между вершинами
                neckline_idx = find_neckline(highs, idx1, idx2)

                if neckline_idx is not None:
                    pattern = {
                        'first_top_index': idx1,
                        'second_top_index': idx2,
                        'first_top_price': price1,
                        'second_top_price': price2,
                        'first_top_time': times[idx1],
                        'second_top_time': times[idx2],
                        'neckline_index': neckline_idx,
                        'neckline_price': highs[neckline_idx],
                        'pattern_height': min(price1, price2) - highs[neckline_idx],
                        'time_between_tops': abs(times[idx2] - times[idx1])
                    }
                    patterns.append(pattern)

    return patterns


def find_local_maxima(prices, window=5):
    """
    Находит индексы локальных максимумов
    """
    # Используем scipy для поиска локальных максимумов
    maxima = argrelextrema(prices, np.greater, order=window)[0]

    # Также проверяем начало и конец массива
    if len(prices) > window:
        # Проверяем первые window элементов
        start_max = np.argmax(prices[:window])
        if start_max == 0:
            maxima = np.append(maxima, 0)

        # Проверяем последние window элементов
        end_max = np.argmax(prices[-window:]) + len(prices) - window
        if end_max == len(prices) - 1:
            maxima = np.append(maxima, len(prices) - 1)

    return np.unique(maxima)


def find_neckline(prices, start_idx, end_idx):
    """
    Находит минимальное значение (шею) между двумя вершинами
    """
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    # Ищем минимум между вершинами
    neckline_idx = np.argmin(prices[start_idx:end_idx + 1]) + start_idx

    # Убедимся, что это действительно минимум (проверим соседние значения)
    if (neckline_idx > 0 and prices[neckline_idx] >= prices[neckline_idx - 1]) or \
            (neckline_idx < len(prices) - 1 and prices[neckline_idx] >= prices[neckline_idx + 1]):
        return None

    return neckline_idx


def analyze_double_top_patterns(data):
    """
    Анализирует данные и выводит информацию о найденных паттернах
    """
    patterns = find_double_top_pattern(data)

    print(f"Найдено паттернов двойной вершины: {len(patterns)}")
    print("-" * 80)

    for i, pattern in enumerate(patterns, 1):
        print(f"Паттерн #{i}:")
        print(f"  Первая вершина: индекс {pattern['first_top_index']}, "
              f"цена {pattern['first_top_price']:.4f}, "
              f"время {pattern['first_top_time']}")
        print(f"  Вторая вершина: индекс {pattern['second_top_index']}, "
              f"цена {pattern['second_top_price']:.4f}, "
              f"время {pattern['second_top_time']}")
        print(f"  Разница в цене: {abs(pattern['first_top_price'] - pattern['second_top_price']):.4f} "
              f"({abs(pattern['first_top_price'] - pattern['second_top_price']) / min(pattern['first_top_price'], pattern['second_top_price']) * 100:.2f}%)")
        print(f"  Шея: индекс {pattern['neckline_index']}, "
              f"цена {pattern['neckline_price']:.4f}")
        print(f"  Высота паттерна: {pattern['pattern_height']:.4f}")
        print(f"  Время между вершинами: {pattern['time_between_tops']} секунд")
        print("-" * 80)

    return patterns


import requests


def get_24h_volume_usdt():
    """
    Получает объем торгов в USDT за 24 часа

    Args:
        symbol: название монеты (например 'BTC')
        is_futures: True для фьючерсов, False для спота
    """

    # Для фьючерсов
    url = f"https://contract.mexc.com/api/v1/contract/ticker"
    response = requests.get(url)
    data = response.json()
    coins_array = []

    for item in data["data"]:
        if item["amount24"] > 50000000:
            coins_array.append(item["symbol"])

    return coins_array


# @pytest.mark.skip()
def test_find_double_top():
    time_start = round(int(time.time()) - 20)
    # time_end = round(int(time.time()))
    time_end = time_start - (15 * 1 * 60)

    coins = get_24h_volume_usdt()

    for i in range(len(coins)):
        returned_symbol_data = find_candles(coins[i], time_end, time_start)

        pattern = analyze_double_top_patterns(returned_symbol_data)
        if len(pattern) > 0:
            print(coins[i])
