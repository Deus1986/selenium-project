import time

import matplotlib.pyplot as plt
import numpy as np
import requests
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
        if item["amount24"] > 5000000:
            coins_array.append(item["symbol"])

    return coins_array


def find_head_shoulders_pattern(data, window=5, tolerance=0.015, min_distance=3):
    """
    Находит паттерн 'Голова и плечи' в данных формата MEXC

    Args:
        data: словарь с данными в формате MEXC
        window: окно для поиска локальных экстремумов
        tolerance: допуск для сравнения высот плеч (%)
        min_distance: минимальное расстояние между точками

    Returns:
        Список найденных паттернов
    """
    # Извлекаем high цены
    highs = np.array(data['data']['high'])
    times = np.array(data['data']['time'])

    # Находим локальные максимумы
    local_maxima = find_local_maxima(highs, window=window)

    patterns = []

    # Ищем паттерны среди локальных максимумов
    for i in range(len(local_maxima) - 4):
        try:
            # Индексы потенциальных точек паттерна: левое плечо, голова, правое плечо
            left_shoulder_idx = local_maxima[i]
            head_idx = local_maxima[i + 1]
            right_shoulder_idx = local_maxima[i + 2]

            # Проверяем расстояния между точками
            if (head_idx - left_shoulder_idx < min_distance or
                    right_shoulder_idx - head_idx < min_distance):
                continue

            # Цены точек
            left_shoulder_price = highs[left_shoulder_idx]
            head_price = highs[head_idx]
            right_shoulder_price = highs[right_shoulder_idx]

            # Проверяем условия паттерна:
            # 1. Голова выше плеч
            # 2. Плечи примерно на одном уровне
            # 3. Голова значительно выше плеч

            head_height_ratio = head_price / max(left_shoulder_price, right_shoulder_price)
            shoulders_diff = abs(left_shoulder_price - right_shoulder_price) / min(left_shoulder_price,
                                                                                   right_shoulder_price)

            if (head_height_ratio > 1.01 and  # Голова минимум на 1% выше
                    shoulders_diff <= tolerance and  # Плечи в пределах допуска
                    head_price > left_shoulder_price and
                    head_price > right_shoulder_price):

                # Находим точки для линии шеи (минимумы между плечами и головой)
                neckline_left_idx = find_neckline_point(highs, left_shoulder_idx, head_idx, 'min')
                neckline_right_idx = find_neckline_point(highs, head_idx, right_shoulder_idx, 'min')

                if neckline_left_idx is not None and neckline_right_idx is not None:
                    # Проверяем, что линия шеи имеет наклон (восходящий или нисходящий)
                    neckline_slope = (highs[neckline_right_idx] - highs[neckline_left_idx]) / (
                            neckline_right_idx - neckline_left_idx)

                    pattern = {
                        'left_shoulder_idx': left_shoulder_idx,
                        'head_idx': head_idx,
                        'right_shoulder_idx': right_shoulder_idx,
                        'left_shoulder_price': left_shoulder_price,
                        'head_price': head_price,
                        'right_shoulder_price': right_shoulder_price,
                        'neckline_left_idx': neckline_left_idx,
                        'neckline_right_idx': neckline_right_idx,
                        'neckline_left_price': highs[neckline_left_idx],
                        'neckline_right_price': highs[neckline_right_idx],
                        'neckline_slope': neckline_slope,
                        'pattern_height': head_price - ((highs[neckline_left_idx] + highs[neckline_right_idx]) / 2),
                        'shoulders_diff_percent': shoulders_diff * 100,
                        'left_shoulder_time': times[left_shoulder_idx],
                        'head_time': times[head_idx],
                        'right_shoulder_time': times[right_shoulder_idx]
                    }

                    patterns.append(pattern)

        except IndexError:
            continue

    return patterns


def find_local_maxima(prices, window=5):
    """Находит индексы локальных максимумов"""
    maxima = argrelextrema(prices, np.greater, order=window)[0]

    # Добавляем проверку границ
    if len(prices) > window:
        # Проверяем первый элемент
        if prices[0] > np.max(prices[1:window + 1]):
            maxima = np.append(maxima, 0)
        # Проверяем последний элемент
        if prices[-1] > np.max(prices[-window - 1:-1]):
            maxima = np.append(maxima, len(prices) - 1)

    return np.unique(maxima)


def find_local_minima(prices, window=5):
    """Находит индексы локальных минимумов"""
    minima = argrelextrema(prices, np.less, order=window)[0]
    return minima


def find_neckline_point(prices, start_idx, end_idx, mode='min'):
    """Находит точку для линии шеи между двумя индексами"""
    if start_idx >= end_idx or end_idx - start_idx < 2:
        return None

    if mode == 'min':
        # Ищем минимальное значение между точками
        min_idx = np.argmin(prices[start_idx:end_idx])
        return start_idx + min_idx
    else:
        # Ищем максимальное значение между точками
        max_idx = np.argmax(prices[start_idx:end_idx])
        return start_idx + max_idx


def plot_head_shoulders_pattern(data, pattern, coin):
    """
    Визуализирует паттерн 'Голова и плечи' для данных MEXC
    """
    highs = np.array(data['data']['high'])
    lows = np.array(data['data']['low'])
    times = np.array(data['data']['time'])

    fig, ax = plt.subplots(figsize=(16, 9))

    # Рисуем свечной график (упрощенный)
    for i in range(len(highs)):
        color = 'green' if data['data']['close'][i] > data['data']['open'][i] else 'red'
        ax.plot([i, i], [lows[i], highs[i]], color=color, linewidth=1.5, alpha=0.7)
        ax.plot(i, data['data']['close'][i], 'o', color=color, markersize=3, alpha=0.8)

    # Извлекаем индексы паттерна
    ls_idx = pattern['left_shoulder_idx']
    h_idx = pattern['head_idx']
    rs_idx = pattern['right_shoulder_idx']
    nl_idx = pattern['neckline_left_idx']
    nr_idx = pattern['neckline_right_idx']

    # Рисуем точки паттерна
    ax.plot(ls_idx, highs[ls_idx], 'ro', markersize=10, label='Левое плечo')
    ax.plot(h_idx, highs[h_idx], 'go', markersize=12, label='Голова')
    ax.plot(rs_idx, highs[rs_idx], 'ro', markersize=10, label='Правое плечo')
    ax.plot(nl_idx, highs[nl_idx], 'bo', markersize=8, label='Шея (лево)')
    ax.plot(nr_idx, highs[nr_idx], 'bo', markersize=8, label='Шея (право)')

    # Рисуем линии паттерна
    # Линия шеи
    ax.plot([nl_idx, nr_idx], [highs[nl_idx], highs[nr_idx]],
            'r--', linewidth=3, label='Линия шеи', alpha=0.8)

    # Соединяем плечи и голову
    ax.plot([ls_idx, h_idx, rs_idx],
            [highs[ls_idx], highs[h_idx], highs[rs_idx]],
            'g--', linewidth=2, alpha=0.6)

    # Добавляем информацию о паттерне
    info_text = f"""Паттерн Голова и Плечи:
Левое плечо: {highs[ls_idx]:.4f}
Голова: {highs[h_idx]:.4f} (+{(highs[h_idx] / highs[ls_idx] - 1) * 100:.2f}%)
Правое плечо: {highs[rs_idx]:.4f}
Разница плеч: {pattern['shoulders_diff_percent']:.2f}%
Высота паттерна: {pattern['pattern_height']:.4f}
Наклон шеи: {pattern['neckline_slope']:.6f}"""

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    # Настройки графика
    ax.set_title(f'Паттерн "Голова и плечи" - {coin}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Временные периоды')
    ax.set_ylabel('Цена')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Выделяем область паттерна
    pattern_start = min(ls_idx, nl_idx) - 2
    pattern_end = max(rs_idx, nr_idx) + 2
    ax.axvspan(pattern_start, pattern_end, alpha=0.1, color='yellow', label='Область паттерна')

    # Форматирование времени на оси X
    time_labels = [f"T+{i}" for i in range(len(times))]
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(time_labels, rotation=45)

    plt.tight_layout()
    plt.show()


def analyze_head_shoulders_patterns(data, patterns):
    """Анализирует и выводит информацию о найденных паттернах"""
    if not patterns:
        print("Паттерны 'Голова и плечи' не найдены")
        return

    print(f"Найдено паттернов 'Голова и плечи': {len(patterns)}")
    print("=" * 80)

    for i, pattern in enumerate(patterns, 1):
        print(f"Паттерн #{i}:")
        print(f"  Левое плечо: индекс {pattern['left_shoulder_idx']}, "
              f"цена {pattern['left_shoulder_price']:.4f}, "
              f"время {pattern['left_shoulder_time']}")
        print(f"  Голова:      индекс {pattern['head_idx']}, "
              f"цена {pattern['head_price']:.4f}, "
              f"время {pattern['head_time']}")
        print(f"  Правое плечо: индекс {pattern['right_shoulder_idx']}, "
              f"цена {pattern['right_shoulder_price']:.4f}, "
              f"время {pattern['right_shoulder_time']}")
        print(f"  Разница плеч: {pattern['shoulders_diff_percent']:.2f}%")
        print(f"  Высота паттерна: {pattern['pattern_height']:.4f}")
        print(f"  Наклон шеи: {pattern['neckline_slope']:.6f}")
        print("-" * 80)


# @pytest.mark.skip()
def test_find_head_shoulders():
    time_start = round(int(time.time()) - 20)
    time_end = time_start - (300 * 1 * 60)

    coins = get_24h_volume_usdt()
    print(len(coins))

    for i in range(len(coins)):
        returned_symbol_data = find_candles(coins[i], time_end, time_start)
        # assert False
        # print(coins[1])
        # print(volume)

        patterns = find_head_shoulders_pattern(returned_symbol_data)

        # Анализ и визуализация
        analyze_head_shoulders_patterns(returned_symbol_data, patterns)

        if patterns:
            # Визуализируем первый найденный паттерн
            plot_head_shoulders_pattern(returned_symbol_data, patterns[0], coins[i])
        else:
            print("Паттерны 'Голова и плечи' не найдены в данных")
        print(coins[i])
