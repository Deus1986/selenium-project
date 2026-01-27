import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import requests


def find_candles(symbol, start_time, end_time):
    params = {
        "interval": "Min60",
        "start": start_time,
        "end": f"{end_time}"
    }
    response = requests.get(f"https://contract.mexc.com/api/v1/contract/kline/{symbol}", params=params)
    assert response.status_code == 200
    print(response.json())
    return response.json()


def get_futures_coins():
    """
    Получает список фьючерсных монет
    """
    url = "https://contract.mexc.com/api/v1/contract/detail"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        return data
    except Exception as e:
        print(f"Ошибка получения списка монет: {e}")
        return None


def get_24h_volume_usdt(min_volume=50000000):
    """
    Получает объем торгов в USDT за 24 часа
    """
    url = f"https://contract.mexc.com/api/v1/contract/ticker"
    response = requests.get(url)
    data = response.json()
    coins_array = []

    for item in data["data"]:
        if item["amount24"] > min_volume:
            coins_array.append(item["symbol"])

    return coins_array

def find_local_maxima(prices, window=5):
    """
    Находит индексы локальных максимумов
    """
    maxima = argrelextrema(prices, np.greater, order=window)[0]

    if len(prices) > window:
        if prices[0] > np.max(prices[1:window + 1]):
            maxima = np.append(maxima, 0)
        if prices[-1] > np.max(prices[-window - 1:-1]):
            maxima = np.append(maxima, len(prices) - 1)

    return np.unique(maxima)


def find_neckline(prices, start_idx, end_idx):
    """
    Находит минимальное значение (шею) между двумя вершинами
    """
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    if end_idx - start_idx < 2:
        return None

    neckline_idx = np.argmin(prices[start_idx:end_idx + 1]) + start_idx
    return neckline_idx


def find_double_top_pattern(data, window=5, tolerance=0.015, min_distance=3):
    """
    Находит паттерн двойная вершина
    """
    if not data or not data.get('success'):
        return []

    highs = np.array(data['data']['high'])
    lows = np.array(data['data']['low'])
    times = np.array(data['data']['time'])

    local_maxima = find_local_maxima(highs, window=window)
    patterns = []

    for i in range(len(local_maxima)):
        for j in range(i + 1, len(local_maxima)):
            idx1 = local_maxima[i]
            idx2 = local_maxima[j]

            if abs(idx2 - idx1) < min_distance:
                continue

            price1 = highs[idx1]
            price2 = highs[idx2]
            price_diff = abs(price1 - price2) / min(price1, price2)

            if price_diff <= tolerance:
                neckline_idx = find_neckline(lows, idx1, idx2)  # Используем lows для шеи

                if neckline_idx is not None:
                    pattern = {
                        'first_top_index': idx1,
                        'second_top_index': idx2,
                        'first_top_price': price1,
                        'second_top_price': price2,
                        'neckline_index': neckline_idx,
                        'neckline_price': lows[neckline_idx],  # Цена шеи из lows
                        'pattern_height': min(price1, price2) - lows[neckline_idx],
                        'completion_index': neckline_idx + (neckline_idx - idx2)  # Проекция завершения
                    }
                    patterns.append(pattern)

    return patterns


def plot_pattern_with_entry_points(data, pattern, symbol):
    """
    Отрисовывает график с паттерном и точками входа
    """
    if not data or not pattern:
        return

    highs = np.array(data['data']['high'])
    lows = np.array(data['data']['low'])
    closes = np.array(data['data']['close'])
    opens = np.array(data['data']['open'])

    fig, ax = plt.subplots(figsize=(15, 10))

    # Рисуем свечной график
    for i in range(len(highs)):
        color = 'green' if closes[i] > opens[i] else 'red'
        ax.plot([i, i], [lows[i], highs[i]], color=color, linewidth=2, alpha=0.8)
        ax.plot(i, closes[i], 'o', color=color, markersize=4, alpha=0.8)

    # Извлекаем точки паттерна
    idx1 = pattern['first_top_index']
    idx2 = pattern['second_top_index']
    neck_idx = pattern['neckline_index']

    # Рисуем паттерн
    ax.plot(idx1, highs[idx1], 'ro', markersize=10, label='Первая вершина')
    ax.plot(idx2, highs[idx2], 'ro', markersize=10, label='Вторая вершина')
    ax.plot(neck_idx, lows[neck_idx], 'bo', markersize=10, label='Шея')

    # Линия шеи
    ax.axhline(y=pattern['neckline_price'], color='blue', linestyle='--',
               alpha=0.7, label='Линия шеи')

    # Точки входа в сделку
    entry_short = pattern['neckline_price'] - (pattern['pattern_height'] * 0.1)  # Пробитие шеи
    entry_long = pattern['neckline_price'] - (pattern['pattern_height'] * 0.5)  # Отскок от цели

    # Целевые уровни
    target_short = pattern['neckline_price'] - pattern['pattern_height']  # Цель для шорта
    target_long = pattern['neckline_price'] + (pattern['pattern_height'] * 0.5)  # Цель для лонга

    # Стоп-лоссы
    stop_short = pattern['neckline_price'] + (pattern['pattern_height'] * 0.1)  # Стоп для шорта
    stop_long = pattern['neckline_price'] - (pattern['pattern_height'] * 0.1)  # Стоп для лонга

    # Рисуем точки входа и цели
    entry_time = neck_idx + 1
    if entry_time < len(highs):
        # Точка входа в шорт (пробитие шеи)
        ax.plot(entry_time, entry_short, 'v', color='red', markersize=12,
                label='Вход в шорт', markeredgewidth=2, markeredgecolor='black')

        # Цель для шорта
        ax.axhline(y=target_short, color='red', linestyle=':', alpha=0.7,
                   label='Цель шорта')

        # Стоп для шорта
        ax.axhline(y=stop_short, color='orange', linestyle=':', alpha=0.7,
                   label='Стоп шорта')

        # Точка входа в лонг (отскок от цели)
        ax.plot(entry_time, entry_long, '^', color='green', markersize=12,
                label='Вход в лонг', markeredgewidth=2, markeredgecolor='black')

        # Цель для лонга
        ax.axhline(y=target_long, color='green', linestyle=':', alpha=0.7,
                   label='Цель лонга')

        # Стоп для лонга
        ax.axhline(y=stop_long, color='orange', linestyle=':', alpha=0.7,
                   label='Стоп лонга')

    # Информация о сделке
    info_text = f"""Паттерн Двойная Вершина - {symbol}

Торговые сигналы:

SHORT (медвежий):
• Вход: {entry_short:.4f} (пробитие шеи)
• Цель: {target_short:.4f} (R:R = 1:1)
• Стоп: {stop_short:.4f}

LONG (отскок):
• Вход: {entry_long:.4f} (отскок от цели)
• Цель: {target_long:.4f} (R:R = 1:1)
• Стоп: {stop_long:.4f}

Параметры паттерна:
• Высота: {pattern['pattern_height']:.4f}
• Вершины: {pattern['first_top_price']:.4f} / {pattern['second_top_price']:.4f}
• Шея: {pattern['neckline_price']:.4f}"""

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

    # Настройки графика
    ax.set_title(f'Двойная вершина - {symbol} - Точки входа', fontsize=16, fontweight='bold')
    ax.set_xlabel('Временные периоды')
    ax.set_ylabel('Цена')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return {
        'short_entry': entry_short,
        'short_target': target_short,
        'short_stop': stop_short,
        'long_entry': entry_long,
        'long_target': target_long,
        'long_stop': stop_long
    }


def analyze_double_top_patterns(data, symbol):
    """
    Анализирует паттерны и возвращает торговые сигналы
    """
    patterns = find_double_top_pattern(data)

    if not patterns:
        print(f"Для {symbol} паттерны не найдены")
        return None

    print(f"\n🎯 Найдено паттернов для {symbol}: {len(patterns)}")

    trade_signals = []

    for i, pattern in enumerate(patterns, 1):
        print(f"\nПаттерн #{i}:")
        print(f"  Вершины: {pattern['first_top_price']:.4f} / {pattern['second_top_price']:.4f}")
        print(f"  Шея: {pattern['neckline_price']:.4f}")
        print(f"  Высота: {pattern['pattern_height']:.4f}")

        # Получаем торговые сигналы
        signals = plot_pattern_with_entry_points(data, pattern, symbol)
        trade_signals.append(signals)

    return trade_signals


def test_find_double_top():
    """
    Основная функция тестирования
    """
    # Временной диапазон (последние 4 часа)
    # time_end = int(time.time() * 1000)
    # time_start = time_end - (8 * 60 * 60 * 1000)  # 4 часа назад
    time_start = round(int(time.time()) - 20)
    time_end = time_start - (20 * 60 * 60)

    coins = get_24h_volume_usdt(min_volume=20000000)  # Минимум 100M объема

    print(f"Анализируем {len(coins)} монет с высоким объемом...")

    found_patterns = []

    for symbol in coins:  # Анализируем первые 5 монет для скорости
        print(f"\n🔍 Анализируем {symbol}...")

        data = find_candles(symbol, time_end, time_start)

        if data and data.get('success'):
            signals = analyze_double_top_patterns(data, symbol)
            if signals:
                found_patterns.append({'symbol': symbol, 'signals': signals})

        time.sleep(1)  # Пауза между запросами
        # assert False
    # Вывод результатов
    if found_patterns:
        print(f"\n🎉 Найдены паттерны для {len(found_patterns)} монет:")
        for item in found_patterns:
            print(f"\n{symbol}:")
            for i, signal in enumerate(item['signals'], 1):
                print(f"  Сигнал #{i}:")
                print(f"    SHORT: вход {signal['short_entry']:.4f}, цель {signal['short_target']:.4f}")
                print(f"    LONG: вход {signal['long_entry']:.4f}, цель {signal['long_target']:.4f}")
    else:
        print("\n❌ Паттерны не найдены")
