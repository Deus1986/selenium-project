import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import requests


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


def find_local_minima(prices, window=5):
    """
    Находит индексы локальных минимумов
    """
    minima = argrelextrema(prices, np.less, order=window)[0]

    if len(prices) > window:
        if prices[0] < np.min(prices[1:window + 1]):
            minima = np.append(minima, 0)
        if prices[-1] < np.min(prices[-window - 1:-1]):
            minima = np.append(minima, len(prices) - 1)

    return np.unique(minima)


def find_resistance_line(prices, start_idx, end_idx):
    """
    Находит максимальное значение (линию сопротивления) между двумя минимумами
    """
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    if end_idx - start_idx < 2:
        return None

    resistance_idx = np.argmax(prices[start_idx:end_idx + 1]) + start_idx
    return resistance_idx


def find_double_bottom_pattern(data, window=5, tolerance=0.015, min_distance=3):
    """
    Находит паттерн двойное дно
    """
    if not data or not data.get('success'):
        return []

    lows = np.array(data['data']['low'])
    highs = np.array(data['data']['high'])
    times = np.array(data['data']['time'])

    local_minima = find_local_minima(lows, window=window)
    patterns = []

    for i in range(len(local_minima)):
        for j in range(i + 1, len(local_minima)):
            idx1 = local_minima[i]
            idx2 = local_minima[j]

            if abs(idx2 - idx1) < min_distance:
                continue

            price1 = lows[idx1]
            price2 = lows[idx2]
            price_diff = abs(price1 - price2) / min(price1, price2)

            if price_diff <= tolerance:
                resistance_idx = find_resistance_line(highs, idx1, idx2)  # Используем highs для сопротивления

                if resistance_idx is not None:
                    pattern = {
                        'first_bottom_index': idx1,
                        'second_bottom_index': idx2,
                        'first_bottom_price': price1,
                        'second_bottom_price': price2,
                        'resistance_index': resistance_idx,
                        'resistance_price': highs[resistance_idx],
                        'pattern_height': highs[resistance_idx] - max(price1, price2),
                        'completion_index': resistance_idx + (resistance_idx - idx2)
                    }
                    patterns.append(pattern)

    return patterns


def plot_double_bottom_with_signals(data, pattern, symbol):
    """
    Отрисовывает график с паттерном двойное дно и точками входа
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
    idx1 = pattern['first_bottom_index']
    idx2 = pattern['second_bottom_index']
    resistance_idx = pattern['resistance_index']

    # Рисуем паттерн
    ax.plot(idx1, lows[idx1], 'go', markersize=10, label='Первое дно')
    ax.plot(idx2, lows[idx2], 'go', markersize=10, label='Второе дно')
    ax.plot(resistance_idx, highs[resistance_idx], 'ro', markersize=10, label='Сопротивление')

    # Линия сопротивления
    ax.axhline(y=pattern['resistance_price'], color='red', linestyle='--',
               alpha=0.7, label='Линия сопротивления')

    # Точки входа в сделку (для бычьего паттерна)
    entry_long = pattern['resistance_price'] + (pattern['pattern_height'] * 0.1)  # Пробитие сопротивления
    target_long = pattern['resistance_price'] + pattern['pattern_height']  # Цель = высота паттерна
    stop_long = pattern['resistance_price'] - (pattern['pattern_height'] * 0.1)  # Стоп под сопротивлением

    # Точки входа для отскока (консервативный вход)
    entry_bounce = max(pattern['first_bottom_price'], pattern['second_bottom_price']) + (
                pattern['pattern_height'] * 0.2)
    target_bounce = pattern['resistance_price']
    stop_bounce = min(pattern['first_bottom_price'], pattern['second_bottom_price']) - (pattern['pattern_height'] * 0.1)

    # Рисуем точки входа и цели
    entry_time = resistance_idx + 1
    if entry_time < len(highs):
        # Точка входа в лонг (пробитие сопротивления)
        ax.plot(entry_time, entry_long, '^', color='green', markersize=12,
                label='Вход в лонг (пробитие)', markeredgewidth=2, markeredgecolor='black')

        # Цель для лонга
        ax.axhline(y=target_long, color='green', linestyle=':', alpha=0.7,
                   label='Цель лонга')

        # Стоп для лонга
        ax.axhline(y=stop_long, color='orange', linestyle=':', alpha=0.7,
                   label='Стоп лонга')

        # Точка входа при отскоке
        bounce_time = max(idx1, idx2) + 1
        if bounce_time < len(highs):
            ax.plot(bounce_time, entry_bounce, '^', color='blue', markersize=10,
                    label='Вход в лонг (отскок)', markeredgewidth=2, markeredgecolor='white')

            # Цель для отскока
            ax.axhline(y=target_bounce, color='blue', linestyle=':', alpha=0.6,
                       label='Цель отскока')

            # Стоп для отскока
            ax.axhline(y=stop_bounce, color='orange', linestyle=':', alpha=0.6)

    # Информация о сделке
    info_text = f"""Паттерн Двойное Дно - {symbol}

🎯 ТОРГОВЫЕ СИГНАЛЫ:

LONG (пробитие сопротивления):
• Вход: {entry_long:.4f} (пробитие сопротивления)
• Цель: {target_long:.4f} (R:R = 1:1)
• Стоп: {stop_long:.4f}

LONG (отскок от дна):
• Вход: {entry_bounce:.4f} (отскок от второго дна)
• Цель: {target_bounce:.4f} (до сопротивления)
• Стоп: {stop_bounce:.4f}

📊 Параметры паттерна:
• Высота: {pattern['pattern_height']:.4f}
• Дна: {pattern['first_bottom_price']:.4f} / {pattern['second_bottom_price']:.4f}
• Сопротивление: {pattern['resistance_price']:.4f}
• Разница доньев: {abs(pattern['first_bottom_price'] - pattern['second_bottom_price']):.4f}"""

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

    # Настройки графика
    ax.set_title(f'Двойное дно - {symbol} - Точки входа', fontsize=16, fontweight='bold')
    ax.set_xlabel('Временные периоды')
    ax.set_ylabel('Цена')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return {
        'long_entry_breakout': entry_long,
        'long_target_breakout': target_long,
        'long_stop_breakout': stop_long,
        'long_entry_bounce': entry_bounce,
        'long_target_bounce': target_bounce,
        'long_stop_bounce': stop_bounce
    }


def analyze_double_bottom_patterns(data, symbol):
    """
    Анализирует паттерны двойное дно и возвращает торговые сигналы
    """
    patterns = find_double_bottom_pattern(data)

    if not patterns:
        print(f"Для {symbol} паттерны двойное дно не найдены")
        return None

    print(f"\n🎯 Найдено паттернов Двойное дно для {symbol}: {len(patterns)}")

    trade_signals = []

    for i, pattern in enumerate(patterns, 1):
        print(f"\nПаттерн #{i}:")
        print(f"  Дна: {pattern['first_bottom_price']:.4f} / {pattern['second_bottom_price']:.4f}")
        print(f"  Сопротивление: {pattern['resistance_price']:.4f}")
        print(f"  Высота: {pattern['pattern_height']:.4f}")

        # Получаем торговые сигналы
        signals = plot_double_bottom_with_signals(data, pattern, symbol)
        trade_signals.append(signals)

    return trade_signals


def test_find_double_bottom_with_image():
    """
    Основная функция тестирования
    """
    # Временной диапазон (последние 4 часа)
    # time_end = int(time.time() * 1000)
    # time_start = time_end - (8 * 60 * 60 * 1000)  # 4 часа назад
    time_start = round(int(time.time()) - 20)
    time_end = time_start - (10 * 1 * 60)

    coins = get_24h_volume_usdt(min_volume=20000000)  # Минимум 100M объема

    print(f"🔍 Анализируем {len(coins)} монет на предмет Двойного дна...")

    found_patterns = []

    for symbol in coins:  # Анализируем первые 5 монет
        print(f"\nАнализируем {symbol}...")

        data = find_candles(symbol, time_end, time_start)

        if data and data.get('success'):
            signals = analyze_double_bottom_patterns(data, symbol)
            if signals:
                found_patterns.append({'symbol': symbol, 'signals': signals})

        time.sleep(1)

    # Вывод результатов
    if found_patterns:
        print(f"\n🎉 Найдены паттерны Двойное дно для {len(found_patterns)} монет:")
        for item in found_patterns:
            symbol = item['symbol']
            print(f"\n{symbol}:")
            for i, signal in enumerate(item['signals'], 1):
                print(f"  Сигнал #{i}:")
                print(
                    f"    Пробитие: вход {signal['long_entry_breakout']:.4f}, цель {signal['long_target_breakout']:.4f}")
                print(f"    Отскок: вход {signal['long_entry_bounce']:.4f}, цель {signal['long_target_bounce']:.4f}")
    else:
        print("\n❌ Паттерны Двойное дно не найдены")