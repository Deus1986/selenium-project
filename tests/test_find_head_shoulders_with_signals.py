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


def get_24h_volume_usdt(min_volume=5000000):
    """Получает монеты с объемом больше указанного"""
    url = "https://contract.mexc.com/api/v1/contract/ticker"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        coins_array = []

        if data and 'data' in data:
            for item in data["data"]:
                if item.get("amount24", 0) > min_volume:
                    coins_array.append(item["symbol"])
        return coins_array
    except Exception as e:
        print(f"Ошибка получения объема: {e}")
        return []


def find_local_maxima(prices, window=5):
    """Находит индексы локальных максимумов"""
    maxima = argrelextrema(prices, np.greater, order=window)[0]
    return np.unique(maxima)


def find_local_minima(prices, window=5):
    """Находит индексы локальных минимумов"""
    minima = argrelextrema(prices, np.less, order=window)[0]
    return minima


def find_neckline_point(prices, start_idx, end_idx, mode='min'):
    """Находит точку для линии шеи"""
    if start_idx >= end_idx or end_idx - start_idx < 2:
        return None
    if mode == 'min':
        return np.argmin(prices[start_idx:end_idx]) + start_idx
    else:
        return np.argmax(prices[start_idx:end_idx]) + start_idx


def calculate_trading_signals(pattern, current_price):
    """Рассчитывает сигналы входа и выхода"""
    neckline_avg = (pattern['neckline_left_price'] + pattern['neckline_right_price']) / 2
    pattern_height = pattern['pattern_height']

    # Медвежий сигнал (пробитие шеи)
    short_entry = neckline_avg - (pattern_height * 0.05)  # Пробитие на 5% от высоты
    short_target = neckline_avg - pattern_height  # Цель = высота паттерна
    short_stop = neckline_avg + (pattern_height * 0.1)  # Стоп над шеей

    # Бычий сигнал (отскок от шеи)
    long_entry = neckline_avg - (pattern_height * 0.02)  # Отскок от шеи
    long_target = neckline_avg + (pattern_height * 0.5)  # Цель = 50% от высоты
    long_stop = neckline_avg - (pattern_height * 0.15)  # Стоп под шеей

    # Определяем активный сигнал
    signals = {
        'short': {
            'entry': short_entry,
            'target': short_target,
            'stop': short_stop,
            'rr_ratio': abs(short_target - short_entry) / abs(short_stop - short_entry),
            'active': current_price <= short_entry
        },
        'long': {
            'entry': long_entry,
            'target': long_target,
            'stop': long_stop,
            'rr_ratio': abs(long_target - long_entry) / abs(long_stop - long_entry),
            'active': current_price >= long_entry and current_price <= neckline_avg
        }
    }

    return signals


def find_head_shoulders_pattern(data, window=5, tolerance=0.015, min_distance=3):
    """Находит паттерн 'Голова и плечи' с сигналами"""
    if not data or not data.get('success'):
        return []

    highs = np.array(data['data']['high'])
    lows = np.array(data['data']['low'])
    closes = np.array(data['data']['close'])
    current_price = closes[-1] if len(closes) > 0 else 0

    local_maxima = find_local_maxima(highs, window=window)
    patterns = []

    for i in range(len(local_maxima) - 4):
        try:
            left_shoulder_idx = local_maxima[i]
            head_idx = local_maxima[i + 1]
            right_shoulder_idx = local_maxima[i + 2]

            if (head_idx - left_shoulder_idx < min_distance or
                    right_shoulder_idx - head_idx < min_distance):
                continue

            left_price = highs[left_shoulder_idx]
            head_price = highs[head_idx]
            right_price = highs[right_shoulder_idx]

            head_ratio = head_price / max(left_price, right_price)
            shoulders_diff = abs(left_price - right_price) / min(left_price, right_price)

            if (head_ratio > 1.01 and shoulders_diff <= tolerance and
                    head_price > left_price and head_price > right_price):

                neck_left_idx = find_neckline_point(lows, left_shoulder_idx, head_idx, 'min')
                neck_right_idx = find_neckline_point(lows, head_idx, right_shoulder_idx, 'min')

                if neck_left_idx is not None and neck_right_idx is not None:
                    pattern = {
                        'left_shoulder_idx': left_shoulder_idx,
                        'head_idx': head_idx,
                        'right_shoulder_idx': right_shoulder_idx,
                        'left_shoulder_price': left_price,
                        'head_price': head_price,
                        'right_shoulder_price': right_price,
                        'neckline_left_idx': neck_left_idx,
                        'neckline_right_idx': neck_right_idx,
                        'neckline_left_price': lows[neck_left_idx],
                        'neckline_right_price': lows[neck_right_idx],
                        'pattern_height': head_price - ((lows[neck_left_idx] + lows[neck_right_idx]) / 2),
                        'shoulders_diff_percent': shoulders_diff * 100
                    }

                    # Добавляем торговые сигналы
                    pattern['signals'] = calculate_trading_signals(pattern, current_price)
                    patterns.append(pattern)

        except IndexError:
            continue

    return patterns


def plot_head_shoulders_with_signals(data, pattern, symbol):
    """Отрисовывает график с сигналами входа и выхода"""
    highs = np.array(data['data']['high'])
    lows = np.array(data['data']['low'])
    closes = np.array(data['data']['close'])
    opens = np.array(data['data']['open'])

    fig, ax = plt.subplots(figsize=(16, 10))

    # Рисуем свечи
    for i in range(len(highs)):
        color = 'green' if closes[i] > opens[i] else 'red'
        ax.plot([i, i], [lows[i], highs[i]], color=color, linewidth=1.5, alpha=0.7)
        ax.plot(i, closes[i], 'o', color=color, markersize=3, alpha=0.8)

    # Точки паттерна
    ls_idx = pattern['left_shoulder_idx']
    h_idx = pattern['head_idx']
    rs_idx = pattern['right_shoulder_idx']
    nl_idx = pattern['neckline_left_idx']
    nr_idx = pattern['neckline_right_idx']

    ax.plot(ls_idx, highs[ls_idx], 'ro', markersize=10, label='Левое плечо')
    ax.plot(h_idx, highs[h_idx], 'go', markersize=12, label='Голова')
    ax.plot(rs_idx, highs[rs_idx], 'ro', markersize=10, label='Правое плечо')
    ax.plot(nl_idx, lows[nl_idx], 'bo', markersize=8, label='Шея')
    ax.plot(nr_idx, lows[nr_idx], 'bo', markersize=8)

    # Линия шеи
    neckline_avg = (pattern['neckline_left_price'] + pattern['neckline_right_price']) / 2
    ax.axhline(y=neckline_avg, color='blue', linestyle='--', alpha=0.7, label='Линия шеи')

    # Сигналы входа и выхода
    signals = pattern['signals']

    # Медвежьи сигналы
    ax.axhline(y=signals['short']['entry'], color='red', linestyle='-',
               alpha=0.8, label='Вход SHORT')
    ax.axhline(y=signals['short']['target'], color='red', linestyle=':',
               alpha=0.6, label='Цель SHORT')
    ax.axhline(y=signals['short']['stop'], color='orange', linestyle=':',
               alpha=0.6, label='Стоп SHORT')

    # Бычьи сигналы
    ax.axhline(y=signals['long']['entry'], color='green', linestyle='-',
               alpha=0.8, label='Вход LONG')
    ax.axhline(y=signals['long']['target'], color='green', linestyle=':',
               alpha=0.6, label='Цель LONG')
    ax.axhline(y=signals['long']['stop'], color='orange', linestyle=':',
               alpha=0.6)

    # Текущая цена
    current_price = closes[-1]
    ax.axhline(y=current_price, color='purple', linestyle='-',
               alpha=0.9, label='Текущая цена')

    # Информация о сигналах
    info_text = f"""Паттерн Голова и Плечи - {symbol}

🎯 ТОРГОВЫЕ СИГНАЛЫ:

SHORT (Медвежий):
• Вход: {signals['short']['entry']:.4f}
• Цель: {signals['short']['target']:.4f}
• Стоп: {signals['short']['stop']:.4f}
• R/R: {signals['short']['rr_ratio']:.2f}
• Статус: {'АКТИВЕН' if signals['short']['active'] else 'не активен'}

LONG (Бычий):
• Вход: {signals['long']['entry']:.4f}
• Цель: {signals['long']['target']:.4f}
• Стоп: {signals['long']['stop']:.4f}
• R/R: {signals['long']['rr_ratio']:.2f}
• Статус: {'АКТИВЕН' if signals['long']['active'] else 'не активен'}

Параметры паттерна:
• Высота: {pattern['pattern_height']:.4f}
• Разница плеч: {pattern['shoulders_diff_percent']:.2f}%
• Текущая цена: {current_price:.4f}"""

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

    ax.set_title(f'Голова и плечи - {symbol} - Торговые сигналы',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Временные периоды')
    ax.set_ylabel('Цена')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return signals


def analyze_and_trade(symbol, time_start, time_end):
    """Анализирует паттерн и возвращает торговые сигналы"""
    print(f"\n🔍 Анализируем {symbol}...")

    data = find_candles(symbol, time_start, time_end)
    if not data or not data.get('success'):
        print(f"Нет данных для {symbol}")
        return None

    patterns = find_head_shoulders_pattern(data)
    if not patterns:
        print(f"Паттерны не найдены для {symbol}")
        return None

    print(f"🎯 Найдено {len(patterns)} паттернов для {symbol}")

    trade_signals = []
    for i, pattern in enumerate(patterns, 1):
        print(f"\n📊 Паттерн #{i}:")
        print(f"   Высота: {pattern['pattern_height']:.4f}")
        print(f"   Плечи: {pattern['left_shoulder_price']:.4f} - {pattern['right_shoulder_price']:.4f}")

        signals = plot_head_shoulders_with_signals(data, pattern, symbol)
        trade_signals.append({
            'symbol': symbol,
            'pattern': pattern,
            'signals': signals
        })

    return trade_signals


def test_find_head_shoulders_with_signals():
    """Основная функция"""
    # Настройки времени
    # time_end = int(time.time() * 1000)
    # time_start = time_end - (6 * 60 * 60 * 1000)  # 6 часов данных

    time_start = round(int(time.time()) - 20)
    time_end = time_start - (150 * 1 * 60)

    # Получаем монеты с объемом
    coins = get_24h_volume_usdt(min_volume=25000000)
    print(f"📈 Анализируем {len(coins)} монет с высоким объемом...")

    all_signals = []

    # Анализируем первые 3 монеты для скорости
    for symbol in coins\
            :
        signals = analyze_and_trade(symbol, time_end, time_start)
        if signals:
            all_signals.extend(signals)
        time.sleep(1)

    # Вывод итоговых сигналов
    if all_signals:
        print(f"\n🎉 ИТОГОВЫЕ ТОРГОВЫЕ СИГНАЛЫ:")
        print("=" * 60)

        for signal in all_signals:
            s = signal['signals']
            print(f"\n{symbol}:")

            if s['short']['active']:
                print(f"   SHORT: Вход {s['short']['entry']:.4f} -> Цель {s['short']['target']:.4f}")
                print(f"          Стоп {s['short']['stop']:.4f} | R/R: {s['short']['rr_ratio']:.2f}")

            if s['long']['active']:
                print(f"   LONG:  Вход {s['long']['entry']:.4f} -> Цель {s['long']['target']:.4f}")
                print(f"          Стоп {s['long']['stop']:.4f} | R/R: {s['long']['rr_ratio']:.2f}")
    else:
        print("\n❌ Активных торговых сигналов не найдено")
