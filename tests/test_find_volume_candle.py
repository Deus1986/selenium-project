import time
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime


def find_candles(symbol, start_time, end_time):
    """Получение свечных данных"""
    params = {
        "interval": "Min30",
        "start": start_time,
        "end": f"{end_time}"
    }
    response = requests.get(f"https://contract.mexc.com/api/v1/contract/kline/{symbol}", params=params)
    assert response.status_code == 200
    return response.json()


def get_24h_volume_usdt(min_volume=50000000):
    """Получает объем торгов в USDT за 24 часа"""
    url = f"https://contract.mexc.com/api/v1/contract/ticker"
    response = requests.get(url)
    data = response.json()
    coins_array = []

    for item in data["data"]:
        if item["amount24"] > min_volume:
            coins_array.append(item["symbol"])

    return coins_array


def analyze_volume_spike_candles(data, symbol, volume_threshold=3.0, body_threshold=0.3):
    """
    Анализирует свечи с маленьким телом и большим объемом

    Args:
        data: свечные данные
        symbol: символ торговой пары
        volume_threshold: минимальное отношение объема к среднему (по умолчанию 3x)
        body_threshold: максимальный размер тела свечи (0.3 = 30% от диапазона)
    """
    if not data or not data.get('success'):
        return []

    # Извлекаем данные
    opens = np.array(data['data']['open'], dtype=float)
    highs = np.array(data['data']['high'], dtype=float)
    lows = np.array(data['data']['low'], dtype=float)
    closes = np.array(data['data']['close'], dtype=float)
    volumes = np.array(data['data']['vol'], dtype=float)
    times = np.array(data['data']['time'])

    spikes = []

    for i in range(1, len(opens)):
        # Пропускаем первые 10 свечей для расчета среднего объема
        if i < 10:
            continue

        current_volume = volumes[i]
        avg_volume = np.mean(volumes[max(0, i - 20):i])  # Средний объем за последние 20 свечей

        if avg_volume == 0:
            continue

        volume_ratio = current_volume / avg_volume

        # Расчет размера тела свечи
        body_size = abs(closes[i] - opens[i])
        high_low_range = highs[i] - lows[i]

        if high_low_range == 0:
            continue

        body_ratio = body_size / high_low_range

        # Проверка условий: маленькое тело + большой объем
        if (body_ratio <= body_threshold and
                volume_ratio >= volume_threshold):

            # Определяем тип свечи
            if body_ratio < 0.05:
                candle_type = "DOJI"
            elif closes[i] > opens[i]:
                candle_type = "SMALL_GREEN"
            else:
                candle_type = "SMALL_RED"

            spike_data = {
                'index': i,
                'timestamp': times[i],
                'datetime': datetime.fromtimestamp(times[i] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                'open': opens[i],
                'high': highs[i],
                'low': lows[i],
                'close': closes[i],
                'volume': current_volume,
                'avg_volume': avg_volume,
                'volume_ratio': round(volume_ratio, 2),
                'body_ratio': round(body_ratio, 4),
                'body_percent': round(body_ratio * 100, 2),
                'candle_type': candle_type,
                'price_change_percent': round(((closes[i] - opens[i]) / opens[i]) * 100, 4)
            }
            spikes.append(spike_data)

    return spikes


def plot_volume_spike_candles(data, spikes, symbol, show_last=50):
    """
    Отрисовывает график со свечами с аномальным объемом
    """
    if not data or not spikes:
        print(f"Нет данных для отрисовки {symbol}")
        return

    # Извлекаем данные
    opens = np.array(data['data']['open'], dtype=float)
    highs = np.array(data['data']['high'], dtype=float)
    lows = np.array(data['data']['low'], dtype=float)
    closes = np.array(data['data']['close'], dtype=float)
    volumes = np.array(data['data']['vol'], dtype=float)

    # Ограничиваем количество отображаемых свечей
    start_idx = max(0, len(opens) - show_last)
    opens = opens[start_idx:]
    highs = highs[start_idx:]
    lows = lows[start_idx:]
    closes = closes[start_idx:]
    volumes = volumes[start_idx:]

    # Создаем график
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # Верхний график - свечи
    for i in range(len(opens)):
        color = 'green' if closes[i] > opens[i] else 'red'
        ax1.plot([i, i], [lows[i], highs[i]], color=color, linewidth=2, alpha=0.8)
        ax1.plot(i, closes[i], 'o', color=color, markersize=4, alpha=0.8)

    # Отмечаем свечи с аномальным объемом
    for spike in spikes:
        idx = spike['index'] - start_idx
        if 0 <= idx < len(opens):
            ax1.plot(idx, closes[idx], 'o', color='gold', markersize=10,
                     markeredgecolor='black', markeredgewidth=2,
                     label='Объемная свеча' if idx == spikes[0]['index'] - start_idx else "")

    # Нижний график - объемы
    volume_colors = ['green' if closes[i] > opens[i] else 'red' for i in range(len(opens))]
    ax2.bar(range(len(volumes)), volumes, color=volume_colors, alpha=0.7)

    # Отмечаем аномальные объемы
    for spike in spikes:
        idx = spike['index'] - start_idx
        if 0 <= idx < len(volumes):
            ax2.bar(idx, volumes[idx], color='gold', alpha=1.0,
                    edgecolor='black', linewidth=2)

    # Настройки графиков
    ax1.set_title(f'Свечи с малым телом и большим объемом - {symbol}', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Цена')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_xlabel('Временные периоды')
    ax2.set_ylabel('Объем')
    ax2.grid(True, alpha=0.3)

    # Добавляем информацию о найденных свечах
    if spikes:
        info_text = f"Найдено свеч: {len(spikes)}\n"
        for spike in spikes[-3:]:  # Показываем последние 3
            info_text += f"\n{spike['datetime']}:\n"
            info_text += f"• Объем: {spike['volume_ratio']}x\n"
            info_text += f"• Тело: {spike['body_percent']}%\n"
            info_text += f"• Тип: {spike['candle_type']}\n"

        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
                 verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

    plt.tight_layout()
    plt.show()


def scan_volume_spike_candles(volume_threshold=2.0, body_threshold=0.3, min_volume_usdt=20000000):
    """
    Основная функция сканирования свечей с аномальным объемом
    """
    # Временной диапазон (последние 4 часа)
    time_start = round(int(time.time()) - 20)
    time_end = time_start - (2 * 30 * 60)  # 4 часа назад

    coins = get_24h_volume_usdt(min_volume=min_volume_usdt)

    print(f"🔍 Сканируем {len(coins)} монет на предмет свечей с малым телом и большим объемом...")
    print(f"📊 Параметры: объем > {volume_threshold}x, тело < {body_threshold * 100}%")

    found_spikes = []

    for symbol in coins:  # Ограничиваем для тестирования
        # print(f"Анализируем {symbol}...")

        try:
            data = find_candles(symbol, time_end, time_start)

            if data and data.get('success'):
                spikes = analyze_volume_spike_candles(
                    data, symbol,
                    volume_threshold=volume_threshold,
                    body_threshold=body_threshold
                )

                if spikes:
                    found_spikes.append({'symbol': symbol, 'spikes': spikes})
                    print(f"  ✅ Найдено: {len(spikes)} свеч")

                    # Отрисовываем график для символа с найденными свечами
                    plot_volume_spike_candles(data, spikes, symbol)

            time.sleep(0.5)  # Пауза между запросами

        except Exception as e:
            print(f"  ❌ Ошибка анализа {symbol}: {e}")
            continue

    # Вывод результатов
    print(f"\n{'=' * 80}")
    print("📊 РЕЗУЛЬТАТЫ СКАНИРОВАНИЯ:")
    print(f"{'=' * 80}")

    if found_spikes:
        # Сортируем по количеству найденных свеч
        found_spikes.sort(key=lambda x: len(x['spikes']), reverse=True)

        for item in found_spikes:
            symbol = item['symbol']
            spikes = item['spikes']

            print(f"\n🎯 {symbol}: {len(spikes)} свеч с аномальным объемом")

            # Группируем по типам свечей
            doji_count = len([s for s in spikes if s['candle_type'] == 'DOJI'])
            green_count = len([s for s in spikes if s['candle_type'] == 'SMALL_GREEN'])
            red_count = len([s for s in spikes if s['candle_type'] == 'SMALL_RED'])

            print(f"   Типы: DOJI: {doji_count}, Зеленые: {green_count}, Красные: {red_count}")

            # Показываем самые сильные сигналы
            strongest = sorted(spikes, key=lambda x: x['volume_ratio'], reverse=True)[:2]
            for spike in strongest:
                print(f"   📈 {spike['datetime']} - Объем: {spike['volume_ratio']}x, "
                      f"Тело: {spike['body_percent']}%, Изменение: {spike['price_change_percent']}%")
    else:
        print("\n❌ Свечи с аномальным объемом не найдены")

    return found_spikes


def get_detailed_analysis(symbol, hours_back=8):
    """
    Детальный анализ конкретного символа
    """
    global spikes
    time_start = round(int(time.time()) - 20)
    time_end = time_start - (5 * 30 * 60)

    print(f"\n🔍 Детальный анализ {symbol} за последние {hours_back} часов:")

    try:
        data = find_candles(symbol, time_start, time_end)

        if data and data.get('success'):
            # Анализ с разными параметрами
            thresholds = [
                (5.0, 0.2),  # Очень строгие
                (3.0, 0.3),  # Средние
                (1.0, 0.9)  # Более мягкие
            ]

            for vol_thresh, body_thresh in thresholds:
                spikes = analyze_volume_spike_candles(
                    data, symbol,
                    volume_threshold=vol_thresh,
                    body_threshold=body_thresh
                )

                print(f"\nПараметры: объем > {vol_thresh}x, тело < {body_thresh * 100}%")
                print(f"Найдено свеч: {len(spikes)}")

                if spikes:
                    # Группируем по часам для анализа активности
                    hours = {}
                    for spike in spikes:
                        hour = spike['datetime'][11:13]
                        hours[hour] = hours.get(hour, 0) + 1

                    print(f"Активные часы: {dict(sorted(hours.items()))}")

            return spikes

    except Exception as e:
        print(f"Ошибка анализа {symbol}: {e}")

    return []


# Пример использования
def test_find_volume_candle():
    # Быстрое сканирование
    # print("🚀 Запуск сканирования свечей с малым телом и большим объемом...")

    # Основное сканирование
    results = scan_volume_spike_candles(
        volume_threshold=0.1,  # Объем в 3 раза больше среднего
        body_threshold=0.99,  # Тело меньше 30% от диапазона
        min_volume_usdt=20000000  # Минимум 20M объема
    )

    # Детальный анализ топовых монет
    if results:
        top_symbol = results[0]['symbol']
        print(f"\n📈 Детальный анализ для {top_symbol}:")
        get_detailed_analysis(top_symbol)
