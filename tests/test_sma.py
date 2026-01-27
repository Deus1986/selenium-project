import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests


def find_candles(symbol, start_time, end_time):
    params = {
        "interval": "Min15",
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


class MATestSignalFinder:
    def __init__(self):
        self.sma_periods = [9, 21, 50]  # Быстрая, средняя, медленная SMA

    def calculate_moving_averages(self, closes):
        """Рассчитывает скользящие средние"""
        ma_signals = {}

        for period in self.sma_periods:
            if len(closes) >= period:
                ma_signals[f'sma_{period}'] = pd.Series(closes).rolling(window=period).mean().values
            else:
                ma_signals[f'sma_{period}'] = np.array([np.nan] * len(closes))

        return ma_signals

    def find_ma_test_signals(self, data, symbol):
        """Ищет сигналы ретеста скользящих средних"""
        closes = np.array(data['close'])
        highs = np.array(data['high'])
        lows = np.array(data['low'])

        # Рассчитываем SMA
        ma_data = self.calculate_moving_averages(closes)

        current_price = closes[-1]
        signals = []

        # Анализируем каждую SMA
        for period in self.sma_periods:
            sma_key = f'sma_{period}'
            sma_values = ma_data[sma_key]

            if len(sma_values) < 2 or np.isnan(sma_values[-1]):
                continue

            current_sma = sma_values[-1]
            prev_sma = sma_values[-2] if len(sma_values) > 1 else current_sma

            # Проверяем ретест для лонга (цена выше SMA и отскакивает)
            long_signal = self.check_long_signal(current_price, current_sma, prev_sma,
                                                 closes, highs, lows, period)
            if long_signal:
                signals.append(long_signal)

            # Проверяем ретест для шорта (цена ниже SMA и отскакивает)
            short_signal = self.check_short_signal(current_price, current_sma, prev_sma,
                                                   closes, highs, lows, period)
            if short_signal:
                signals.append(short_signal)

        return signals

    def check_long_signal(self, current_price, current_sma, prev_sma, closes, highs, lows, period):
        """Проверяет сигнал лонг на ретесте SMA"""
        # Условия для лонга:
        # 1. Цена выше SMA
        # 2. SMA восходящая
        # 3. Недавно был тест SMA снизу
        # 4. Текущая свеча зеленая

        if current_price <= current_sma:
            return None

        # Проверяем восходящий тренд SMA
        sma_trend = current_sma > prev_sma

        # Ищем недавний тест SMA снизу
        recent_test = self.find_recent_ma_test(closes, current_sma, lookback=10, test_type='support')

        # Текущая свеча зеленая
        last_close = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else last_close
        is_green_candle = last_close > prev_close

        if sma_trend and recent_test and is_green_candle:
            score = 60
            score += 10 if recent_test['strength'] == 'strong' else 0
            score += 10 if (current_price - current_sma) / current_sma > 0.001 else 0

            # Рассчитываем уровни для торговли
            atr = self.calculate_atr(highs, lows, closes)

            return {
                'type': 'LONG',
                'ma_period': period,
                'ma_value': current_sma,
                'price': current_price,
                'score': score,
                'entry_price': current_price,
                'stop_loss': current_sma - (atr * 1.0),
                'take_profit': current_price + (current_price - current_sma) * 2,
                'distance_to_ma': ((current_price - current_sma) / current_sma) * 100,
                'conditions': [
                    f"Цена выше SMA{period}",
                    f"SMA{period} восходящая",
                    f"Недавний тест поддержки",
                    f"Зеленая свеча"
                ]
            }

        return None

    def check_short_signal(self, current_price, current_sma, prev_sma, closes, highs, lows, period):
        """Проверяет сигнал шорт на ретесте SMA"""
        # Условия для шорта:
        # 1. Цена ниже SMA
        # 2. SMA нисходящая
        # 3. Недавно был тест SMA сверху
        # 4. Текущая свеча красная

        if current_price >= current_sma:
            return None

        # Проверяем нисходящий тренд SMA
        sma_trend = current_sma < prev_sma

        # Ищем недавний тест SMA сверху
        recent_test = self.find_recent_ma_test(closes, current_sma, lookback=10, test_type='resistance')

        # Текущая свеча красная
        last_close = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else last_close
        is_red_candle = last_close < prev_close

        if sma_trend and recent_test and is_red_candle:
            score = 60
            score += 10 if recent_test['strength'] == 'strong' else 0
            score += 10 if (current_sma - current_price) / current_sma > 0.001 else 0

            # Рассчитываем уровни для торговли
            atr = self.calculate_atr(highs, lows, closes)

            return {
                'type': 'SHORT',
                'ma_period': period,
                'ma_value': current_sma,
                'price': current_price,
                'score': score,
                'entry_price': current_price,
                'stop_loss': current_sma + (atr * 1.0),
                'take_profit': current_price - (current_sma - current_price) * 2,
                'distance_to_ma': ((current_sma - current_price) / current_sma) * 100,
                'conditions': [
                    f"Цена ниже SMA{period}",
                    f"SMA{period} нисходящая",
                    f"Недавний тест сопротивления",
                    f"Красная свеча"
                ]
            }

        return None

    def find_recent_ma_test(self, closes, ma_value, lookback=10, test_type='support'):
        """Ищет недавний тест скользящей средней"""
        if len(closes) < lookback:
            return None

        recent_closes = closes[-lookback:-1]  # Исключаем текущую свечу

        if test_type == 'support':
            # Ищем тест поддержки (цена касалась MA снизу и отскочила)
            touches = [i for i, price in enumerate(recent_closes)
                       if abs(price - ma_value) / ma_value < 0.002]  # Касание в пределах 0.2%

            if touches:
                last_touch = max(touches)
                # Проверяем, что после касания цена пошла в нужном направлении
                if test_type == 'support':
                    if closes[last_touch] < closes[last_touch + 1]:
                        return {'index': last_touch, 'strength': 'strong'}

        else:  # resistance
            # Ищем тест сопротивления (цена касалась MA сверху и отскочила)
            touches = [i for i, price in enumerate(recent_closes)
                       if abs(price - ma_value) / ma_value < 0.002]

            if touches:
                last_touch = max(touches)
                if test_type == 'resistance':
                    if closes[last_touch] > closes[last_touch + 1]:
                        return {'index': last_touch, 'strength': 'strong'}

        return None

    def calculate_atr(self, highs, lows, closes, period=14):
        """Рассчитывает Average True Range"""
        if len(highs) < period + 1:
            return np.mean(highs) * 0.01  # Fallback

        tr = []
        for i in range(1, len(highs)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i - 1])
            low_close = abs(lows[i] - closes[i - 1])
            true_range = max(high_low, high_close, low_close)
            tr.append(true_range)

        return np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)

    def plot_ma_test_signals(self, data, signals, symbol):
        """Строит график с сигналами ретеста SMA"""
        if not signals:
            print("Нет сигналов для построения графика")
            return

        closes = data['close']
        highs = data['high']
        lows = data['low']
        times = data['time']

        fig, ax = plt.subplots(figsize=(15, 10))

        # Рисуем свечи
        for i in range(len(closes)):
            color = 'green' if closes[i] > (highs[i] + lows[i]) / 2 else 'red'
            ax.plot([i, i], [lows[i], highs[i]], color=color, linewidth=1, alpha=0.7)
            ax.plot(i, closes[i], 'o', color=color, markersize=3, alpha=0.8)

        # Рисуем скользящие средние
        ma_data = self.calculate_moving_averages(closes)
        colors = ['blue', 'orange', 'purple']

        for i, period in enumerate(self.sma_periods):
            sma_key = f'sma_{period}'
            if sma_key in ma_data:
                sma_values = ma_data[sma_key]
                # Отображаем только валидные значения
                valid_indices = ~np.isnan(sma_values)
                if np.any(valid_indices):
                    ax.plot(np.where(valid_indices)[0], sma_values[valid_indices],
                            color=colors[i % len(colors)], linewidth=2, label=f'SMA {period}')

        # Размечаем сигналы
        for signal in signals:
            idx = len(closes) - 1  # Текущий бар
            color = 'green' if signal['type'] == 'LONG' else 'red'
            marker = '^' if signal['type'] == 'LONG' else 'v'

            ax.plot(idx, signal['price'], marker, color=color, markersize=12,
                    markeredgewidth=2, markeredgecolor='black')

            # Подписываем сигнал
            ax.annotate(f"{signal['type']} SMA{signal['ma_period']}\nScore: {signal['score']}",
                        xy=(idx, signal['price']), xytext=(10, 30 if signal['type'] == 'LONG' else -30),
                        textcoords='offset points', ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        ax.set_title(f'Сигналы ретеста SMA - {symbol}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Временные периоды')
        ax.set_ylabel('Цена')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

    def analyze_and_plot(self, data, symbol):
        """Анализирует и строит график"""
        signals = self.find_ma_test_signals(data, symbol)

        if signals:
            print(f"🎯 Найдено сигналов для {symbol}: {len(signals)}")
            for signal in signals:
                self.print_signal_details(signal)

            self.plot_ma_test_signals(data, signals, symbol)
        else:
            print(f"❌ Сигналы ретеста SMA не найдены для {symbol}")

    def print_signal_details(self, signal):
        """Выводит детали сигнала"""
        print(f"\n{'=' * 60}")
        print(f"🎯 СИГНАЛ {signal['type']} НА РЕТЕСТЕ SMA{signal['ma_period']}")
        print(f"{'=' * 60}")
        print(f"📊 Цена: {signal['price']:.2f}")
        print(f"📈 SMA{signal['ma_period']}: {signal['ma_value']:.2f}")
        print(f"📏 Расстояние до MA: {signal['distance_to_ma']:.3f}%")
        print(f"💪 Сила сигнала: {signal['score']}/100")

        print(f"\n🎯 Торговые уровни:")
        print(f"   Вход: {signal['entry_price']:.2f}")

        if signal['type'] == 'LONG':
            print(
                f"   Стоп: {signal['stop_loss']:.2f} (-{((1 - signal['stop_loss'] / signal['entry_price']) * 100):.1f}%)")
            print(
                f"   Тейк: {signal['take_profit']:.2f} (+{((signal['take_profit'] / signal['entry_price']) - 1) * 100:.1f}%)")
        else:
            print(
                f"   Стоп: {signal['stop_loss']:.2f} (+{((signal['stop_loss'] / signal['entry_price']) - 1) * 100:.1f}%)")
            print(
                f"   Тейк: {signal['take_profit']:.2f} (-{((1 - signal['take_profit'] / signal['entry_price']) * 100):.1f}%)")

        print(f"\n📋 Условия:")
        for condition in signal['conditions']:
            print(f"   ✓ {condition}")


def test_find_double_top():
    """
    Основная функция тестирования
    """
    # Временной диапазон (последние 4 часа)
    # time_end = int(time.time() * 1000)
    # time_start = time_end - (8 * 60 * 60 * 1000)  # 4 часа назад
    time_start = round(int(time.time()) - 20)
    time_end = time_start - (100 * 15 * 60)

    coins = get_24h_volume_usdt(min_volume=20000000)  # Минимум 100M объема

    print(f"Анализируем {len(coins)} монет с высоким объемом...")

    for symbol in coins:  # Анализируем первые 5 монет для скорости
        print(f"\n🔍 Анализируем {symbol}...")

        data = find_candles(symbol, time_end, time_start)

        finder = MATestSignalFinder()
        finder.analyze_and_plot(data["data"], symbol)
