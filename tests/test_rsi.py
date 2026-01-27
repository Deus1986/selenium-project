from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
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




class RSIOverboughtShort:
    def __init__(self, rsi_period=14, overbought_level=70, strong_overbought=80):
        self.rsi_period = rsi_period
        self.overbought_level = overbought_level
        self.strong_overbought = strong_overbought

    def calculate_rsi(self, closes):
        """Рассчитывает RSI"""
        if len(closes) < self.rsi_period + 1:
            return np.array([50] * len(closes))

        # Calculate price changes
        deltas = np.diff(closes)

        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Calculate EMA of gains and losses
        avg_gains = np.zeros_like(closes)
        avg_losses = np.zeros_like(closes)

        # Initial values
        avg_gains[self.rsi_period] = np.mean(gains[:self.rsi_period])
        avg_losses[self.rsi_period] = np.mean(losses[:self.rsi_period])

        # EMA calculation
        for i in range(self.rsi_period + 1, len(closes)):
            avg_gains[i] = (avg_gains[i - 1] * (self.rsi_period - 1) + gains[i - 1]) / self.rsi_period
            avg_losses[i] = (avg_losses[i - 1] * (self.rsi_period - 1) + losses[i - 1]) / self.rsi_period

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        # Fill initial values with 50
        rsi[:self.rsi_period] = 50

        return rsi

    def find_overbought_signals(self, data):
        """Ищет сигналы перекупленности для шорта"""
        closes = np.array(data['close'])
        highs = np.array(data['high'])
        lows = np.array(data['low'])

        # Рассчитываем RSI
        rsi = self.calculate_rsi(closes)
        current_rsi = rsi[-1]
        current_price = closes[-1]

        signals = []

        # Проверяем условия для шорта
        if current_rsi >= self.overbought_level:
            signal_strength = self.analyze_signal_strength(rsi, closes, highs, lows)

            if signal_strength['score'] >= 60:
                # Рассчитываем торговые уровни
                trade_levels = self.calculate_trade_levels(current_price, highs, lows, closes)

                signal = {
                    'type': 'SHORT',
                    'current_price': current_price,
                    'current_rsi': current_rsi,
                    'score': signal_strength['score'],
                    'entry_price': trade_levels['entry'],
                    'stop_loss': trade_levels['stop_loss'],
                    'take_profit': trade_levels['take_profit'],
                    'risk_reward_ratio': trade_levels['rr_ratio'],
                    'conditions': signal_strength['conditions'],
                    'rsi_values': rsi,
                    'timestamp': datetime.now().isoformat()
                }
                signals.append(signal)

        return signals

    def analyze_signal_strength(self, rsi, closes, highs, lows):
        """Анализирует силу сигнала перекупленности"""
        conditions = []
        score = 0

        current_rsi = rsi[-1]

        # 1. Уровень RSI
        if current_rsi >= self.strong_overbought:
            conditions.append(f"Сильная перекупленность (RSI: {current_rsi:.1f})")
            score += 40
        elif current_rsi >= self.overbought_level:
            conditions.append(f"Перекупленность (RSI: {current_rsi:.1f})")
            score += 30

        # 2. Дивергенция RSI
        if self.check_rsi_divergence(rsi, highs, 'bearish'):
            conditions.append("Медвежья дивергенция RSI")
            score += 25

        # 3. Длительность перекупленности
        overbought_bars = self.count_consecutive_overbought(rsi)
        if overbought_bars >= 3:
            conditions.append(f"Перекупленность {overbought_bars} баров подряд")
            score += 15

        # 4. Объем на пиках
        if self.check_volume_at_highs(closes, highs):
            conditions.append("Высокий объем на пиках")
            score += 10

        # 5. Форма свечи
        if self.is_reversal_candle(closes, highs, lows):
            conditions.append("Разворотная свеча")
            score += 20

        return {'score': min(score, 100), 'conditions': conditions}

    def check_rsi_divergence(self, rsi, prices, divergence_type='bearish'):
        """Проверяет дивергенцию RSI"""
        if len(rsi) < 10:
            return False

        # Ищем последние два пика
        recent_rsi = rsi[-10:]
        recent_prices = prices[-10:]

        if divergence_type == 'bearish':
            # Медвежья дивергенция: цены делают更高的高点, RSI - более низкие高点
            price_peak1 = np.max(recent_prices[:5])
            price_peak2 = np.max(recent_prices[5:])
            rsi_peak1 = np.max(recent_rsi[:5])
            rsi_peak2 = np.max(recent_rsi[5:])

            return price_peak2 > price_peak1 and rsi_peak2 < rsi_peak1

        return False

    def count_consecutive_overbought(self, rsi):
        """Считает количество баров подряд в перекупленности"""
        count = 0
        for i in range(len(rsi) - 1, -1, -1):
            if rsi[i] >= self.overbought_level:
                count += 1
            else:
                break
        return count

    def check_volume_at_highs(self, closes, highs):
        """Проверяет объем на пиках цены"""
        # Заглушка - в реальном коде нужно использовать данные объема
        return True

    def is_reversal_candle(self, closes, highs, lows):
        """Проверяет разворотную свечу"""
        if len(closes) < 3:
            return False

        current_close = closes[-1]
        current_high = highs[-1]
        current_low = lows[-1]
        prev_close = closes[-2]

        # Доджи или медвежья свеча с длинной верхней тенью
        candle_body = abs(current_close - prev_close)
        candle_range = current_high - current_low

        if candle_range > 0:
            body_ratio = candle_body / candle_range
            upper_shadow_ratio = (current_high - max(current_close, prev_close)) / candle_range

            # Разворотные признаки
            if (body_ratio < 0.3 and upper_shadow_ratio > 0.4) or \
                    (current_close < prev_close and upper_shadow_ratio > 0.3):
                return True

        return False

    def calculate_trade_levels(self, current_price, highs, lows, closes):
        """Рассчитывает уровни для торговли"""
        # ATR для расчета стоп-лосса
        atr = self.calculate_atr(highs, lows, closes)

        # Уровни на основе волатильности
        entry_price = current_price
        stop_loss = current_price + (atr * 1.5)

        # Цель: RSI вернется к 50-60
        # Простая логика: цель = текущая цена - 2 * ATR
        take_profit = current_price - (atr * 2)

        # Минимальный тейк-профит
        min_profit = current_price * 0.98
        take_profit = min(take_profit, min_profit)

        risk = stop_loss - entry_price
        reward = entry_price - take_profit
        rr_ratio = reward / risk if risk > 0 else 0

        return {
            'entry': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'rr_ratio': rr_ratio,
            'atr': atr
        }

    def calculate_atr(self, highs, lows, closes, period=14):
        """Рассчитывает Average True Range"""
        if len(highs) < period + 1:
            return np.mean(highs) * 0.01

        tr = []
        for i in range(1, len(highs)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i - 1])
            low_close = abs(lows[i] - closes[i - 1])
            true_range = max(high_low, high_close, low_close)
            tr.append(true_range)

        return np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)

    def plot_rsi_signals(self, data, signals, symbol):
        """Строит график с сигналами RSI"""
        closes = data['close']
        highs = data['high']
        lows = data['low']
        times = data['time']

        # Создаем subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[3, 1])

        # Верхний график - цена
        for i in range(len(closes)):
            color = 'green' if closes[i] > (highs[i] + lows[i]) / 2 else 'red'
            ax1.plot([i, i], [lows[i], highs[i]], color=color, linewidth=1.5, alpha=0.8)
            ax1.plot(i, closes[i], 'o', color=color, markersize=4, alpha=0.8)

        # Размечаем сигналы на ценовом графике
        for signal in signals:
            idx = len(closes) - 1
            ax1.plot(idx, signal['current_price'], 'v', color='red', markersize=15,
                     markeredgewidth=2, markeredgecolor='black', label='SHORT Signal')

            # Стоп-лосс и тейк-профит
            ax1.axhline(y=signal['stop_loss'], color='orange', linestyle='--', alpha=0.7, label='Stop Loss')
            ax1.axhline(y=signal['take_profit'], color='green', linestyle='--', alpha=0.7, label='Take Profit')

        ax1.set_title(f'Сигналы перекупленности RSI - {symbol}', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Цена')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Нижний график - RSI
        rsi_values = signals[0]['rsi_values'] if signals else self.calculate_rsi(closes)
        ax2.plot(range(len(rsi_values)), rsi_values, 'b-', linewidth=2, label='RSI')

        # Уровни перекупленности
        ax2.axhline(y=self.overbought_level, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax2.axhline(y=self.strong_overbought, color='darkred', linestyle='--', alpha=0.7,
                    label='Strong Overbought (80)')
        ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)

        ax2.set_ylabel('RSI')
        ax2.set_xlabel('Временные периоды')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def print_signal_details(self, signal):
        """Выводит детали сигнала"""
        print(f"\n{'=' * 60}")
        print(f"🎯 СИГНАЛ ШОРТ ПО ПЕРЕКУПЛЕННОСТИ RSI")
        print(f"{'=' * 60}")
        print(f"📊 Текущая цена: {signal['current_price']:.2f}")
        print(f"📈 Текущий RSI: {signal['current_rsi']:.1f}")
        print(f"💪 Сила сигнала: {signal['score']}/100")
        print(f"📉 Соотношение R/R: {signal['risk_reward_ratio']:.2f}")

        print(f"\n🎯 Торговые уровни:")
        print(f"   Вход: {signal['entry_price']:.2f}")
        print(f"   Стоп: {signal['stop_loss']:.2f} (+{((signal['stop_loss'] / signal['entry_price']) - 1) * 100:.2f}%)")
        print(
            f"   Тейк: {signal['take_profit']:.2f} (-{(1 - signal['take_profit'] / signal['entry_price']) * 100:.2f} %)")

        print(f"\n📋 Условия входа:")
        for condition in signal['conditions']:
            print(f"   ✓ {condition}")

        print(f"\n⏰ Время анализа: {signal['timestamp']}")

    def analyze_data(self, data, symbol):
        """Анализирует данные и выводит результаты"""
        signals = self.find_overbought_signals(data['data'])

        if signals:
            print(f"🎯 Найдено {len(signals)} сигналов перекупленности для {symbol}")

            for signal in signals:
                self.print_signal_details(signal)

            self.plot_rsi_signals(data['data'], signals, symbol)

            # Рекомендация
            best_signal = max(signals, key=lambda x: x['score'])
            if best_signal['score'] >= 80:
                print(f"\n🚀 ВЫСОКАЯ ВЕРОЯТНОСТЬ - НЕМЕДЛЕННЫЙ ВХОД В ШОРТ!")
            elif best_signal['score'] >= 65:
                print(f"\n✅ ХОРОШИЙ СИГНАЛ - МОЖНО ВХОДИТЬ В ШОРТ")
            else:
                print(f"\n⚠️  УМЕРЕННЫЙ СИГНАЛ - ОСТОРОЖНЫЙ ВХОД")

        else:
            print(f"❌ Сигналы перекупленности не найдены для {symbol}")

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

        analyzer = RSIOverboughtShort(
            rsi_period=14,
            overbought_level=70,
            strong_overbought=80
        )

        # Анализ данных
        analyzer.analyze_data(data, symbol)
