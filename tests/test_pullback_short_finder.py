import time
import numpy as np
import requests
import pandas as pd
from datetime import datetime


class PullbackShortFinder:
    def __init__(self):
        self.base_url = "https://contract.mexc.com/api/v1"

    def get_realtime_data(self, symbol, interval="Min5", limit=500):
        """Получает реальные данные для анализа"""
        try:
            params = {
                "interval": interval,
                "limit": limit
            }

            response = requests.get(f"{self.base_url}/contract/kline/{symbol}",
                                    params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    # Правильно извлекаем данные из формата MEXC
                    return data['data']

        except Exception as e:
            print(f"Ошибка получения данных для {symbol}: {e}")

        return None

    def detect_recent_extremum(self, highs, lows, lookback_period=20):
        """Обнаруживает недавний экстремум"""
        if len(highs) < lookback_period:
            return None

        # Ищем максимум за последний период
        recent_highs = highs[-lookback_period:]
        extremum_index = np.argmax(recent_highs) + (len(highs) - lookback_period)
        extremum_price = highs[extremum_index]

        return {
            'index': extremum_index,
            'price': extremum_price,
            'type': 'high',
            'time_period': lookback_period
        }

    def calculate_pullback_levels(self, extremum, current_price, lows):
        """Рассчитывает уровни отката"""
        extremum_price = extremum['price']

        # Находим минимум между экстремумом и текущим моментом
        recent_lows = lows[extremum['index']:]
        if len(recent_lows) > 0:
            min_low = min(recent_lows)
        else:
            min_low = current_price * 0.9  # fallback

        # Уровни Фибоначчи отката
        fib_levels = {
            '0.0': extremum_price,
            '0.236': extremum_price - (extremum_price - min_low) * 0.236,
            '0.382': extremum_price - (extremum_price - min_low) * 0.382,
            '0.5': extremum_price - (extremum_price - min_low) * 0.5,
            '0.618': extremum_price - (extremum_price - min_low) * 0.618,
            '0.786': extremum_price - (extremum_price - min_low) * 0.786,
            '1.0': min_low
        }

        return fib_levels

    def is_pullback_to_short_zone(self, current_price, fib_levels):
        """Проверяет, находится ли цена в зоне для шорта"""
        # Зона для шорта: от 0.382 до 0.618 Фибо
        short_zone_low = fib_levels['0.382']
        short_zone_high = fib_levels['0.618']

        return short_zone_low <= current_price <= short_zone_high

    def analyze_pullback_short(self, symbol):
        """Анализирует откат для шорт входа"""
        print(f"🔍 Анализируем откат для шорта: {symbol}")

        # Получаем данные
        data = self.get_realtime_data(symbol, interval="Min15", limit=50)
        if not data:
            print(f"   ❌ Нет данных для {symbol}")
            return None

        # Правильно извлекаем данные из формата MEXC
        try:
            # MEXC возвращает данные в формате {'time': [], 'high': [], 'low': [], ...}
            highs = data['high']
            lows = data['low']
            closes = data['close']
            current_price = closes[-1] if closes else None

            if not current_price:
                print(f"   ❌ Нет текущей цены для {symbol}")
                return None

        except (KeyError, TypeError) as e:
            print(f"   ❌ Ошибка формата данных для {symbol}: {e}")
            return None

        # Обнаруживаем экстремум
        extremum = self.detect_recent_extremum(highs, lows)
        if not extremum:
            print(f"   ❌ Не найден экстремум для {symbol}")
            return None

        # Рассчитываем уровни Фибо
        fib_levels = self.calculate_pullback_levels(extremum, current_price, lows)

        # Проверяем, находится ли цена в зоне для шорта
        in_short_zone = self.is_pullback_to_short_zone(current_price, fib_levels)

        if not in_short_zone:
            print(f"   ❌ Цена не в зоне шорта для {symbol}")
            return None

        # Анализируем текущие условия
        analysis = self.analyze_current_conditions(data, extremum, fib_levels, symbol)

        return analysis

    def analyze_current_conditions(self, data, extremum, fib_levels, symbol):
        """Анализирует текущие рыночные условия"""
        try:
            highs = data['high']
            lows = data['low']
            closes = data['close']
            volumes = data['vol']

            current_price = closes[-1]
            current_volume = volumes[-1]
            avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else current_volume

            # Технические индикаторы
            sma_fast = pd.Series(closes).rolling(window=9).mean().iloc[-1]
            sma_slow = pd.Series(closes).rolling(window=21).mean().iloc[-1]

            # Анализ свечей
            last_close = closes[-1]
            prev_close = closes[-2] if len(closes) > 1 else last_close
            is_red_candle = last_close < prev_close

            # Сигналы
            signals = []
            score = 0

            # 1. Цена в зоне Фибо 0.382-0.618
            signals.append(f"Цена в зоне Фибо для шорта")
            score += 30

            # 2. Медвежий кроссовер SMA
            if sma_fast < sma_slow:
                signals.append("Медвежий SMA кроссовер")
                score += 20

            # 3. Красная свеча
            if is_red_candle:
                signals.append("Красная свеча")
                score += 15

            # 4. Высокий объем
            if current_volume > avg_volume * 1.5:
                signals.append("Высокий объем продаж")
                score += 20

            # 5. Цена ниже экстремума
            if current_price < extremum['price']:
                signals.append("Цена ниже экстремума")
                score += 10

            # Рассчитываем уровни для торговли
            atr = np.mean([highs[-i] - lows[-i] for i in range(1, 6)]) if len(highs) >= 5 else current_price * 0.01

            entry_price = current_price
            stop_loss = current_price + (atr * 1.5)  # Стоп выше
            take_profit = current_price - (atr * 2)  # Тейк ниже

            risk = stop_loss - entry_price
            reward = entry_price - take_profit
            rr_ratio = reward / risk if risk > 0 else 0

            return {
                'symbol': symbol,
                'signal': 'SHORT_PULLBACK' if score >= 50 else 'NEUTRAL',
                'score': score,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': rr_ratio,
                'signals': signals,
                'current_price': current_price,
                'extremum_price': extremum['price'],
                'pullback_depth': ((extremum['price'] - current_price) / extremum['price']) * 100,
                'fib_levels': fib_levels,
                'volume_ratio': current_volume / avg_volume,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"   ❌ Ошибка анализа для {symbol}: {e}")
            return None

    def find_immediate_pullback_shorts(self, symbols):
        """Ищет немедленные сигналы отката для шорта"""
        immediate_signals = []

        for symbol in symbols:
            analysis = self.analyze_pullback_short(symbol)

            if analysis and analysis['signal'] == 'SHORT_PULLBACK' and analysis['score'] >= 60:
                immediate_signals.append(analysis)
                print(f"   ✅ Откат для шорта найден (сила: {analysis['score']})")
            elif analysis:
                print(f"   ❌ Откат не подходит (сила: {analysis['score']})")
            else:
                print(f"   ❌ Откат не найден")

            time.sleep(0.3)

        return immediate_signals

    def get_high_volume_symbols(self, min_volume=5000000, limit=15):
        """Получает символы с высоким объемом"""
        try:
            url = f"{self.base_url}/contract/ticker"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    high_volume_coins = [
                        coin['symbol'] for coin in data['data']
                        if coin.get('amount24', 0) > min_volume
                    ]
                    return high_volume_coins[:limit]

        except Exception as e:
            print(f"Ошибка получения списка монет: {e}")

        return []

    def print_pullback_signal(self, signal):
        """Выводит детали сигнала отката"""
        print(f"\n🎯 СИГНАЛ ОТКАТА ДЛЯ ШОРТА: {signal['symbol']}")
        print(f"📊 Текущая цена: {signal['current_price']:.4f}")
        print(f"📉 Глубина отката: {signal['pullback_depth']:.1f}% от экстремума")
        print(f"🏔️  Цена экстремума: {signal['extremum_price']:.4f}")
        print(f"💪 Сила сигнала: {signal['score']}/100")
        print(f"📈 Соотношение R/R: {signal['risk_reward_ratio']:.2f}")
        print(f"📊 Объем: x{signal['volume_ratio']:.1f} от среднего")

        print(f"\n🎯 Уровни Фибоначчи:")
        for level, price in signal['fib_levels'].items():
            print(f"   {level}: {price:.4f}")

        print(f"\n🎯 Торговые уровни:")
        print(f"   Вход: {signal['entry_price']:.4f}")
        print(f"   Стоп: {signal['stop_loss']:.4f} (+{((signal['stop_loss'] / signal['entry_price']) - 1) * 100:.1f}%)")
        print(
            f"   Тейк: {signal['take_profit']:.4f} (-{(1 - signal['take_profit'] / signal['entry_price']) * 100:.1f}%)")

        print(f"\n📋 Сигналы входа ({len(signal['signals'])}):")
        for signal_text in signal['signals']:
            print(f"   ✓ {signal_text}")

        print(f"\n⏰ Время анализа: {signal['timestamp']}")
        print("=" * 60)


# Тестовая функция
def test_main_pullback_short():
    print("🚀 Поиск откатов для немедленного шорт входа...")
    print("=" * 60)

    finder = PullbackShortFinder()

    # Получаем монеты с высоким объемом (меньше для теста)
    symbols = finder.get_high_volume_symbols(min_volume=10000000, limit=5)

    if not symbols:
        print("❌ Не удалось получить список монет")
        return

    print(f"📊 Анализируем {len(symbols)} монет:")
    for symbol in symbols:
        print(f"   • {symbol}")

    print("\n" + "=" * 60)

    # Ищем сигналы отката
    pullback_signals = finder.find_immediate_pullback_shorts(symbols)

    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ ПОИСКА ОТКАТОВ:")
    print("=" * 60)

    if pullback_signals:
        pullback_signals.sort(key=lambda x: x['score'], reverse=True)

        print(f"🎯 Найдено {len(pullback_signals)} откатов для шорта:")

        for i, signal in enumerate(pullback_signals, 1):
            print(f"\n{i}. {signal['symbol']} - Сила: {signal['score']}/100")
            finder.print_pullback_signal(signal)

            if signal['score'] >= 75:
                print("   🚀 ВЫСОКАЯ ВЕРОЯТНОСТЬ - НЕМЕДЛЕННЫЙ ВХОД!")
            elif signal['score'] >= 60:
                print("   ✅ ХОРОШИЙ СИГНАЛ - МОЖНО ВХОДИТЬ")

    else:
        print("❌ Откатов для шорта не найдено")
        print("   Ищите возможности после новых экстремумов")
