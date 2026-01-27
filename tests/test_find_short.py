import time
import numpy as np
import requests
import pandas as pd
from datetime import datetime


class ImmediateShortSignalFinder:
    def __init__(self):
        self.base_url = "https://contract.mexc.com/api/v1"

    def get_realtime_data(self, symbol):
        """Получает реальные данные для анализа"""
        try:
            # Текущие данные
            ticker_url = f"{self.base_url}/contract/ticker/{symbol}"
            ticker_response = requests.get(ticker_url, timeout=5)

            # Последние свечи
            kline_url = f"{self.base_url}/contract/kline/{symbol}"
            kline_params = {"interval": "Min5", "limit": 50}
            kline_response = requests.get(kline_url, params=kline_params, timeout=5)

            if (ticker_response.status_code == 200 and
                    kline_response.status_code == 200):

                ticker_data = ticker_response.json()
                kline_data = kline_response.json()

                if (ticker_data.get('success') and
                        kline_data.get('success')):
                    return {
                        'symbol': symbol,
                        'current_price': float(ticker_data['data']['lastPrice']),
                        'price_change': float(ticker_data['data']['changePercent']),
                        'high_24h': float(ticker_data['data']['high24Price']),
                        'low_24h': float(ticker_data['data']['low24Price']),
                        'volume_24h': float(ticker_data['data']['volume24']),
                        'funding_rate': float(ticker_data['data'].get('fundingRate', 0)),
                        'candles': kline_data['data'],
                        'timestamp': datetime.now().isoformat()
                    }

        except Exception as e:
            print(f"Ошибка получения данных для {symbol}: {e}")

        return None

    def calculate_technical_indicators(self, closes, highs, lows):
        """Рассчитывает технические индикаторы"""
        if len(closes) < 20:
            return {}

        closes = np.array(closes)
        highs = np.array(highs)
        lows = np.array(lows)

        # Простые скользящие средние
        sma_fast = pd.Series(closes).rolling(window=9).mean().iloc[-1]
        sma_slow = pd.Series(closes).rolling(window=21).mean().iloc[-1]

        # RSI (упрощенный расчет)
        gains = np.where(np.diff(closes) > 0, np.diff(closes), 0)
        losses = np.where(np.diff(closes) < 0, -np.diff(closes), 0)

        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 1
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 1

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # Процент от максимума
        current_price = closes[-1]
        high_24h = np.max(highs[-24:]) if len(highs) >= 24 else np.max(highs)
        from_high = ((current_price - high_24h) / high_24h) * 100

        # Моментум
        momentum = ((current_price - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 else 0

        # Объемный анализ
        volumes = [candle['vol'] for candle in self.current_candles] if hasattr(self, 'current_candles') else []
        volume_avg = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1] if volumes else 1
        volume_ratio = volumes[-1] / volume_avg if volume_avg > 0 else 1

        return {
            'sma_fast': sma_fast,
            'sma_slow': sma_slow,
            'rsi': rsi,
            'from_high': from_high,
            'momentum': momentum,
            'volume_ratio': volume_ratio,
            'price_vs_fast_sma': ((current_price - sma_fast) / sma_fast) * 100,
            'sma_crossover': sma_fast < sma_slow  # Медвежий кроссовер
        }

    def analyze_market_conditions(self, market_data):
        """Анализирует рыночные условия для шорта"""
        if not market_data:
            return None

        symbol = market_data['symbol']
        current_price = market_data['current_price']
        price_change = market_data['price_change']

        # Извлекаем данные свечей
        candles = market_data['candles']
        closes = [candle['close'] for candle in candles]
        highs = [candle['high'] for candle in candles]
        lows = [candle['low'] for candle in candles]

        # Сохраняем для volume анализа
        self.current_candles = candles

        # Рассчитываем индикаторы
        indicators = self.calculate_technical_indicators(closes, highs, lows)

        # Критерии для шорт входа
        conditions = []
        score = 0

        # 1. Цена в красной свече
        last_close = closes[-1] if len(closes) > 0 else current_price
        prev_close = closes[-2] if len(closes) > 1 else last_close
        is_red_candle = last_close < prev_close
        if is_red_candle:
            conditions.append("Красная свеча")
            score += 20

        # 2. RSI перекупленность
        if indicators['rsi'] > 65:
            conditions.append(f"RSI перекуплен ({indicators['rsi']:.1f})")
            score += 25

        # 3. Цена устала у максимумов
        if indicators['from_high'] > -2:  # В пределах 2% от максимума
            conditions.append(f"У максимумов ({indicators['from_high']:.1f}% от high)")
            score += 15

        # 4. Отрицательный моментум
        if indicators['momentum'] < 0:
            conditions.append(f"Отрицательный моментум ({indicators['momentum']:.1f}%)")
            score += 15

        # 5. Медвежий кроссовер SMA
        if indicators['sma_crossover']:
            conditions.append("Медвежий SMA кроссовер (9 < 21)")
            score += 20

        # 6. Высокий объем при падении
        if indicators['volume_ratio'] > 1.5 and is_red_candle:
            conditions.append(f"Высокий объем продаж (x{indicators['volume_ratio']:.1f})")
            score += 25

        # 7. Отрицательное изменение цены
        if price_change < 0:
            conditions.append(f"Цена падает ({price_change:.1f}%)")
            score += 15

        # 8. Положительное финансирование (выгодно для шорта)
        if market_data['funding_rate'] > 0.0001:
            conditions.append(f"Положительное финансирование ({market_data['funding_rate']:.4f})")
            score += 10

        # Дополнительные условия
        if len(closes) > 3:
            # Серия красных свечей
            red_candles = sum(1 for i in range(1, 4) if closes[-i] < closes[-i - 1])
            if red_candles >= 2:
                conditions.append(f"Серия из {red_candles} красных свечей")
                score += 10

            # Усиление нисходящего движения
            if all(closes[-i] < closes[-i - 1] for i in range(1, 3)):
                conditions.append("Усиление нисходящего тренда")
                score += 15

        # Рассчитываем уровни для торговли
        atr = np.mean([highs[-i] - lows[-i] for i in range(1, 6)]) if len(highs) >= 5 else current_price * 0.01

        entry_price = current_price
        stop_loss = current_price + (atr * 1.5)  # Стоп на 1.5 ATR
        take_profit = current_price - (atr * 2)  # Тейк на 2 ATR

        risk = stop_loss - entry_price
        reward = entry_price - take_profit
        rr_ratio = reward / risk if risk > 0 else 0

        return {
            'symbol': symbol,
            'signal': 'SHORT' if score >= 50 else 'NEUTRAL',
            'score': score,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': rr_ratio,
            'conditions': conditions,
            'current_price': current_price,
            'price_change': price_change,
            'volume_24h': market_data['volume_24h'],
            'timestamp': market_data['timestamp']
        }

    def find_immediate_short_signals(self, symbols):
        """Ищет немедленные сигналы для шорта"""
        immediate_signals = []
        while len(immediate_signals) == 0:
            for symbol in symbols:
                print(f"🔍 Анализируем {symbol}...")

                market_data = self.get_realtime_data(symbol)
                if not market_data:
                    continue

                analysis = self.analyze_market_conditions(market_data)
                if not analysis:
                    continue

                if analysis['signal'] == 'SHORT' and analysis['score'] >= 60:
                    immediate_signals.append(analysis)
                    print(f"   ✅ Сигнал SHORT (сила: {analysis['score']})")
                else:
                    print(f"   ❌ Нет сигнала (сила: {analysis['score']})")

                time.sleep(0.5)  # Пауза между запросами

        return immediate_signals

    def get_high_volume_symbols(self, min_volume=10000000, limit=700):
        """Получает символы с высоким объемом"""
        try:
            url = f"{self.base_url}/contract/ticker"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    # Фильтруем по объему и сортируем
                    high_volume_coins = [
                        coin['symbol'] for coin in data['data']
                        if coin.get('amount24', 0) > min_volume
                    ]
                    return high_volume_coins[:limit]

        except Exception as e:
            print(f"Ошибка получения списка монет: {e}")

        return []

    def print_signal_details(self, signal):
        """Выводит детали сигнала"""
        print(f"\n🎯 СИГНАЛ НЕМЕДЛЕННОГО ШОРТА: {signal['symbol']}")
        print(f"📊 Текущая цена: {signal['current_price']:.4f}")
        print(f"📈 Изменение: {signal['price_change']:.2f}%")
        print(f"💪 Сила сигнала: {signal['score']}/100")
        print(f"📉 Объем 24ч: ${signal['volume_24h']:,.0f}")

        print(f"\n🎯 Торговые уровни:")
        print(f"   Вход: {signal['entry_price']:.4f}")
        print(f"   Стоп: {signal['stop_loss']:.4f} (+{((signal['stop_loss'] / signal['entry_price']) - 1) * 100:.1f}%)")
        print(
            f"   Тейк: {signal['take_profit']:.4f} (-{((signal['entry_price'] / signal['take_profit']) - 1) * 100:.1f}%)")
        print(f"   R/R: {signal['risk_reward_ratio']:.2f}")

        print(f"\n📋 Условия входа ({len(signal['conditions'])}):")
        for condition in signal['conditions']:
            print(f"   ✓ {condition}")

        print(f"\n⏰ Время анализа: {signal['timestamp']}")
        print("=" * 60)


# Основная функция
def test_main():
    print("🚀 Поиск немедленных сигналов для шорт входа...")
    print("=" * 60)

    # Инициализация
    finder = ImmediateShortSignalFinder()

    # Получаем монеты с высоким объемом
    symbols = finder.get_high_volume_symbols(min_volume=1000000)

    if not symbols:
        print("❌ Не удалось получить список монет")
        return

    print(f"📊 Анализируем {len(symbols)} монет с высоким объемом:")
    for symbol in symbols:
        print(f"   • {symbol}")

    print("\n" + "=" * 60)

    # Ищем сигналы
    short_signals = finder.find_immediate_short_signals(symbols)

    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ ПОИСКА:")
    print("=" * 60)

    if short_signals:
        # Сортируем по силе сигнала
        short_signals.sort(key=lambda x: x['score'], reverse=True)

        print(f"🎯 Найдено {len(short_signals)} сигналов для немедленного шорта:")

        for i, signal in enumerate(short_signals, 1):
            print(f"\n{i}. {signal['symbol']} - Сила: {signal['score']}/100")
            finder.print_signal_details(signal)

            # Рекомендация к действию
            if signal['score'] >= 80:
                print("   🚀 ВЫСОКАЯ ВЕРОЯТНОСТЬ - НЕМЕДЛЕННЫЙ ВХОД!")
            elif signal['score'] >= 70:
                print("   ✅ СИЛЬНЫЙ СИГНАЛ - РЕКОМЕНДУЕТСЯ ВХОД")
            else:
                print("   ⚠️  УМЕРЕННЫЙ СИГНАЛ - ОСТОРОЖНЫЙ ВХОД")

    else:
        print("❌ Сигналов для немедленного шорта не найдено")
        print("   Рынок может быть в восходящем тренде или консолидации")
