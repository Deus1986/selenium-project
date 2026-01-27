import time
import numpy as np
import pandas as pd
import requests
import talib
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


def get_high_volume_symbols(min_volume=20000000):
    """Получает список монет с высоким объемом"""
    url = "https://contract.mexc.com/api/v1/contract/ticker"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        symbols = []

        if 'data' in data:
            for item in data["data"]:
                if item["amount24"] > min_volume:
                    symbols.append({
                        'symbol': item['symbol'],
                        'volume_24h': item['amount24'],
                        'price_change_percent': float(item['riseFallRate']) * 100,
                        'last_price': float(item['lastPrice'])
                    })

        return sorted(symbols, key=lambda x: x['volume_24h'], reverse=True)
    except Exception as e:
        print(f"Ошибка получения списка монет: {e}")
        return []


class TwoCandlePatternScanner:
    def __init__(self):
        self.min_volume = 20000000

    def get_candles(self, symbol, interval="Min60", limit=10):
        """Получение данных - нужно всего 10 свечей для анализа последних двух"""
        url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
        params = {"interval": interval, "limit": limit}

        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                return data
        except Exception as e:
            print(f"Ошибка получения данных для {symbol}: {e}")
        return None

    def create_dataframe(self, data):
        """Создание DataFrame из данных MEXC"""
        if not data or not data.get('success') or not data.get('data'):
            return None

        raw_data = data['data']

        try:
            if isinstance(raw_data, dict):
                df = pd.DataFrame({
                    'timestamp': raw_data['time'],
                    'open': raw_data['open'],
                    'high': raw_data['high'],
                    'low': raw_data['low'],
                    'close': raw_data['close'],
                    'volume': raw_data['vol']
                })
            elif isinstance(raw_data, list):
                df = pd.DataFrame(raw_data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume'
                ])
            else:
                return None

            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna()

            if len(df) < 2:  # Нужно минимум 2 свечи
                return None

            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('datetime').reset_index(drop=True)

            return df

        except Exception as e:
            print(f"Ошибка создания DataFrame: {e}")
            return None

    def analyze_two_candle_pattern(self, df):
        """Анализ паттерна из двух свечей"""
        if len(df) < 2:
            return None

        try:
            # Берем последние две свечи
            current_candle = df.iloc[-1]  # Текущая (вторая) свеча
            prev_candle = df.iloc[-2]  # Предыдущая (первая) свеча

            # Определяем тип свечей
            prev_is_bullish = prev_candle['close'] > prev_candle['open']
            prev_is_bearish = prev_candle['close'] < prev_candle['open']

            current_is_bullish = current_candle['close'] > current_candle['open']
            current_is_bearish = current_candle['close'] < current_candle['open']

            # Размеры тел свечей
            prev_body_size = abs(prev_candle['close'] - prev_candle['open'])
            current_body_size = abs(current_candle['close'] - current_candle['open'])

            # Паттерн 1: Медвежья свеча → Бычья свеча (разворот вверх)
            if (prev_is_bearish and current_is_bullish and
                    current_body_size > prev_body_size):

                pattern_type = "BEARISH_TO_BULLISH_REVERSAL"
                direction = "LONG"
                confidence = self.calculate_pattern_confidence(df, "BULLISH")

                # Расчет уровней для лонга
                entry_price = current_candle['close']
                stop_loss = min(prev_candle['low'], current_candle['low'])
                take_profit = entry_price + (entry_price - stop_loss) * 2

                return {
                    'pattern_type': pattern_type,
                    'direction': direction,
                    'confidence': confidence,
                    'entry_price': round(entry_price, 6),
                    'stop_loss': round(stop_loss, 6),
                    'take_profit': round(take_profit, 6),
                    'risk_reward': round((take_profit - entry_price) / (entry_price - stop_loss), 2),
                    'prev_candle_type': 'BEARISH',
                    'current_candle_type': 'BULLISH',
                    'body_size_ratio': round(current_body_size / prev_body_size, 2),
                    'timestamp': datetime.now()
                }

            # Паттерн 2: Бычья свеча → Медвежья свеча (разворот вниз)
            elif (prev_is_bullish and current_is_bearish and
                  current_body_size > prev_body_size):

                pattern_type = "BULLISH_TO_BEARISH_REVERSAL"
                direction = "SHORT"
                confidence = self.calculate_pattern_confidence(df, "BEARISH")

                # Расчет уровней для шорта
                entry_price = current_candle['close']
                stop_loss = max(prev_candle['high'], current_candle['high'])
                take_profit = entry_price - (stop_loss - entry_price) * 2

                return {
                    'pattern_type': pattern_type,
                    'direction': direction,
                    'confidence': confidence,
                    'entry_price': round(entry_price, 6),
                    'stop_loss': round(stop_loss, 6),
                    'take_profit': round(take_profit, 6),
                    'risk_reward': round((entry_price - take_profit) / (stop_loss - entry_price), 2),
                    'prev_candle_type': 'BULLISH',
                    'current_candle_type': 'BEARISH',
                    'body_size_ratio': round(current_body_size / prev_body_size, 2),
                    'timestamp': datetime.now()
                }

            return None

        except Exception as e:
            print(f"Ошибка анализа паттерна: {e}")
            return None

    def calculate_pattern_confidence(self, df, direction):
        """Расчет уверенности в паттерне"""
        confidence = 50  # Базовая уверенность

        try:
            current_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]

            # Размер тела второй свечи значительно больше первой
            prev_body = abs(prev_candle['close'] - prev_candle['open'])
            current_body = abs(current_candle['close'] - current_candle['open'])

            if current_body > prev_body * 1.5:
                confidence += 20
            elif current_body > prev_body * 1.2:
                confidence += 10

            # Объем на второй свече
            if len(df) > 5:
                avg_volume = df['volume'].iloc[-6:-1].mean()
                if current_candle['volume'] > avg_volume * 1.2:
                    confidence += 15
                elif current_candle['volume'] > avg_volume:
                    confidence += 5

            # Положение закрытия относительно диапазона
            if direction == "BULLISH":
                if current_candle['close'] > current_candle['open'] and current_candle['close'] > prev_candle['close']:
                    confidence += 10
            else:  # BEARISH
                if current_candle['close'] < current_candle['open'] and current_candle['close'] < prev_candle['close']:
                    confidence += 10

        except Exception as e:
            print(f"Ошибка расчета уверенности: {e}")

        return min(confidence, 95)

    def analyze_symbol(self, symbol):
        """Анализ символа на паттерн двух свечей"""
        print(f"🔍 Анализ {symbol}...", end=" ")

        try:
            df = self.get_candles(symbol, "Min60", 10)  # Нужно всего 10 свечей
            if df is None:
                print("❌ Нет данных")
                return None

            df = self.create_dataframe(df)
            if df is None:
                print("❌ Не удалось создать DataFrame")
                return None

            pattern = self.analyze_two_candle_pattern(df)

            if pattern:
                pattern_name = pattern['pattern_type'].replace('_', ' ').title()
                print(f"✅ {pattern_name} (уверенность: {pattern['confidence']}%, R/R: {pattern['risk_reward']}:1)")
                return {
                    'symbol': symbol,
                    'pattern': pattern,
                    'data': df
                }
            else:
                print("❌ Паттерн не найден")
                return None

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return None

    def scan_for_two_candle_patterns(self, symbol_count=20):
        """Сканирование символов на паттерн двух свечей"""
        print("🎯 СКАНЕР ПАТТЕРНА ДВУХ СВЕЧЕЙ")
        print("=" * 70)
        print("🔍 Поиск ситуаций:")
        print("   • Медвежья свеча → Бычья свеча (тело бычьей > тела медвежьей)")
        print("   • Бычья свеча → Медвежья свеча (тело медвежьей > тела бычьей)")
        print("=" * 70)

        symbols_data = get_high_volume_symbols(min_volume=self.min_volume)
        symbols = [item['symbol'] for item in symbols_data[:symbol_count]]

        print(f"📊 Анализируем {len(symbols)} монет...")
        print("=" * 70)

        results = []

        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] ", end="")
            result = self.analyze_symbol(symbol)

            if result:
                results.append(result)

            time.sleep(0.2)

        # Сортировка по уверенности
        results.sort(key=lambda x: x['pattern']['confidence'], reverse=True)

        # Группировка по типам паттернов
        bullish_reversals = [r for r in results if "BEARISH_TO_BULLISH" in r['pattern']['pattern_type']]
        bearish_reversals = [r for r in results if "BULLISH_TO_BEARISH" in r['pattern']['pattern_type']]

        print(f"\n{'=' * 70}")
        print("📊 РЕЗУЛЬТАТЫ СКАНИРОВАНИЯ:")
        print(f"   • Всего паттернов: {len(results)}")
        print(f"   • Разворотов вверх: {len(bullish_reversals)}")
        print(f"   • Разворотов вниз: {len(bearish_reversals)}")

        if results:
            print(f"\n🎯 ЛУЧШИЕ ПАТТЕРНЫ:")

            for i, result in enumerate(results[:10], 1):
                pattern = result['pattern']
                direction_icon = "🟢" if pattern['direction'] == 'LONG' else "🔴"
                pattern_name = pattern['pattern_type'].replace('_', ' ').title()

                print(f"{i}. {direction_icon} {result['symbol']:15} | "
                      f"{pattern_name:<25} | "
                      f"Уверенность: {pattern['confidence']}% | "
                      f"R/R: {pattern['risk_reward']}:1 | "
                      f"Размер: x{pattern['body_size_ratio']}")

        return results


def main():
    """Основная функция"""
    scanner = TwoCandlePatternScanner()

    print("🎯 СКАНЕР ПАТТЕРНА ДВУХ СВЕЧЕЙ")
    print("=" * 60)
    print("🔍 Поиск разворотных паттернов на последних двух свечах")
    print("=" * 60)

    while True:
        print("\nВыберите действие:")
        print("1 - Сканировать топ-монеты")
        print("2 - Анализировать конкретную монету")
        print("3 - Выход")

        choice = input("\nВведите номер: ").strip()

        if choice == "1":
            count = int(input("Количество монет (10-30): ") or "20")
            scanner.scan_for_two_candle_patterns(count)

        elif choice == "2":
            symbol = input("Введите символ (например: BTC_USDT): ").strip().upper()
            result = scanner.analyze_symbol(symbol)

            if result:
                pattern = result['pattern']
                print(f"\n🎯 ПАТТЕРН ДЛЯ {symbol}:")
                print(f"   • Тип: {pattern['pattern_type'].replace('_', ' ')}")
                print(f"   • Направление: {pattern['direction']}")
                print(f"   • Первая свеча: {pattern['prev_candle_type']}")
                print(f"   • Вторая свеча: {pattern['current_candle_type']}")
                print(f"   • Точка входа: {pattern['entry_price']:.6f}")
                print(f"   • Стоп-лосс: {pattern['stop_loss']:.6f}")
                print(f"   • Тейк-профит: {pattern['take_profit']:.6f}")
                print(f"   • Соотношение риск/прибыль: {pattern['risk_reward']}:1")
                print(f"   • Уверенность: {pattern['confidence']}%")
                print(f"   • Соотношение размеров: x{pattern['body_size_ratio']}")
            else:
                print(f"   ❌ Паттерн не найден для {symbol}")

        elif choice == "3":
            print("Выход...")
            break

        else:
            print("Неверный выбор")


if __name__ == "__main__":
    main()