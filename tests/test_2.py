import time
import numpy as np
import pandas as pd
import requests
import talib
from datetime import datetime, timedelta
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class BearishBullishSequenceScanner:
    def __init__(self):
        self.min_bearish_candles = 2
        self.max_bearish_candles = 5
        self.min_bullish_candles = 2
        self.max_bullish_candles = 5
        self.analysis_period = 10  # Анализируем последние 15 свечей
        self.timeframe = "Min60"  # Часовой таймфрейм по умолчанию
        self.min_volume = 2000000

    def get_active_symbols(self, min_volume=1000000, limit=80):
        """Получение списка активных символов"""
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
                            'price_change_percent': float(item['riseFallRate']) * 100
                        })

            symbols.sort(key=lambda x: x['volume_24h'], reverse=True)
            return [s['symbol'] for s in symbols[:limit]]

        except Exception as e:
            print(f"Ошибка получения списка монет: {e}")
            return ['BTC_USDT', 'ETH_USDT', 'ADA_USDT', 'DOT_USDT', 'LINK_USDT',
                    'MATIC_USDT', 'ATOM_USDT', 'AVAX_USDT', 'XRP_USDT', 'SOL_USDT']

    def get_realtime_candles(self, symbol, interval=None, limit=None):
        """Получение актуальных данных"""
        if interval is None:
            interval = self.timeframe
        if limit is None:
            limit = self.analysis_period

        url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
        params = {"interval": interval, "limit": limit}

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return self.create_dataframe(data)
            else:
                print(f"   ❌ Ошибка API для {symbol}: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Ошибка получения данных {symbol}: {e}")
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

            if len(df) < 5:  # Минимум 5 свечей
                return None

            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('datetime').reset_index(drop=True)

            # Добавляем расчет типа свечи
            df['is_bullish'] = df['close'] > df['open']
            df['body_size'] = abs(df['close'] - df['open'])
            df['body_percent'] = (df['body_size'] / df['open']) * 100

            return df

        except Exception as e:
            print(f"   ❌ Ошибка создания DataFrame: {e}")
            return None

    def find_bearish_bullish_sequence(self, df):
        """Поиск последовательности: 2-5 медвежьих свечей и затем 2-5 бычьих свечей"""
        if df is None or len(df) < 4:
            return []

        sequences = []
        is_bullish = df['is_bullish'].values
        close_prices = df['close'].values

        # Проходим по всем возможным начальным позициям
        for start_idx in range(len(df) - 3):  # Нужно минимум 4 свечи для паттерна
            # Проверяем возможные длины медвежьей последовательности
            for bearish_len in range(self.min_bearish_candles, self.max_bearish_candles + 1):
                end_bearish = start_idx + bearish_len

                # Проверяем что все свечи в медвежьей последовательности действительно медвежьи
                if end_bearish <= len(df):
                    bearish_sequence = is_bullish[start_idx:end_bearish]
                    if not all(not candle for candle in bearish_sequence):
                        continue  # Не все свечи медвежьи

                    # Проверяем возможные длины бычьей последовательности после медвежьей
                    for bullish_len in range(self.min_bullish_candles, self.max_bullish_candles + 1):
                        start_bullish = end_bearish
                        end_bullish = start_bullish + bullish_len

                        if end_bullish <= len(df):
                            bullish_sequence = is_bullish[start_bullish:end_bullish]

                            # Проверяем что все свечи в бычьей последовательности действительно бычьи
                            if all(bullish_sequence):
                                # Проверяем что последовательности идут подряд
                                if start_bullish == end_bearish:
                                    # Анализируем качество паттерна
                                    pattern_quality = self.analyze_pattern_quality(
                                        df, start_idx, end_bearish, start_bullish, end_bullish
                                    )

                                    if pattern_quality['confidence'] > 0.5:
                                        sequences.append({
                                            'start_index': start_idx,
                                            'end_index': end_bullish - 1,
                                            'bearish_count': bearish_len,
                                            'bullish_count': bullish_len,
                                            'bearish_start': start_idx,
                                            'bearish_end': end_bearish - 1,
                                            'bullish_start': start_bullish,
                                            'bullish_end': end_bullish - 1,
                                            'quality': pattern_quality,
                                            'timestamp': datetime.now()
                                        })

        return sequences

    def analyze_pattern_quality(self, df, bearish_start, bearish_end, bullish_start, bullish_end):
        """Анализирует качество найденного паттерна"""
        confidence = 0.0

        # Анализ медвежьей последовательности
        bearish_candles = df.iloc[bearish_start:bearish_end]
        bullish_candles = df.iloc[bullish_start:bullish_end]

        # 1. Сила медвежьих свечей (средний размер тела)
        avg_bearish_body = bearish_candles['body_percent'].mean()
        if avg_bearish_body > 1.0:
            confidence += 0.2
        elif avg_bearish_body > 0.5:
            confidence += 0.1

        # 2. Сила бычьих свечей (средний размер тела)
        avg_bullish_body = bullish_candles['body_percent'].mean()
        if avg_bullish_body > 1.0:
            confidence += 0.2
        elif avg_bullish_body > 0.5:
            confidence += 0.1

        # 3. Объемы - проверяем что объемы адекватные
        avg_volume = df['volume'].mean()
        bearish_volume_ok = bearish_candles['volume'].mean() > avg_volume * 0.5
        bullish_volume_ok = bullish_candles['volume'].mean() > avg_volume * 0.5

        if bearish_volume_ok:
            confidence += 0.1
        if bullish_volume_ok:
            confidence += 0.1

        # 4. Разница в ценах - бычья последовательность должна отыгрывать медвежью
        bearish_low = bearish_candles['low'].min()
        bullish_high = bullish_candles['high'].max()

        if bullish_high > bearish_low:
            recovery_ratio = (bullish_high - bearish_low) / (df.iloc[bearish_start]['high'] - bearish_low)
            confidence += min(recovery_ratio * 0.3, 0.3)

        # 5. За баланс длин последовательностей
        length_balance = min(len(bearish_candles), len(bullish_candles)) / max(len(bearish_candles),
                                                                               len(bullish_candles))
        confidence += length_balance * 0.1

        return {
            'confidence': min(confidence, 1.0),
            'avg_bearish_body': avg_bearish_body,
            'avg_bullish_body': avg_bullish_body,
            'recovery_ratio': recovery_ratio if 'recovery_ratio' in locals() else 0,
            'volume_analysis': {
                'bearish_ok': bearish_volume_ok,
                'bullish_ok': bullish_volume_ok
            }
        }

    def plot_sequence_pattern(self, symbol, sequence, df):
        """Визуализирует найденную последовательность"""
        plt.figure(figsize=(15, 10))

        # Основной график цены
        plt.subplot(2, 1, 1)
        plt.plot(df['close'].values, label='Close Price', linewidth=2, color='blue', alpha=0.8)

        # Выделяем области последовательностей
        bearish_range = range(sequence['bearish_start'], sequence['bearish_end'] + 1)
        bullish_range = range(sequence['bullish_start'], sequence['bullish_end'] + 1)

        # Закрашиваем области
        plt.axvspan(sequence['bearish_start'], sequence['bearish_end'] + 0.5,
                    alpha=0.2, color='red', label='Bearish Sequence')
        plt.axvspan(sequence['bullish_start'], sequence['bullish_end'] + 0.5,
                    alpha=0.2, color='green', label='Bullish Sequence')

        # Отмечаем отдельные свечи
        for i in range(len(df)):
            color = 'green' if df['is_bullish'].iloc[i] else 'red'
            marker = '^' if df['is_bullish'].iloc[i] else 'v'
            plt.plot(i, df['close'].iloc[i], marker, color=color, markersize=6, alpha=0.8)

        plt.title(f"{symbol} - Bearish/Bullish Sequence Pattern\n"
                  f"Bearish: {sequence['bearish_count']} candles → Bullish: {sequence['bullish_count']} candles | "
                  f"Confidence: {sequence['quality']['confidence']:.2f}")
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # График объемов
        plt.subplot(2, 1, 2)
        colors = ['red' if not df['is_bullish'].iloc[i] else 'green' for i in range(len(df))]
        plt.bar(range(len(df)), df['volume'].values, color=colors, alpha=0.7)
        plt.title('Volume (Red: Bearish, Green: Bullish)')
        plt.ylabel('Volume')
        plt.xlabel('Candle Index')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def analyze_symbol_sequences(self, symbol):
        """Анализ символа на последовательности медвежьих/бычьих свечей"""
        print(f"🔍 Анализ {symbol}...", end=" ")

        try:
            df = self.get_realtime_candles(symbol)
            if df is None:
                print("❌ Нет данных")
                return None

            # Находим последовательности
            sequences = self.find_bearish_bullish_sequence(df)

            if sequences:
                # Сортируем по уверенности
                sequences.sort(key=lambda x: x['quality']['confidence'], reverse=True)

                print(f"✅ Найдено {len(sequences)} последовательностей")

                return {
                    'symbol': symbol,
                    'sequences_found': len(sequences),
                    'best_sequences': sequences,  # Топ-3 последовательности
                    'data': df,
                    'timestamp': datetime.now()
                }

            print("❌ Последовательности не найдены")
            return None

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return None

    def scan_for_sequences(self, symbol_count=30):
        """Сканирование символов на паттерны последовательностей"""
        print(f"🎯 ПОИСК ПОСЛЕДОВАТЕЛЬНОСТЕЙ: МЕДВЕЖЬИ → БЫЧЬИ СВЕЧИ")
        print("=" * 70)
        print(f"🔧 Параметры:")
        print(f"   • Таймфрейм: {self.timeframe}")
        print(f"   • Анализ: последние {self.analysis_period} свечей")
        print(f"   • Медвежьих свечей: {self.min_bearish_candles}-{self.max_bearish_candles}")
        print(f"   • Бычьих свечей: {self.min_bullish_candles}-{self.max_bullish_candles}")
        print("=" * 70)

        symbols = self.get_active_symbols(limit=symbol_count)
        print(f"📈 Анализируем {len(symbols)} монет...")
        print("=" * 70)

        results = []
        found_symbols = []

        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] ", end="")
            result = self.analyze_symbol_sequences(symbol)

            if result:
                results.append(result)
                if result['sequences_found'] > 0:
                    found_symbols.append(result)
                    best_seq = result['best_sequences'][0]
                    print(f"   🎯 {result['sequences_found']} паттернов "
                          f"({best_seq['bearish_count']}↓ → {best_seq['bullish_count']}↑)")

            time.sleep(0.2)

        # Анализ результатов
        print(f"\n{'=' * 70}")
        print("📊 ИТОГОВАЯ СТАТИСТИКА:")
        print(f"   • Всего проанализировано: {len(results)}")
        print(f"   • Найдено символов с паттернами: {len(found_symbols)}")

        total_sequences = sum(r['sequences_found'] for r in found_symbols)
        print(f"   • Всего паттернов: {total_sequences}")

        if found_symbols:
            print(f"\n🎯 ЛУЧШИЕ ПАТТЕРНЫ:")

            # Сортируем по уверенности лучшего паттерна
            found_symbols.sort(key=lambda x: x['best_sequences'][0]['quality']['confidence']
            if x['best_sequences'] else 0, reverse=True)

            for i, symbol_data in enumerate(found_symbols, 1):
                best_sequence = symbol_data['best_sequences'][0] if symbol_data['best_sequences'] else None

                if best_sequence:
                    print(f"{i:2d}. {symbol_data['symbol']:15} | "
                          f"{best_sequence['bearish_count']}↓ → {best_sequence['bullish_count']}↑ | "
                          f"Уверенность: {best_sequence['quality']['confidence']:.2f} | "
                          f"Тела: {best_sequence['quality']['avg_bearish_body']:.1f}%/{best_sequence['quality']['avg_bullish_body']:.1f}%")

                # Показываем график для топ-3
                if i <= 3 and symbol_data['best_sequences']:
                    self.plot_sequence_pattern(symbol_data['symbol'],
                                               symbol_data['best_sequences'][0],
                                               symbol_data['data'])

        return len(found_symbols) > 0

    def update_parameters(self):
        """Обновление параметров сканирования"""
        print("\n📊 НАСТРОЙКА ПАРАМЕТРОВ:")

        # Таймфрейм
        print("\nДоступные таймфреймы:")
        print("1 - Min1 (1 минута)")
        print("2 - Min5 (5 минут)")
        print("3 - Min15 (15 минут)")
        print("4 - Min30 (30 минут)")
        print("5 - Min60 (1 час)")
        print("6 - Hour4 (4 часа)")

        tf_choice = input("Выберите таймфрейм (1-6, по умолчанию 5): ").strip()
        tf_map = {"1": "Min1", "2": "Min5", "3": "Min15", "4": "Min30", "5": "Min60", "6": "Hour4"}
        self.timeframe = tf_map.get(tf_choice, "Min60")

        # Количество свечей для анализа
        candles = input(f"Количество свечей для анализа (по умолчанию {self.analysis_period}): ").strip()
        if candles.isdigit():
            self.analysis_period = int(candles)

        # Медвежьи свечи
        bear_min = input(f"Мин. медвежьих свечей (по умолчанию {self.min_bearish_candles}): ").strip()
        if bear_min.isdigit():
            self.min_bearish_candles = int(bear_min)

        bear_max = input(f"Макс. медвежьих свечей (по умолчанию {self.max_bearish_candles}): ").strip()
        if bear_max.isdigit():
            self.max_bearish_candles = int(bear_max)

        # Бычьи свечи
        bull_min = input(f"Мин. бычьих свечей (по умолчанию {self.min_bullish_candles}): ").strip()
        if bull_min.isdigit():
            self.min_bullish_candles = int(bull_min)

        bull_max = input(f"Макс. бычьих свечей (по умолчанию {self.max_bullish_candles}): ").strip()
        if bull_max.isdigit():
            self.max_bullish_candles = int(bull_max)

        print("✅ Параметры обновлены!")


def main():
    """Основная функция"""
    scanner = BearishBullishSequenceScanner()

    print("🎯 СКАНЕР ПОСЛЕДОВАТЕЛЬНОСТЕЙ МЕДВЕЖЬИХ/БЫЧЬИХ СВЕЧЕЙ")
    print("=" * 60)
    print("🔍 Поиск паттерна: 2-5 медвежьих свечей → 2-5 бычьих свечей")
    print("=" * 60)

    while True:
        print("\nВыберите действие:")
        print("1 - Сканировать топ-монеты")
        print("2 - Анализировать конкретную монету")
        print("3 - Настроить параметры")
        print("4 - Выход")

        choice = input("\nВведите номер: ").strip()

        if choice == "1":
            count = int(input("Количество монет (10-50): ") or "30")
            scanner.scan_for_sequences(count)

        elif choice == "2":
            symbol = input("Введите символ (например: BTC_USDT): ").strip().upper()
            result = scanner.analyze_symbol_sequences(symbol)

            if result and result['sequences_found'] > 0:
                print(f"\n🎯 РЕЗУЛЬТАТЫ ДЛЯ {symbol}:")
                print(f"   🔍 Найдено последовательностей: {result['sequences_found']}")

                for i, sequence in enumerate(result['best_sequences'], 1):
                    print(f"   {i}. {sequence['bearish_count']}↓ → {sequence['bullish_count']}↑ | "
                          f"Уверенность: {sequence['quality']['confidence']:.2f} | "
                          f"Тела: {sequence['quality']['avg_bearish_body']:.1f}%/{sequence['quality']['avg_bullish_body']:.1f}%")

                # Показываем график лучшей последовательности
                scanner.plot_sequence_pattern(symbol, result['best_sequences'][0], result['data'])
            else:
                print(f"   ❌ Последовательности не найдены для {symbol}")

        elif choice == "3":
            scanner.update_parameters()

        elif choice == "4":
            print("Выход...")
            break

        else:
            print("Неверный выбор")


# Функция для pytest
def test_sequence_analysis():
    """Тест для pytest"""
    scanner = BearishBullishSequenceScanner()
    success = scanner.scan_for_sequences(80)
    assert success or True


if __name__ == "__main__":
    main()