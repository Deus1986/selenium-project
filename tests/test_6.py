import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
from typing import List, Dict, Optional

warnings.filterwarnings('ignore')


class HighVolumeBullishScanner:
    def __init__(self, volume_multiplier=3.0, min_price_change=1.0):
        self.volume_multiplier = volume_multiplier  # Во сколько раз объем выше среднего
        self.min_price_change = min_price_change  # Минимальное изменение цены в %
        self.base_url = "https://api.binance.com/api/v3"

    def get_top_volume_symbols(self, limit: int = 50) -> List[Dict]:
        """Получение топовых символов по объему"""
        print("📈 Получение списка ликвидных монет...")

        try:
            url = f"{self.base_url}/ticker/24hr"
            response = requests.get(url, timeout=10)
            data = response.json()

            usdt_pairs = []
            for item in data:
                symbol = item['symbol']
                if symbol.endswith('USDT') and not any(x in symbol for x in ['UP', 'DOWN', 'BULL', 'BEAR']):
                    quote_volume = float(item['quoteVolume'])
                    if quote_volume > 10000000:  # Минимум 10M объема
                        usdt_pairs.append({
                            'symbol': symbol,
                            'quote_volume': quote_volume,
                            'price_change_percent': float(item['priceChangePercent']),
                            'last_price': float(item['lastPrice'])
                        })

            # Сортируем по объему и берем топ
            usdt_pairs.sort(key=lambda x: x['quote_volume'], reverse=True)
            return usdt_pairs[:limit]

        except Exception as e:
            print(f"❌ Ошибка получения списка монет: {e}")
            return self.get_fallback_symbols()

    def get_fallback_symbols(self) -> List[Dict]:
        """Резервный список популярных пар"""
        return [
            {'symbol': 'BTCUSDT', 'quote_volume': 25000000000},
            {'symbol': 'ETHUSDT', 'quote_volume': 12000000000},
            {'symbol': 'BNBUSDT', 'quote_volume': 3000000000},
            {'symbol': 'ADAUSDT', 'quote_volume': 1500000000},
            {'symbol': 'XRPUSDT', 'quote_volume': 3500000000},
            {'symbol': 'DOGEUSDT', 'quote_volume': 2000000000},
            {'symbol': 'SOLUSDT', 'quote_volume': 1500000000},
            {'symbol': 'DOTUSDT', 'quote_volume': 1200000000},
            {'symbol': 'MATICUSDT', 'quote_volume': 900000000},
            {'symbol': 'LTCUSDT', 'quote_volume': 2500000000}
        ]

    def get_5min_klines(self, symbol: str, limit: int = 50) -> Optional[pd.DataFrame]:
        """Получение 5-минутных данных"""
        try:
            url = f"{self.base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': '5m',
                'limit': limit
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return self.parse_klines_to_dataframe(data)
            else:
                return None

        except Exception as e:
            print(f"❌ Ошибка получения данных для {symbol}: {e}")
            return None

    def parse_klines_to_dataframe(self, data: List) -> pd.DataFrame:
        """Парсинг данных свечей в DataFrame"""
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades_count',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        # Конвертация типов данных
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Расчет дополнительных параметров
        df = self.calculate_candle_metrics(df)

        return df

    def calculate_candle_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет метрик свечей"""
        df = df.copy()

        # Основные параметры свечи
        df['body_size'] = abs(df['close'] - df['open'])
        df['total_range'] = df['high'] - df['low']
        df['is_bullish'] = df['close'] > df['open']
        df['is_doji'] = abs(df['close'] - df['open']) / df['total_range'] < 0.1

        # Процентные изменения
        df['price_change_percent'] = ((df['close'] - df['open']) / df['open']) * 100
        df['body_percent'] = (df['body_size'] / df['open']) * 100

        # Анализ объема
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()

        # Wick analysis
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['upper_wick_ratio'] = df['upper_wick'] / df['body_size']
        df['lower_wick_ratio'] = df['lower_wick'] / df['body_size']

        # Сила свечи
        df['candle_strength'] = df['body_percent'] * df['volume_ratio']

        return df

    def scan_high_volume_bullish_candles(self, df: pd.DataFrame, lookback_candles: int = 3) -> List[Dict]:
        """Поиск бычьих свечей с высоким объемом"""
        if df is None or len(df) < lookback_candles + 20:
            return []

        signals = []

        # Анализируем последние N свечей
        for i in range(len(df) - lookback_candles, len(df)):
            candle_data = df.iloc[i]

            # Проверяем условия для бычьей свечи с высоким объемом
            if self.is_high_volume_bullish_candle(candle_data):
                signal = {
                    'symbol': '',  # Будет заполнено позже
                    'timestamp': candle_data['timestamp'],
                    'open': candle_data['open'],
                    'high': candle_data['high'],
                    'low': candle_data['low'],
                    'close': candle_data['close'],
                    'volume': candle_data['volume'],
                    'volume_ratio': candle_data['volume_ratio'],
                    'volume_zscore': candle_data.get('volume_zscore', 0),
                    'price_change_percent': candle_data['price_change_percent'],
                    'candle_strength': candle_data['candle_strength'],
                    'candle_index': i,
                    'signal_strength': self.calculate_signal_strength(candle_data),
                    'pattern': self.identify_candle_pattern(df, i)
                }
                signals.append(signal)

        return signals

    def is_high_volume_bullish_candle(self, candle_data: pd.Series) -> bool:
        """Проверка условий для бычьей свечи с высоким объемом"""
        # Бычья свеча
        if not candle_data['is_bullish'] or candle_data['is_doji']:
            return False

        # Высокий объем (в X раз выше среднего)
        if candle_data['volume_ratio'] < self.volume_multiplier:
            return False

        # Значительное движение цены
        if abs(candle_data['price_change_percent']) < self.min_price_change:
            return False

        # Качество свечи (не слишком много wicks)
        if candle_data.get('upper_wick_ratio', 0) > 2.0:
            return False

        return True

    def calculate_signal_strength(self, candle_data: pd.Series) -> float:
        """Расчет силы сигнала"""
        strength_factors = []

        # Фактор объема
        volume_strength = min(candle_data['volume_ratio'] / 5.0, 1.0)
        strength_factors.append(volume_strength)

        # Фактор движения цены
        price_strength = min(abs(candle_data['price_change_percent']) / 5.0, 1.0)
        strength_factors.append(price_strength)

        # Фактор Z-score объема
        if 'volume_zscore' in candle_data and not pd.isna(candle_data['volume_zscore']):
            zscore_strength = min(abs(candle_data['volume_zscore']) / 3.0, 1.0)
            strength_factors.append(zscore_strength)

        return round(np.mean(strength_factors), 2)

    def identify_candle_pattern(self, df: pd.DataFrame, current_index: int) -> str:
        """Идентификация свечного паттерна"""
        if current_index < 2:
            return "SINGLE_BULLISH"

        current = df.iloc[current_index]
        prev1 = df.iloc[current_index - 1]
        prev2 = df.iloc[current_index - 2]

        # Bullish Engulfing
        if (not prev1['is_bullish'] and current['is_bullish'] and
                current['close'] > prev1['open'] and current['open'] < prev1['close']):
            return "BULLISH_ENGULFING"

        # Hammer-like pattern
        if (current['is_bullish'] and
                current.get('lower_wick_ratio', 0) > 2.0 and
                current.get('upper_wick_ratio', 0) < 0.5):
            return "HAMMER"

        # Three White Soldiers (частично)
        if (current_index >= 2 and
                all(df.iloc[i]['is_bullish'] for i in range(current_index - 2, current_index + 1)) and
                all(df.iloc[i]['close'] > df.iloc[i - 1]['close'] for i in
                    range(current_index - 1, current_index + 1))):
            return "THREE_WHITE_SOLDIERS"

        return "STRONG_BULLISH"

    def analyze_symbol(self, symbol_info: Dict) -> Optional[Dict]:
        """Анализ символа на наличие бычьих свечей с высоким объемом"""
        symbol = symbol_info['symbol']

        print(f"🔍 Анализ {symbol}...", end=" ")

        try:
            # Получаем 5-минутные данные
            df = self.get_5min_klines(symbol, limit=50)

            if df is None or len(df) < 25:
                print("❌ Недостаточно данных")
                return None

            # Ищем сигналы
            signals = self.scan_high_volume_bullish_candles(df.tail(3))  # Анализируем последние 3 свечи

            if signals:
                print(f"🎯 Найдено {len(signals)} сигналов!")
                return {
                    'symbol': symbol,
                    '24h_volume': symbol_info['quote_volume'],
                    'signals': signals,
                    'data': df,
                    'current_price': df['close'].iloc[-1],
                    'analysis_timestamp': datetime.now()
                }
            else:
                print("⏳ Нет сигналов")
                return None

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return None

    def print_signal_details(self, result: Dict):
        """Вывод деталей сигнала"""
        symbol = result['symbol']
        signals = result['signals']
        volume_24h = result['24h_volume']

        print(f"\n🎯 ВЫСОКООБЪЕМНЫЕ БЫЧЬИ СВЕЧИ НА {symbol}")
        print(f"   📊 24h Volume: {volume_24h:,.0f} USDT")
        print(f"   💰 Текущая цена: {result['current_price']:.6f}")
        print(f"   ⏰ Время анализа: {result['analysis_timestamp'].strftime('%H:%M:%S')}")
        print("-" * 60)

        for i, signal in enumerate(signals, 1):
            print(f"\n   Свеча #{i}:")
            print(f"   🕐 Время: {signal['timestamp'].strftime('%H:%M')}")
            print(f"   📈 Паттерн: {signal['pattern']}")
            print(f"   💪 Сила сигнала: {signal['signal_strength']}/1.0")
            print(f"   💰 Цена Open: {signal['open']:.6f}")
            print(f"   💰 Цена Close: {signal['close']:.6f}")
            print(f"   📊 Изменение: {signal['price_change_percent']:+.2f}%")
            print(f"   🎯 Объем: x{signal['volume_ratio']:.1f} от среднего")
            print(f"   📈 Z-Score объема: {signal.get('volume_zscore', 0):.2f}")
            print(f"   💥 Сила свечи: {signal['candle_strength']:.2f}")

    def plot_signals(self, result: Dict):
        """Визуализация сигналов"""
        symbol = result['symbol']
        df = result['data']
        signals = result['signals']

        plt.figure(figsize=(16, 12))

        # График цены
        plt.subplot(3, 1, 1)
        plt.plot(df['timestamp'], df['close'], label='Close Price', linewidth=1.5, color='blue', alpha=0.8)

        # Отмечаем сигнальные свечи
        for signal in signals:
            idx = signal['candle_index']
            plt.plot(df['timestamp'].iloc[idx], df['close'].iloc[idx],
                     '^', markersize=15, color='green', markeredgecolor='black',
                     markeredgewidth=2, label='High Volume Bullish')

        plt.title(f'{symbol} - Бычьи свечи с высоким объемом\n'
                  f'24h Volume: {result["24h_volume"]:,.0f} USDT')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # График объема
        plt.subplot(3, 1, 2)
        plt.bar(df['timestamp'], df['volume'], color='orange', alpha=0.6, label='Volume')
        plt.plot(df['timestamp'], df['volume_ma_20'], color='red', linewidth=2, label='Volume MA 20')

        # Отмечаем высокие объемы
        for signal in signals:
            idx = signal['candle_index']
            plt.bar(df['timestamp'].iloc[idx], df['volume'].iloc[idx],
                    color='green', alpha=0.8)

        plt.title('Объем торгов')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # График силы свечей
        plt.subplot(3, 1, 3)
        plt.plot(df['timestamp'], df['candle_strength'], label='Candle Strength', color='purple', linewidth=2)
        plt.axhline(y=df['candle_strength'].mean(), color='red', linestyle='--', label='Average Strength')

        for signal in signals:
            idx = signal['candle_index']
            plt.plot(df['timestamp'].iloc[idx], df['candle_strength'].iloc[idx],
                     'o', markersize=8, color='green', markeredgecolor='black')

        plt.title('Сила свечей (Изменение цены × Объем)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def monitor_high_volume_bullish(self, scan_interval: int = 300, max_symbols: int = 100):
        """Непрерывный мониторинг"""
        print("🎯 МОНИТОРИНГ БЫЧЬИХ СВЕЧЕЙ С ВЫСОКИМ ОБЪЕМОМ")
        print("=" * 70)
        print(f"🔧 Параметры:")
        print(f"   • Таймфрейм: 5 минут")
        print(f"   • Анализ последних: 3 свечи")
        print(f"   • Множитель объема: {self.volume_multiplier}x")
        print(f"   • Мин. изменение цены: {self.min_price_change}%")
        print(f"   • Интервал сканирования: {scan_interval} сек")
        print("=" * 70)

        scan_count = 0

        while True:
            scan_count += 1
            print(f"\n📊 Сканирование #{scan_count} - {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 70)

            # Получаем актуальные символы
            symbols = self.get_top_volume_symbols(limit=max_symbols)
            print(f"📈 Анализ {len(symbols)} высоколиквидных монет...")

            found_signals = []

            for symbol_info in symbols:
                result = self.analyze_symbol(symbol_info)
                if result:
                    found_signals.append(result)
                    self.print_signal_details(result)

            if found_signals:
                print(f"\n✅ Найдено сигналов: {len(found_signals)}")
                # Показываем графики
                for signal in found_signals:
                    self.plot_signals(signal)
            else:
                print(f"\n⏳ Сигналы не найдены. Следующее сканирование через {scan_interval} сек...")

            time.sleep(scan_interval)


def main():
    """Основная функция"""
    scanner = HighVolumeBullishScanner()

    try:
        print("🎯 СКАНЕР БЫЧЬИХ СВЕЧЕЙ С ВЫСОКИМ ОБЪЕМОМ")
        print("=" * 50)

        # Настройка параметров
        volume_multiplier = float(input("Множитель объема (по умолчанию 3.0): ") or "3.0")
        min_price_change = float(input("Мин. изменение цены % (по умолчанию 1.0): ") or "1.0")

        scanner = HighVolumeBullishScanner(
            volume_multiplier=volume_multiplier,
            min_price_change=min_price_change
        )

        while True:
            print("\nВыберите режим:")
            print("1 - Непрерывный мониторинг")
            print("2 - Разовое сканирование")
            print("3 - Изменить параметры")
            print("4 - Выход")

            choice = input("\nВведите номер: ").strip()

            if choice == "1":
                interval = int(input("Интервал сканирования (сек): ") or "300")
                max_symbols = int(input("Максимум монет: ") or "30")
                scanner.monitor_high_volume_bullish(interval, max_symbols)

            elif choice == "2":
                symbols = scanner.get_top_volume_symbols(limit=100)
                print(f"\n🔍 Разовое сканирование {len(symbols)} монет...")
                found_signals = []

                for symbol_info in symbols:
                    result = scanner.analyze_symbol(symbol_info)
                    if result:
                        found_signals.append(result)
                        scanner.print_signal_details(result)

                if not found_signals:
                    print("\n⏳ Бычьи свечи с высоким объемом не найдены")

            elif choice == "3":
                new_volume_mult = float(input("Новый множитель объема: ") or "3.0")
                new_price_change = float(input("Новое мин. изменение цены %: ") or "1.0")
                scanner.volume_multiplier = new_volume_mult
                scanner.min_price_change = new_price_change
                print(f"✅ Параметры обновлены: Volume x{new_volume_mult}, Min Change {new_price_change}%")

            elif choice == "4":
                print("Выход...")
                break

            else:
                print("Неверный выбор")

    except KeyboardInterrupt:
        print("\n\n⏹️  Остановлено пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")


if __name__ == "__main__":
    main()