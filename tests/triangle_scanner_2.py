import time
import numpy as np
import pandas as pd
import requests
import talib
from datetime import datetime, timedelta
import warnings
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class RealtimeTriangleEntry:
    def __init__(self):
        self.entry_confidence_threshold = 0.7
        self.min_volume = 1000000
        self.last_analysis = {}

    def get_active_symbols(self, min_volume=1000000, limit=30):
        """Получение активных символов в реальном времени"""
        url = "https://contract.mexc.com/api/v1/contract/ticker"
        try:
            response = requests.get(url, timeout=5)
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

            symbols.sort(key=lambda x: x['volume_24h'], reverse=True)
            return symbols[:limit]

        except Exception as e:
            print(f"Ошибка получения списка монет: {e}")
            return []

    def get_current_candles(self, symbol, interval="Min5", limit=50):
        """Получение текущих данных на 5-минутном таймфрейме"""
        url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
        params = {"interval": interval, "limit": limit}

        try:
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return self.create_dataframe(data)
        except Exception as e:
            print(f"   ❌ Ошибка получения данных {symbol}: {e}")
        return None

    def create_dataframe(self, data):
        """Создание DataFrame"""
        if not data or not data.get('success') or not data.get('data'):
            return None

        raw_data = data['data']

        try:
            if isinstance(raw_data, list):
                df = pd.DataFrame(raw_data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume'
                ])

                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                df = df.dropna()
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('datetime').reset_index(drop=True)

                return df

        except Exception as e:
            print(f"   ❌ Ошибка создания DataFrame: {e}")
        return None

    def calculate_realtime_indicators(self, df):
        """Расчет индикаторов для реального времени"""
        if len(df) < 10:
            return df

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        try:
            # Быстрые индикаторы для реального времени
            df['ema_9'] = talib.EMA(close, timeperiod=9)
            df['ema_21'] = talib.EMA(close, timeperiod=21)
            df['rsi_14'] = talib.RSI(close, timeperiod=14)
            df['atr'] = talib.ATR(high, low, close, timeperiod=14)
            df['macd'], df['macd_signal'], _ = talib.MACD(close)

            df = df.fillna(method='bfill').fillna(method='ffill')

        except Exception as e:
            print(f"   ❌ Ошибка расчета индикаторов: {e}")

        return df

    def find_realtime_triangle_breakout(self, df):
        """Поиск пробоя треугольника в реальном времени"""
        if df is None or len(df) < 20:
            return None

        try:
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values

            # Находим последние экстремумы
            recent_peaks, recent_troughs = self.find_recent_extremes(highs, lows)

            if len(recent_peaks) < 2 or len(recent_troughs) < 2:
                return None

            # Анализируем последние точки для треугольника
            triangle_info = self.analyze_recent_triangle(recent_peaks, recent_troughs, df)

            if triangle_info and triangle_info['confidence'] > self.entry_confidence_threshold:
                # Проверяем пробой в реальном времени
                breakout_info = self.check_realtime_breakout(df, triangle_info)
                if breakout_info['breakout_detected']:
                    return {
                        **triangle_info,
                        **breakout_info,
                        'entry_signal': True,
                        'timestamp': datetime.now()
                    }

            return None

        except Exception as e:
            print(f"   ❌ Ошибка поиска треугольника: {e}")
            return None

    def find_recent_extremes(self, highs, lows, lookback=20):
        """Поиск последних экстремумов"""
        peaks = []
        troughs = []

        # Анализируем только последние свечи
        start_idx = max(0, len(highs) - lookback)

        for i in range(start_idx + 3, len(highs) - 3):
            # Максимумы
            if (highs[i] >= highs[i - 1] and highs[i] >= highs[i - 2] and
                    highs[i] >= highs[i + 1] and highs[i] >= highs[i + 2]):
                peaks.append((i, highs[i]))

            # Минимумы
            if (lows[i] <= lows[i - 1] and lows[i] <= lows[i - 2] and
                    lows[i] <= lows[i + 1] and lows[i] <= lows[i + 2]):
                troughs.append((i, lows[i]))

        return peaks[-4:], troughs[-4:]  # Берем только последние 4 точки

    def analyze_recent_triangle(self, peaks, troughs, df):
        """Анализ треугольника на последних данных"""
        if len(peaks) < 2 or len(troughs) < 2:
            return None

        # Сортируем точки по времени
        all_points = sorted(peaks + troughs, key=lambda x: x[0])

        # Проверяем чередование
        if not self.check_points_alternation(all_points, peaks, troughs):
            return None

        # Анализируем линии
        upper_slope = self.calculate_line_slope(peaks)
        lower_slope = self.calculate_line_slope(troughs)

        triangle_type = self.classify_triangle_type(upper_slope, lower_slope)

        if triangle_type:
            confidence = self.calculate_pattern_confidence(peaks, troughs, df)

            return {
                'type': triangle_type,
                'upper_slope': upper_slope,
                'lower_slope': lower_slope,
                'peaks': peaks,
                'troughs': troughs,
                'confidence': confidence,
                'current_price': df['close'].iloc[-1],
                'data': df
            }

        return None

    def check_points_alternation(self, all_points, peaks, troughs):
        """Проверяет чередование точек"""
        if len(all_points) < 4:
            return False

        for i in range(len(all_points) - 1):
            current_in_peaks = all_points[i] in peaks
            next_in_peaks = all_points[i + 1] in peaks
            if current_in_peaks == next_in_peaks:
                return False
        return True

    def calculate_line_slope(self, points):
        """Вычисляет наклон линии"""
        if len(points) < 2:
            return 0

        points_sorted = sorted(points, key=lambda x: x[0])
        x1, y1 = points_sorted[0]
        x2, y2 = points_sorted[-1]

        if x2 == x1:
            return 0

        return (y2 - y1) / (x2 - x1)

    def classify_triangle_type(self, upper_slope, lower_slope):
        """Классифицирует тип треугольника"""
        threshold = 1e-5

        if upper_slope < -threshold and lower_slope > threshold:
            return "symmetrical"
        elif abs(upper_slope) < threshold and lower_slope > threshold:
            return "ascending"
        elif upper_slope < -threshold and abs(lower_slope) < threshold:
            return "descending"

        return None

    def calculate_pattern_confidence(self, peaks, troughs, df):
        """Вычисляет уверенность в паттерне"""
        confidence = 0.5  # Базовая уверенность

        # Проверяем объем
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].tail(10).mean()
        if current_volume > avg_volume * 1.2:
            confidence += 0.2

        # Проверяем RSI
        current_rsi = df['rsi_14'].iloc[-1]
        if 40 <= current_rsi <= 60:  # Нейтральная зона
            confidence += 0.1

        # Проверяем сходимость линий
        slope_diff = abs(self.calculate_line_slope(peaks) - self.calculate_line_slope(troughs))
        confidence += min(slope_diff * 1000, 0.3)

        return min(confidence, 1.0)

    def check_realtime_breakout(self, df, triangle_info):
        """Проверяет пробой в реальном времени"""
        current_data = df.iloc[-1]
        current_high = current_data['high']
        current_low = current_data['low']
        current_close = current_data['close']

        # Определяем уровни пробития
        resistance_level = self.calculate_resistance_level(triangle_info['peaks'])
        support_level = self.calculate_support_level(triangle_info['troughs'])

        breakout_direction = None
        entry_price = current_close
        stop_loss = None
        take_profit = None

        # Проверяем пробой сопротивления (лонг)
        if current_close > resistance_level and current_high > resistance_level:
            breakout_direction = "LONG"
            stop_loss = support_level
            take_profit = entry_price + (entry_price - stop_loss) * 1.5

        # Проверяем пробой поддержки (шорт)
        elif current_close < support_level and current_low < support_level:
            breakout_direction = "SHORT"
            stop_loss = resistance_level
            take_profit = entry_price - (stop_loss - entry_price) * 1.5

        if breakout_direction:
            return {
                'breakout_detected': True,
                'breakout_direction': breakout_direction,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'resistance_level': resistance_level,
                'support_level': support_level,
                'risk_reward_ratio': self.calculate_risk_reward(entry_price, stop_loss, take_profit)
            }

        return {'breakout_detected': False}

    def calculate_resistance_level(self, peaks):
        """Вычисляет уровень сопротивления"""
        if not peaks:
            return 0
        return max(price for _, price in peaks)

    def calculate_support_level(self, troughs):
        """Вычисляет уровень поддержки"""
        if not troughs:
            return 0
        return min(price for _, price in troughs)

    def calculate_risk_reward(self, entry, stop_loss, take_profit):
        """Вычисляет соотношение риск/вознаграждение"""
        if entry is None or stop_loss is None or take_profit is None:
            return 0

        if entry > stop_loss:  # LONG
            risk = entry - stop_loss
            reward = take_profit - entry
        else:  # SHORT
            risk = stop_loss - entry
            reward = entry - take_profit

        if risk > 0:
            return round(reward / risk, 2)
        return 0

    def analyze_symbol_for_entry(self, symbol):
        """Анализ символа для поиска точки входа"""
        print(f"🔍 Анализ {symbol}...", end=" ")

        try:
            # Получаем текущие данные
            df = self.get_current_candles(symbol, "Min5", 40)
            if df is None:
                print("❌ Нет данных")
                return None

            # Рассчитываем индикаторы
            df = self.calculate_realtime_indicators(df)

            # Ищем пробой треугольника
            entry_signal = self.find_realtime_triangle_breakout(df)

            if entry_signal:
                print("🎯 СИГНАЛ ВХОДА!")
                return {
                    'symbol': symbol,
                    **entry_signal,
                    'timestamp': datetime.now()
                }
            else:
                print("⏳ Ожидание")
                return None

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return None

    def monitor_entries_realtime(self, symbol_count=20, scan_interval=30):
        """Мониторинг точек входа в реальном времени"""
        print("🎯 РЕАЛЬНЫЙ МОНИТОРИНГ ТОЧЕК ВХОДА")
        print("=" * 70)
        print(f"🔧 Параметры:")
        print(f"   • Таймфрейм: 5 минут")
        print(f"   • Количество монет: {symbol_count}")
        print(f"   • Интервал сканирования: {scan_interval} сек")
        print(f"   • Минимальная уверенность: {self.entry_confidence_threshold}")
        print("=" * 70)

        scan_count = 0

        while True:
            scan_count += 1
            print(f"\n📊 Сканирование #{scan_count} - {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 50)

            # Получаем актуальные символы
            symbols_data = self.get_active_symbols(self.min_volume, symbol_count)
            symbols = [s['symbol'] for s in symbols_data]

            found_entries = []

            for symbol in symbols:
                entry_signal = self.analyze_symbol_for_entry(symbol)
                if entry_signal:
                    found_entries.append(entry_signal)

                    # Выводим детали сигнала
                    self.print_entry_signal(entry_signal)

                    # Показываем график для визуализации
                    self.plot_entry_signal(entry_signal)

            # Статистика сканирования
            if found_entries:
                print(f"\n✅ Найдено сигналов: {len(found_entries)}")
            else:
                print(f"\n⏳ Сигналы не найдены. Следующее сканирование через {scan_interval} сек...")

            # Ждем перед следующим сканированием
            time.sleep(scan_interval)

    def print_entry_signal(self, signal):
        """Выводит информацию о сигнале входа"""
        print(f"\n🎯 СИГНАЛ ВХОДА НА {signal['symbol']}")
        print(f"   📊 Тип: {signal['type'].upper()} треугольник")
        print(f"   📈 Направление: {signal['breakout_direction']}")
        print(f"   💰 Цена входа: {signal['entry_price']:.6f}")
        print(f"   🛡️  Стоп-лосс: {signal['stop_loss']:.6f}")
        print(f"   🎯 Тейк-профит: {signal['take_profit']:.6f}")
        print(f"   📊 Risk/Reward: {signal['risk_reward_ratio']}:1")
        print(f"   💪 Уверенность: {signal['confidence']:.2f}")
        print(f"   ⏰ Время: {signal['timestamp'].strftime('%H:%M:%S')}")

    def plot_entry_signal(self, signal):
        """Визуализирует сигнал входа"""
        df = signal['data']

        plt.figure(figsize=(12, 8))

        # График цены
        plt.subplot(2, 1, 1)
        plt.plot(df['close'].values, label='Close', linewidth=1, color='blue')
        plt.plot(df['high'].values, alpha=0.3, linewidth=0.5, color='green')
        plt.plot(df['low'].values, alpha=0.3, linewidth=0.5, color='red')

        # Разметка уровней
        plt.axhline(y=signal['resistance_level'], color='red', linestyle='--',
                    label=f'Resistance: {signal["resistance_level"]:.6f}')
        plt.axhline(y=signal['support_level'], color='green', linestyle='--',
                    label=f'Support: {signal["support_level"]:.6f}')

        # Точка входа
        plt.plot(len(df) - 1, signal['entry_price'], 'ro', markersize=8, label='Entry')

        plt.title(f"{signal['symbol']} - {signal['breakout_direction']} ENTRY SIGNAL")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Объемы
        plt.subplot(2, 1, 2)
        plt.bar(range(len(df)), df['volume'].values, alpha=0.7, color='orange')
        plt.title('Volume')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():
    """Основная функция"""
    analyzer = RealtimeTriangleEntry()

    print("🎯 СИСТЕМА ПОИСКА ТОЧЕК ВХОДА В РЕАЛЬНОМ ВРЕМЕНИ")
    print("=" * 60)

    print("\nВыберите режим:")
    print("1 - Непрерывный мониторинг (авто-трейдинг)")
    print("2 - Разовое сканирование")
    print("3 - Выход")

    choice = input("\nВведите номер: ").strip()

    if choice == "1":
        count = int(input("Количество монет для мониторинга (10-30): ") or "20")
        interval = int(input("Интервал сканирования в секундах (10-60): ") or "30")
        analyzer.monitor_entries_realtime(count, interval)

    elif choice == "2":
        symbols_data = analyzer.get_active_symbols(limit=100)
        symbols = [s['symbol'] for s in symbols_data]

        print(f"\n🔍 Разовое сканирование {len(symbols)} монет...")
        found_entries = []

        for symbol in symbols:
            entry = analyzer.analyze_symbol_for_entry(symbol)
            if entry:
                found_entries.append(entry)
                analyzer.print_entry_signal(entry)

        if not found_entries:
            print("\n⏳ Сигналы входа не найдены")

    elif choice == "3":
        print("Выход...")
        return

    else:
        print("Неверный выбор")


if __name__ == "__main__":
    main()