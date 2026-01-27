import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class FadingMomentumScanner:
    def __init__(self):
        self.min_sequence_length = 3
        self.volume_threshold = 0.8  # Минимальный объем относительно среднего
        self.trend_strength_threshold = 2.0  # Минимальная сила тренда в %

    def get_active_symbols(self, min_volume=1000000, limit=30):
        """Получение активных символов"""
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
                            'last_price': float(item['lastPrice'])
                        })

            symbols.sort(key=lambda x: x['volume_24h'], reverse=True)
            return symbols[:limit]

        except Exception as e:
            print(f"Ошибка получения списка монет: {e}")
            return []

    def get_current_candles(self, symbol, interval="Min30", limit=10):
        """Получение текущих данных"""
        url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
        params = {"interval": interval, "limit": limit}

        try:
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return self.create_dataframe(data)
        except Exception as e:
            return None
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

                # Добавляем расчет тела свечи
                df['body_size'] = abs(df['close'] - df['open'])
                df['body_percent'] = (df['body_size'] / df['open']) * 100
                df['is_bullish'] = df['close'] > df['open']
                df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
                df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

                return df

        except Exception as e:
            return None
        return None

    def find_fading_bullish_sequence(self, df):
        """Поиск затухающей бычьей последовательности"""
        if df is None or len(df) < 10:
            return None

        # Ищем бычьи свечи
        bullish_candles = df[df['is_bullish'] == True]
        if len(bullish_candles) < self.min_sequence_length:
            return None

        sequences = []
        current_sequence = []

        # Проходим по всем свечам в обратном порядке (от последней к первой)
        for i in range(len(df) - 1, -1, -1):
            candle = df.iloc[i]

            if candle['is_bullish']:
                if not current_sequence:
                    current_sequence.append((i, candle))
                else:
                    last_idx, last_candle = current_sequence[-1]
                    # Проверяем последовательность и уменьшение тела
                    if i == last_idx - 1 and candle['body_size'] < last_candle['body_size']:
                        current_sequence.append((i, candle))
                    else:
                        if len(current_sequence) >= self.min_sequence_length:
                            sequences.append(current_sequence)
                        current_sequence = [(i, candle)] if candle['is_bullish'] else []
            else:
                if len(current_sequence) >= self.min_sequence_length:
                    sequences.append(current_sequence)
                current_sequence = []

        # Добавляем последнюю последовательность
        if len(current_sequence) >= self.min_sequence_length:
            sequences.append(current_sequence)

        # Фильтруем и анализируем последовательности
        valid_sequences = []
        for seq in sequences:
            analysis = self.analyze_fading_sequence(seq, df)
            if analysis['is_valid']:
                valid_sequences.append(analysis)

        return valid_sequences if valid_sequences else None

    def analyze_fading_sequence(self, sequence, df):
        """Анализ затухающей последовательности"""
        # Сортируем от старых к новым
        sequence_sorted = sorted(sequence, key=lambda x: x[0])

        bodies = [candle['body_size'] for _, candle in sequence_sorted]
        body_percents = [candle['body_percent'] for _, candle in sequence_sorted]
        volumes = [candle['volume'] for _, candle in sequence_sorted]

        # Проверяем монотонное уменьшение тела
        is_monotonic_decreasing = all(bodies[i] >= bodies[i + 1] for i in range(len(bodies) - 1))

        # Проверяем общую силу тренда
        first_candle = sequence_sorted[0][1]
        last_candle = sequence_sorted[-1][1]
        total_move = ((last_candle['close'] - first_candle['open']) / first_candle['open']) * 100

        # Проверяем объемы
        avg_volume = df['volume'].tail(20).mean()
        volume_ok = all(vol > avg_volume * self.volume_threshold for vol in volumes)

        # Анализируем фитили
        wick_analysis = self.analyze_wicks(sequence_sorted)

        # Вычисляем силу затухания
        fading_strength = self.calculate_fading_strength(bodies)

        is_valid = (is_monotonic_decreasing and
                    total_move >= self.trend_strength_threshold and
                    volume_ok)

        return {
            'is_valid': is_valid,
            'sequence_length': len(sequence_sorted),
            'bodies': bodies,
            'body_percents': body_percents,
            'total_move_percent': total_move,
            'fading_strength': fading_strength,
            'volume_analysis': volume_ok,
            'wick_analysis': wick_analysis,
            'start_price': first_candle['open'],
            'end_price': last_candle['close'],
            'start_index': sequence_sorted[0][0],
            'end_index': sequence_sorted[-1][0],
            'timestamp': datetime.now(),
            'data': df
        }

    def analyze_wicks(self, sequence):
        """Анализ фитилей свечей"""
        upper_wicks = [candle['upper_wick'] for _, candle in sequence]
        lower_wicks = [candle['lower_wick'] for _, candle in sequence]
        bodies = [candle['body_size'] for _, candle in sequence]

        # Проверяем увеличение верхних фитилей (признак сопротивления)
        upper_wick_increasing = any(upper_wicks[i] < upper_wicks[i + 1] for i in range(len(upper_wicks) - 1))

        # Проверяем соотношение фитиль/тело
        wick_to_body_ratios = [upper_wicks[i] / bodies[i] if bodies[i] > 0 else 0 for i in range(len(bodies))]
        high_wick_ratio = any(ratio > 1.0 for ratio in wick_to_body_ratios)

        return {
            'upper_wick_increasing': upper_wick_increasing,
            'high_wick_ratio': high_wick_ratio,
            'avg_upper_wick': np.mean(upper_wicks),
            'avg_lower_wick': np.mean(lower_wicks)
        }

    def calculate_fading_strength(self, bodies):
        """Вычисляет силу затухания"""
        if len(bodies) < 2:
            return 0

        # Процент уменьшения от первой к последней свече
        fading_percent = ((bodies[0] - bodies[-1]) / bodies[0]) * 100

        # Плавность затухания (чем ближе к 1, тем плавнее)
        smoothness = 1.0
        for i in range(len(bodies) - 1):
            if bodies[i] > 0:
                decrease_ratio = bodies[i + 1] / bodies[i]
                smoothness *= decrease_ratio

        return min(fading_percent * smoothness, 100)

    def calculate_entry_signals(self, sequence_analysis):
        """Расчет сигналов для входа"""
        if not sequence_analysis['is_valid']:
            return None

        current_price = sequence_analysis['end_price']
        fading_strength = sequence_analysis['fading_strength']

        # Сила сигнала зависит от силы затухания
        signal_strength = min(fading_strength / 20, 1.0)  # Нормализуем до 0-1

        # Уровни для шорт позиции
        entry_price = current_price
        stop_loss = current_price * 1.01  # Стоп на 1% выше
        take_profit = current_price * 0.98  # Тейк на 2% ниже

        risk_reward = (entry_price - take_profit) / (stop_loss - entry_price)

        return {
            'signal_type': 'BEARISH_REVERSAL',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': round(risk_reward, 2),
            'signal_strength': round(signal_strength, 2),
            'confidence': sequence_analysis['fading_strength'] / 100,
            'sequence_length': sequence_analysis['sequence_length'],
            'total_move_before_fade': sequence_analysis['total_move_percent']
        }

    def scan_symbol_for_fading_momentum(self, symbol):
        """Сканирование символа на затухание импульса"""
        print(f"🔍 Анализ {symbol}...", end=" ")

        try:
            df = self.get_current_candles(symbol, "Min5", 30)
            if df is None:
                print("❌ Нет данных")
                return None

            sequences = self.find_fading_bullish_sequence(df)

            if sequences:
                best_sequence = max(sequences, key=lambda x: x['fading_strength'])
                entry_signals = self.calculate_entry_signals(best_sequence)

                if entry_signals and entry_signals['signal_strength'] > 0.3:
                    print("🎯 ЗАТУХАНИЕ НАЙДЕНО!")
                    return {
                        'symbol': symbol,
                        'sequence_analysis': best_sequence,
                        'entry_signals': entry_signals,
                        'timestamp': datetime.now()
                    }

            print("⏳ Нет сигналов")
            return None

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return None

    def plot_fading_sequence(self, analysis):
        """Визуализация затухающей последовательности"""
        symbol = analysis['symbol']
        seq_analysis = analysis['sequence_analysis']
        df = seq_analysis['data']
        start_idx = seq_analysis['start_index']
        end_idx = seq_analysis['end_index']

        plt.figure(figsize=(15, 10))

        # График цены
        plt.subplot(3, 1, 1)
        plt.plot(df['close'].values, label='Close Price', linewidth=1, color='blue')

        # Выделяем последовательность
        sequence_range = range(start_idx, end_idx + 1)
        sequence_prices = df['close'].iloc[sequence_range].values
        plt.plot(sequence_range, sequence_prices, 'r-', linewidth=2, label='Fading Sequence')

        # Отмечаем начало и конец последовательности
        plt.plot(start_idx, df['close'].iloc[start_idx], 'go', markersize=8, label='Sequence Start')
        plt.plot(end_idx, df['close'].iloc[end_idx], 'ro', markersize=8, label='Sequence End')

        plt.title(f"{symbol} - Fading Bullish Momentum\n"
                  f"Sequence Length: {seq_analysis['sequence_length']} candles | "
                  f"Fading Strength: {seq_analysis['fading_strength']:.1f}% | "
                  f"Total Move: {seq_analysis['total_move_percent']:.2f}%")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # График размеров тел свечей
        plt.subplot(3, 1, 2)
        bodies = seq_analysis['bodies']
        plt.plot(range(len(bodies)), bodies, 'ro-', linewidth=2, markersize=6)
        plt.title('Candle Body Sizes (Decreasing)')
        plt.ylabel('Body Size')
        plt.grid(True, alpha=0.3)

        # График объемов
        plt.subplot(3, 1, 3)
        volumes = [df['volume'].iloc[i] for i in range(start_idx, end_idx + 1)]
        plt.bar(range(len(volumes)), volumes, alpha=0.7, color='orange')
        plt.title('Volume During Sequence')
        plt.ylabel('Volume')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def monitor_fading_momentum(self, symbol_count=20, scan_interval=60):
        """Мониторинг затухания импульса в реальном времени"""
        print("🎯 МОНИТОРИНГ ЗАТУХАНИЯ БЫЧЬЕГО ИМПУЛЬСА")
        print("=" * 70)
        print(f"🔧 Параметры:")
        print(f"   • Минимальная последовательность: {self.min_sequence_length} свечи")
        print(f"   • Таймфрейм: 5 минут")
        print(f"   • Интервал сканирования: {scan_interval} сек")
        print("=" * 70)

        scan_count = 0

        while True:
            scan_count += 1
            print(f"\n📊 Сканирование #{scan_count} - {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 50)

            symbols_data = self.get_active_symbols(limit=symbol_count)
            symbols = [s['symbol'] for s in symbols_data]

            found_signals = []

            for symbol in symbols:
                signal = self.scan_symbol_for_fading_momentum(symbol)
                if signal:
                    found_signals.append(signal)

                    # Выводим детали сигнала
                    self.print_signal_details(signal)

                    # Показываем график
                    self.plot_fading_sequence(signal)

            if found_signals:
                print(f"\n✅ Найдено сигналов: {len(found_signals)}")
            else:
                print(f"\n⏳ Сигналы не найдены. Следующее сканирование через {scan_interval} сек...")

            time.sleep(scan_interval)

    def print_signal_details(self, signal):
        """Вывод деталей сигнала"""
        entry = signal['entry_signals']
        seq = signal['sequence_analysis']

        print(f"\n🎯 СИГНАЛ ЗАТУХАНИЯ НА {signal['symbol']}")
        print(f"   📊 Тип: {entry['signal_type']}")
        print(f"   📈 Длина последовательности: {seq['sequence_length']} свеч")
        print(f"   📉 Сила затухания: {seq['fading_strength']:.1f}%")
        print(f"   🚀 Предыдущее движение: {seq['total_move_percent']:.2f}%")
        print(f"   💰 Цена входа: {entry['entry_price']:.6f}")
        print(f"   🛡️  Стоп-лосс: {entry['stop_loss']:.6f}")
        print(f"   🎯 Тейк-профит: {entry['take_profit']:.6f}")
        print(f"   📊 Risk/Reward: {entry['risk_reward_ratio']}:1")
        print(f"   💪 Сила сигнала: {entry['signal_strength']:.2f}")
        print(f"   ⏰ Время: {signal['timestamp'].strftime('%H:%M:%S')}")


def main():
    """Основная функция"""
    scanner = FadingMomentumScanner()

    print("🎯 СКАНЕР ЗАТУХАЮЩЕЙ СИЛЫ БЫЧЬЕГО ТРЕНДА")
    print("=" * 60)

    print("\nВыберите режим:")
    print("1 - Непрерывный мониторинг")
    print("2 - Разовое сканирование")
    print("3 - Выход")

    choice = input("\nВведите номер: ").strip()

    if choice == "1":
        count = int(input("Количество монет (10-30): ") or "20")
        interval = int(input("Интервал сканирования в секундах (30-120): ") or "60")
        scanner.monitor_fading_momentum(count, interval)

    elif choice == "2":
        symbols_data = scanner.get_active_symbols(limit=80)
        symbols = [s['symbol'] for s in symbols_data]

        print(f"\n🔍 Разовое сканирование {len(symbols)} монет...")
        found_signals = []

        for symbol in symbols:
            signal = scanner.scan_symbol_for_fading_momentum(symbol)
            if signal:
                found_signals.append(signal)
                scanner.print_signal_details(signal)

        if not found_signals:
            print("\n⏳ Сигналы затухания не найдены")

    elif choice == "3":
        print("Выход...")
        return

    else:
        print("Неверный выбор")


if __name__ == "__main__":
    main()