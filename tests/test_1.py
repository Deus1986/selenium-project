import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import talib
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class StrongReversalScanner:
    def __init__(self):
        self.min_volume = 1000000
        self.reversal_confidence_threshold = 0.7
        self.trend_strength_min = 3.0  # Минимальная сила предыдущего тренда в %

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
                            'last_price': float(item['lastPrice']),
                            'change_percent': float(item['riseFallRate']) * 100
                        })

            symbols.sort(key=lambda x: x['volume_24h'], reverse=True)
            return symbols[:limit]

        except Exception as e:
            print(f"Ошибка получения списка монет: {e}")
            return []

    def get_hourly_candles(self, symbol, limit=100):
        """Получение часовых данных"""
        url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
        params = {"interval": "Min60", "limit": limit}

        try:
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return self.create_dataframe(data)
        except Exception as e:
            return None
        return None

    def create_dataframe(self, data):
        """Создание DataFrame с индикаторами"""
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

                # Расчет дополнительных параметров свечей
                df['body_size'] = abs(df['close'] - df['open'])
                df['body_percent'] = (df['body_size'] / df['open']) * 100
                df['is_bullish'] = df['close'] > df['open']
                df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
                df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
                df['total_range'] = df['high'] - df['low']
                df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / df['body_size']

                # Технические индикаторы
                close = df['close'].values
                high = df['high'].values
                low = df['low'].values

                df['rsi_14'] = talib.RSI(close, timeperiod=14)
                df['ema_21'] = talib.EMA(close, timeperiod=21)
                df['ema_50'] = talib.EMA(close, timeperiod=50)
                df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)
                df['atr'] = talib.ATR(high, low, close, timeperiod=14)
                df['adx'] = talib.ADX(high, low, close, timeperiod=14)

                # Уровни поддержки и сопротивления
                df['resistance'] = df['high'].rolling(10).max()
                df['support'] = df['low'].rolling(10).min()

                return df

        except Exception as e:
            return None
        return None

    def scan_strong_reversal_patterns(self, df):
        """Поиск сильных паттернов разворота для шорт входа"""
        if df is None or len(df) < 20:
            return None

        patterns_found = []

        # 1. Пин-бар (Pin Bar) на сопротивлении
        pin_bar_signal = self.find_pin_bar_reversal(df)
        if pin_bar_signal:
            patterns_found.append(pin_bar_signal)

        # 2. Медвежье поглощение (Bearish Engulfing)
        engulfing_signal = self.find_bearish_engulfing(df)
        if engulfing_signal:
            patterns_found.append(engulfing_signal)

        # 3. Вечерняя звезда (Evening Star)
        evening_star_signal = self.find_evening_star(df)
        if evening_star_signal:
            patterns_found.append(evening_star_signal)

        # 4. Тройная вершина (Triple Top)
        triple_top_signal = self.find_triple_top(df)
        if triple_top_signal:
            patterns_found.append(triple_top_signal)

        # 5. Дивергенция RSI
        divergence_signal = self.find_rsi_divergence(df)
        if divergence_signal:
            patterns_found.append(divergence_signal)

        # 6. Затухание импульса на сильном тренде
        momentum_fade_signal = self.find_momentum_fade(df)
        if momentum_fade_signal:
            patterns_found.append(momentum_fade_signal)

        if patterns_found:
            # Выбираем лучший паттерн по уверенности
            best_pattern = max(patterns_found, key=lambda x: x['confidence'])
            return best_pattern

        return None

    def find_pin_bar_reversal(self, df):
        """Поиск пин-бара на сопротивлении"""
        if len(df) < 3:
            return None

        current_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        # Проверяем условия пин-бара
        is_pin_bar = (
                current_candle['upper_wick'] > current_candle['body_size'] * 2 and  # Длинный верхний фитиль
                current_candle['lower_wick'] < current_candle['body_size'] * 0.5 and  # Короткий нижний фитиль
                current_candle['body_size'] > 0 and
                current_candle['high'] >= current_candle['resistance'] * 0.995  # Касание сопротивления
        )

        if is_pin_bar:
            # Проверяем контекст - предыдущий бычий тренд
            trend_strength = self.calculate_trend_strength(df, lookback=10)

            confidence = 0.6
            if current_candle['volume'] > df['volume'].tail(10).mean():
                confidence += 0.2
            if df['rsi_14'].iloc[-1] > 60:
                confidence += 0.1
            if trend_strength > self.trend_strength_min:
                confidence += 0.1

            return {
                'pattern': 'PIN_BAR',
                'confidence': min(confidence, 1.0),
                'entry_price': current_candle['close'],
                'resistance_level': current_candle['resistance'],
                'signal_candle_index': len(df) - 1,
                'volume_boost': current_candle['volume'] > df['volume'].tail(10).mean(),
                'rsi_level': df['rsi_14'].iloc[-1]
            }

        return None

    def find_bearish_engulfing(self, df):
        """Поиск медвежьего поглощения"""
        if len(df) < 3:
            return None

        current_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        # Условия медвежьего поглощения
        is_engulfing = (
                prev_candle['is_bullish'] and  # Предыдущая свеча бычья
                not current_candle['is_bullish'] and  # Текущая свеча медвежья
                current_candle['open'] > prev_candle['close'] and  # Открытие выше закрытия предыдущей
                current_candle['close'] < prev_candle['open'] and  # Закрытие ниже открытия предыдущей
                current_candle['body_size'] > prev_candle['body_size'] * 1.2  # Большее тело
        )

        if is_engulfing:
            confidence = 0.7
            if current_candle['volume'] > prev_candle['volume'] * 1.5:
                confidence += 0.2
            if df['rsi_14'].iloc[-1] > 65:
                confidence += 0.1

            return {
                'pattern': 'BEARISH_ENGULFING',
                'confidence': min(confidence, 1.0),
                'entry_price': current_candle['close'],
                'signal_candle_index': len(df) - 1,
                'volume_boost': current_candle['volume'] > prev_candle['volume'] * 1.5,
                'rsi_level': df['rsi_14'].iloc[-1]
            }

        return None

    def find_evening_star(self, df):
        """Поиск вечерней звезды"""
        if len(df) < 4:
            return None

        candle_1 = df.iloc[-3]  # Бычья свеча
        candle_2 = df.iloc[-2]  # Доджи или маленькая свеча
        candle_3 = df.iloc[-1]  # Медвежья свеча

        is_evening_star = (
                candle_1['is_bullish'] and
                candle_1['body_size'] > candle_1['total_range'] * 0.6 and  # Сильная бычья свеча
                candle_2['body_size'] < candle_2['total_range'] * 0.3 and  # Маленькое тело (доджи)
                not candle_3['is_bullish'] and  # Медвежья свеча
                candle_3['close'] < candle_1['body_size'] * 0.5 and  # Закрытие в середине первой свечи
                candle_2['high'] > candle_1['high']  # Вторая свеча выше первой
        )

        if is_evening_star:
            confidence = 0.75
            if candle_3['volume'] > candle_1['volume']:
                confidence += 0.15
            if df['rsi_14'].iloc[-1] > 70:
                confidence += 0.1

            return {
                'pattern': 'EVENING_STAR',
                'confidence': min(confidence, 1.0),
                'entry_price': candle_3['close'],
                'signal_candle_index': len(df) - 1,
                'volume_boost': candle_3['volume'] > candle_1['volume'],
                'rsi_level': df['rsi_14'].iloc[-1]
            }

        return None

    def find_triple_top(self, df):
        """Поиск тройной вершины"""
        if len(df) < 15:
            return None

        # Ищем три приблизительно равных максимума
        highs = df['high'].tail(15).values
        peaks = []

        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                peaks.append((i, highs[i]))

        if len(peaks) >= 3:
            recent_peaks = peaks[-3:]
            peak_prices = [price for _, price in recent_peaks]

            # Проверяем что цены вершин близки
            price_variance = np.std(peak_prices) / np.mean(peak_prices)

            if price_variance < 0.02:  # Вершины в пределах 2%
                current_price = df['close'].iloc[-1]
                resistance_level = np.mean(peak_prices)

                # Проверяем пробой поддержки
                if current_price < resistance_level * 0.98:
                    confidence = 0.8
                    if df['volume'].iloc[-1] > df['volume'].tail(10).mean() * 1.2:
                        confidence += 0.1

                    return {
                        'pattern': 'TRIPLE_TOP',
                        'confidence': min(confidence, 1.0),
                        'entry_price': current_price,
                        'resistance_level': resistance_level,
                        'signal_candle_index': len(df) - 1,
                        'volume_boost': df['volume'].iloc[-1] > df['volume'].tail(10).mean() * 1.2
                    }

        return None

    def find_rsi_divergence(self, df):
        """Поиск медвежьей дивергенции RSI"""
        if len(df) < 20:
            return None

        # Берем последние 15 периодов для анализа
        prices = df['close'].tail(15).values
        rsi = df['rsi_14'].tail(15).values

        # Ищем расхождения между ценой и RSI
        price_peaks = []
        rsi_peaks = []

        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                price_peaks.append((i, prices[i]))
            if rsi[i] > rsi[i - 1] and rsi[i] > rsi[i + 1]:
                rsi_peaks.append((i, rsi[i]))

        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            last_price_peak = price_peaks[-1][1]
            prev_price_peak = price_peaks[-2][1]
            last_rsi_peak = rsi_peaks[-1][1]
            prev_rsi_peak = rsi_peaks[-2][1]

            # Медвежья дивергенция: цена делает новый максимум, а RSI - нет
            is_divergence = (
                    last_price_peak > prev_price_peak and
                    last_rsi_peak < prev_rsi_peak and
                    last_rsi_peak > 60  # В зоне перекупленности
            )

            if is_divergence:
                confidence = 0.7
                if df['macd_hist'].iloc[-1] < 0:  # MACD гистограмма отрицательная
                    confidence += 0.2

                return {
                    'pattern': 'RSI_DIVERGENCE',
                    'confidence': min(confidence, 1.0),
                    'entry_price': df['close'].iloc[-1],
                    'signal_candle_index': len(df) - 1,
                    'rsi_level': rsi[-1],
                    'macd_confirmation': df['macd_hist'].iloc[-1] < 0
                }

        return None

    def find_momentum_fade(self, df):
        """Поиск затухания импульса на сильном тренде"""
        if len(df) < 10:
            return None

        # Анализируем последние 5 свечей
        recent_candles = df.tail(5)
        bullish_candles = recent_candles[recent_candles['is_bullish'] == True]

        if len(bullish_candles) >= 3:
            bodies = bullish_candles['body_size'].values
            volumes = bullish_candles['volume'].values

            # Проверяем уменьшение размера тела и объема
            body_decreasing = all(bodies[i] >= bodies[i + 1] for i in range(len(bodies) - 1))
            volume_decreasing = all(volumes[i] >= volumes[i + 1] for i in range(len(volumes) - 1))

            if body_decreasing and volume_decreasing:
                trend_strength = self.calculate_trend_strength(df, lookback=10)

                if trend_strength > self.trend_strength_min:
                    confidence = 0.65
                    if df['rsi_14'].iloc[-1] > 70:
                        confidence += 0.2
                    if df['adx'].iloc[-1] > 25:  # Сильный тренд
                        confidence += 0.15

                    return {
                        'pattern': 'MOMENTUM_FADE',
                        'confidence': min(confidence, 1.0),
                        'entry_price': df['close'].iloc[-1],
                        'signal_candle_index': len(df) - 1,
                        'trend_strength': trend_strength,
                        'rsi_level': df['rsi_14'].iloc[-1]
                    }

        return None

    def calculate_trend_strength(self, df, lookback=10):
        """Вычисление силы тренда"""
        if len(df) < lookback:
            return 0

        start_price = df['close'].iloc[-lookback]
        end_price = df['close'].iloc[-1]
        trend_strength = ((end_price - start_price) / start_price) * 100

        return abs(trend_strength)

    def calculate_short_entry(self, pattern, df):
        """Расчет параметров шорт входа"""
        current_price = pattern['entry_price']
        atr = df['atr'].iloc[-1]

        # Стоп-лосс выше сопротивления/максимума
        if 'resistance_level' in pattern:
            stop_loss = pattern['resistance_level'] * 1.005
        else:
            stop_loss = current_price * 1.02

        # Тейк-профит на основе ATR
        take_profit = current_price - (atr * 2)

        risk = stop_loss - current_price
        reward = current_price - take_profit
        risk_reward = reward / risk if risk > 0 else 0

        return {
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': round(risk_reward, 2),
            'atr_value': atr,
            'position_size_suggestion': self.calculate_position_size(risk, current_price)
        }

    def calculate_position_size(self, risk_per_trade, current_price):
        """Расчет размера позиции"""
        # Предполагаем риск 1% от депозита
        account_balance = 1000  # Можно настроить
        risk_amount = account_balance * 0.01
        position_size = risk_amount / (risk_per_trade / current_price)

        return round(position_size, 4)

    def scan_symbol_for_reversal(self, symbol):
        """Сканирование символа на разворотные паттерны"""
        print(f"🔍 Анализ {symbol}...", end=" ")

        try:
            df = self.get_hourly_candles(symbol, 100)
            if df is None:
                print("❌ Нет данных")
                return None

            pattern = self.scan_strong_reversal_patterns(df)

            if pattern and pattern['confidence'] >= self.reversal_confidence_threshold:
                entry_params = self.calculate_short_entry(pattern, df)

                print("🎯 СИГНАЛ ШОРТ!")
                return {
                    'symbol': symbol,
                    'pattern': pattern,
                    'entry': entry_params,
                    'timestamp': datetime.now(),
                    'data': df
                }
            else:
                print("⏳ Нет сигналов")
                return None

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return None

    def plot_reversal_signal(self, signal):
        """Визуализация разворотного сигнала"""
        symbol = signal['symbol']
        pattern = signal['pattern']
        entry = signal['entry']
        df = signal['data']

        plt.figure(figsize=(15, 10))

        # График цены
        plt.subplot(2, 1, 1)
        plt.plot(df['close'].values, label='Close Price', linewidth=1, color='blue', alpha=0.7)
        plt.plot(df['high'].values, alpha=0.3, color='green', linewidth=0.5)
        plt.plot(df['low'].values, alpha=0.3, color='red', linewidth=0.5)

        # Отмечаем сигнальную свечу
        signal_idx = pattern['signal_candle_index']
        plt.plot(signal_idx, df['close'].iloc[signal_idx], 'ro', markersize=10, label='Signal Candle')

        # Уровни входа и стопа
        plt.axhline(y=entry['entry_price'], color='orange', linestyle='-', label=f'Entry: {entry["entry_price"]:.6f}')
        plt.axhline(y=entry['stop_loss'], color='red', linestyle='--', label=f'Stop: {entry["stop_loss"]:.6f}')
        plt.axhline(y=entry['take_profit'], color='green', linestyle='--', label=f'TP: {entry["take_profit"]:.6f}')

        if 'resistance_level' in pattern:
            plt.axhline(y=pattern['resistance_level'], color='purple', linestyle=':',
                        label=f'Resistance: {pattern["resistance_level"]:.6f}')

        plt.title(f"{symbol} - {pattern['pattern']} SHORT SIGNAL\n"
                  f"Confidence: {pattern['confidence']:.2f} | R/R: {entry['risk_reward_ratio']}:1")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Индикаторы
        plt.subplot(2, 1, 2)
        plt.plot(df['rsi_14'].values, label='RSI', color='purple')
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        plt.title('RSI Indicator')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def monitor_reversal_signals(self, symbol_count=20, scan_interval=300):
        """Мониторинг разворотных сигналов"""
        print("🎯 МОНИТОРИНГ СИЛЬНЫХ РАЗВОРОТНЫХ ПАТТЕРНОВ НА ШОРТ")
        print("=" * 70)
        print(f"🔧 Параметры:")
        print(f"   • Таймфрейм: 1 час")
        print(f"   • Минимальная уверенность: {self.reversal_confidence_threshold}")
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
                signal = self.scan_symbol_for_reversal(symbol)
                if signal:
                    found_signals.append(signal)

                    # Выводим детали сигнала
                    self.print_signal_details(signal)

                    # Показываем график
                    self.plot_reversal_signal(signal)

            if found_signals:
                print(f"\n✅ Найдено сигналов: {len(found_signals)}")
            else:
                print(f"\n⏳ Сигналы не найдены. Следующее сканирование через {scan_interval} сек...")

            time.sleep(scan_interval)

    def print_signal_details(self, signal):
        """Вывод деталей сигнала"""
        pattern = signal['pattern']
        entry = signal['entry']

        print(f"\n🎯 СИГНАЛ ШОРТ НА {signal['symbol']}")
        print(f"   📊 Паттерн: {pattern['pattern']}")
        print(f"   💪 Уверенность: {pattern['confidence']:.2f}")
        print(f"   💰 Цена входа: {entry['entry_price']:.6f}")
        print(f"   🛡️  Стоп-лосс: {entry['stop_loss']:.6f}")
        print(f"   🎯 Тейк-профит: {entry['take_profit']:.6f}")
        print(f"   📊 Risk/Reward: {entry['risk_reward_ratio']}:1")
        print(f"   📈 ATR: {entry['atr_value']:.6f}")
        print(f"   💵 Размер позиции: {entry['position_size_suggestion']}")
        if 'rsi_level' in pattern:
            print(f"   📉 RSI: {pattern['rsi_level']:.1f}")
        print(f"   ⏰ Время: {signal['timestamp'].strftime('%H:%M:%S')}")


def main():
    """Основная функция"""
    scanner = StrongReversalScanner()

    print("🎯 СКАНЕР СИЛЬНЫХ РАЗВОРОТНЫХ ПАТТЕРНОВ ДЛЯ ШОРТ")
    print("=" * 60)

    print("\nВыберите режим:")
    print("1 - Непрерывный мониторинг")
    print("2 - Разовое сканирование")
    print("3 - Выход")

    choice = input("\nВведите номер: ").strip()

    if choice == "1":
        count = int(input("Количество монет (10-30): ") or "20")
        interval = int(input("Интервал сканирования в секундах (300-1800): ") or "300")
        scanner.monitor_reversal_signals(count, interval)

    elif choice == "2":
        symbols_data = scanner.get_active_symbols(limit=80)
        symbols = [s['symbol'] for s in symbols_data]

        print(f"\n🔍 Разовое сканирование {len(symbols)} монет...")
        found_signals = []

        for symbol in symbols:
            signal = scanner.scan_symbol_for_reversal(symbol)
            if signal:
                found_signals.append(signal)
                scanner.print_signal_details(signal)

        if not found_signals:
            print("\n⏳ Разворотные сигналы не найдены")

    elif choice == "3":
        print("Выход...")
        return

    else:
        print("Неверный выбор")


if __name__ == "__main__":
    main()