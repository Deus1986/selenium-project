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


class LongShadowCandleTrader:
    def __init__(self):
        self.min_confidence = 75
        self.min_profit_ratio = 2.0

    def get_candles(self, symbol, interval="Min60", limit=100):
        """Получение часовых данных"""
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
            print(f"   ❌ Нет данных или неверный формат ответа")
            return None

        raw_data = data['data']

        try:
            if isinstance(raw_data, dict):
                required_fields = ['time', 'open', 'close', 'high', 'low', 'vol']
                missing_fields = [field for field in required_fields if field not in raw_data]
                if missing_fields:
                    print(f"   ❌ Отсутствуют поля: {missing_fields}")
                    return None

                df = pd.DataFrame({
                    'timestamp': raw_data['time'],
                    'open': raw_data['open'],
                    'high': raw_data['high'],
                    'low': raw_data['low'],
                    'close': raw_data['close'],
                    'volume': raw_data['vol']
                })

            elif isinstance(raw_data, list):
                if len(raw_data) == 0:
                    return None

                first_item = raw_data[0]
                if isinstance(first_item, list) and len(first_item) >= 6:
                    df = pd.DataFrame(raw_data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume'
                    ])
                else:
                    print(f"   ❌ Неизвестный формат списка")
                    return None
            else:
                print(f"   ❌ Неподдерживаемый формат данных: {type(raw_data)}")
                return None

            # Конвертируем в числовые типы
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna()

            if len(df) < 50:
                print(f"   ⚠️ Недостаточно данных после очистки: {len(df)} строк")
                return None

            # Исправленная конвертация timestamp
            try:
                # Проверяем размер timestamp (могут быть в секундах или миллисекундах)
                sample_timestamp = df['timestamp'].iloc[0]
                if sample_timestamp > 1e12:  # Если больше 1e12, это вероятно миллисекунды
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                else:  # Иначе считаем секундами
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            except Exception as e:
                print(f"   ⚠️ Ошибка конвертации времени: {e}")
                # Создаем временные метки на основе индекса
                df['datetime'] = pd.date_range(end=datetime.now(), periods=len(df), freq='1H')

            df = df.sort_values('datetime').reset_index(drop=True)
            print(f"   ✅ Успешно создан DataFrame с {len(df)} строками")
            print(f"   📅 Диапазон времени: {df['datetime'].min()} - {df['datetime'].max()}")
            return df

        except Exception as e:
            print(f"   ❌ Ошибка создания DataFrame: {e}")
            return None

    def calculate_hourly_indicators(self, df):
        """Расчет индикаторов для часового таймфрейма"""
        if len(df) < 20:
            return df

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        try:
            # Трендовые индикаторы
            df['ema_9'] = talib.EMA(close, timeperiod=9)
            df['ema_21'] = talib.EMA(close, timeperiod=21)
            df['sma_50'] = talib.SMA(close, timeperiod=50)

            # Моментум
            df['rsi_14'] = talib.RSI(close, timeperiod=14)
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)

            # Stochastic
            stoch_result = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            df['stoch_k'] = stoch_result[0]
            df['stoch_d'] = stoch_result[1]

            # Волатильность
            df['atr'] = talib.ATR(high, low, close, timeperiod=14)
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close, timeperiod=20)

            # Объем
            df['volume_sma'] = talib.SMA(volume, timeperiod=20)
            df['volume_ratio'] = volume / df['volume_sma']

            # Анализ свечей - ОСНОВНЫЕ МЕТРИКИ ДЛЯ ДЛИННЫХ ТЕНЕЙ
            df['candle_body'] = abs(df['close'] - df['open'])
            df['candle_size'] = df['high'] - df['low']
            df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']

            # Отношения теней к телу свечи
            df['upper_shadow_ratio'] = np.where(df['candle_body'] > 0, df['upper_shadow'] / df['candle_body'], 0)
            df['lower_shadow_ratio'] = np.where(df['candle_body'] > 0, df['lower_shadow'] / df['candle_body'], 0)

            # Отношения теней к общему размеру свечи
            df['upper_shadow_percent'] = np.where(df['candle_size'] > 0, (df['upper_shadow'] / df['candle_size']) * 100,
                                                  0)
            df['lower_shadow_percent'] = np.where(df['candle_size'] > 0, (df['lower_shadow'] / df['candle_size']) * 100,
                                                  0)

            # Общая характеристика тени
            df['max_shadow_ratio'] = np.maximum(df['upper_shadow_ratio'], df['lower_shadow_ratio'])
            df['max_shadow_percent'] = np.maximum(df['upper_shadow_percent'], df['lower_shadow_percent'])

            df['body_ratio'] = np.where(df['candle_size'] > 0, df['candle_body'] / df['candle_size'], 0)
            df['body_percent'] = (df['candle_body'] / df['close']) * 100
            df['volume_zscore'] = (volume - df['volume_sma']) / df['volume_sma'].std()

            # Дополнительные индикаторы для контекста
            df['price_vs_ema9'] = (close / df['ema_9'] - 1) * 100
            df['price_vs_ema21'] = (close / df['ema_21'] - 1) * 100

            df = df.fillna(method='bfill').fillna(method='ffill')
            print(f"   ✅ Рассчитаны индикаторы для анализа теней")

        except Exception as e:
            print(f"   ❌ Ошибка расчета индикаторов: {e}")
            # Заглушки для индикаторов
            indicator_defaults = {
                'stoch_k': 50, 'stoch_d': 50, 'rsi_14': 50,
                'macd': 0, 'macd_signal': 0, 'atr': df['close'] * 0.02,
                'body_percent': 1.0, 'volume_zscore': 0, 'volume_ratio': 1.0,
                'upper_shadow_ratio': 0, 'lower_shadow_ratio': 0,
                'upper_shadow_percent': 0, 'lower_shadow_percent': 0,
                'max_shadow_ratio': 0, 'max_shadow_percent': 0
            }
            for col, default_val in indicator_defaults.items():
                df[col] = default_val

        return df

    def find_long_shadow_candles(self, df, min_shadow_ratio=2.0, min_shadow_percent=40, lookback_periods=5):
        """
        Поиск свечей с очень длинными тенями

        Args:
            df: DataFrame с данными свечей
            min_shadow_ratio: Минимальное отношение тени к телу (например, 2.0 = тень в 2 раза больше тела)
            min_shadow_percent: Минимальный процент тени от общего размера свечи
            lookback_periods: Количество последних свечей для анализа
        """
        if len(df) < lookback_periods:
            print(f"   ⚠️ Недостаточно данных для анализа")
            return []

        long_shadow_candles = []

        # Анализируем последние свечи
        for i in range(-lookback_periods, 0):
            idx = len(df) + i
            if idx < 0:
                continue

            row = df.iloc[idx]

            # Проверяем условия для длинной верхней тени
            has_long_upper_shadow = (
                    row['upper_shadow_ratio'] >= min_shadow_ratio and
                    row['upper_shadow_percent'] >= min_shadow_percent
            )

            # Проверяем условия для длинной нижней тени
            has_long_lower_shadow = (
                    row['lower_shadow_ratio'] >= min_shadow_ratio and
                    row['lower_shadow_percent'] >= min_shadow_percent
            )

            if has_long_upper_shadow or has_long_lower_shadow:
                # Определяем тип свечи
                if has_long_upper_shadow and has_long_lower_shadow:
                    candle_type = "LONG_UPPER_LOWER_SHADOW"
                elif has_long_upper_shadow:
                    candle_type = "LONG_UPPER_SHADOW"
                else:
                    candle_type = "LONG_LOWER_SHADOW"

                # Определяем направление (бычье/медвежье)
                is_bullish = row['close'] > row['open']
                direction = "BULLISH" if is_bullish else "BEARISH"

                # Определяем силу сигнала
                signal_strength = self.calculate_shadow_signal_strength(row, candle_type)

                candle_info = {
                    'timestamp': row['timestamp'],
                    'datetime': row['datetime'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'candle_type': candle_type,
                    'direction': direction,
                    'signal_strength': signal_strength,
                    'upper_shadow_ratio': round(row['upper_shadow_ratio'], 2),
                    'lower_shadow_ratio': round(row['lower_shadow_ratio'], 2),
                    'upper_shadow_percent': round(row['upper_shadow_percent'], 1),
                    'lower_shadow_percent': round(row['lower_shadow_percent'], 1),
                    'body_percent': round(row['body_percent'], 2),
                    'volume_ratio': round(row['volume_ratio'], 2),
                    'rsi': round(row['rsi_14'], 1) if 'rsi_14' in row else 50,
                    'price_vs_ema9': round(row['price_vs_ema9'], 2),
                    'is_current': (idx == len(df) - 1)  # Является ли текущей свечой
                }

                long_shadow_candles.append(candle_info)

                shadow_type = "ВЕРХНЯЯ" if has_long_upper_shadow else "НИЖНЯЯ"
                if has_long_upper_shadow and has_long_lower_shadow:
                    shadow_type = "ОБЕ ТЕНИ"

                print(f"   ✅ Найдена свеча с длинной {shadow_type} тенью "
                      f"(отношение: {candle_info['upper_shadow_ratio'] if has_long_upper_shadow else candle_info['lower_shadow_ratio']:.1f}x, "
                      f"доля: {candle_info['upper_shadow_percent'] if has_long_upper_shadow else candle_info['lower_shadow_percent']:.1f}%)")

        return long_shadow_candles

    def calculate_shadow_signal_strength(self, row, candle_type):
        """Расчет силы сигнала на основе дополнительных факторов"""
        signal_score = 0

        # Базовый счет за соотношение теней
        if candle_type == "LONG_UPPER_SHADOW":
            signal_score += min(row['upper_shadow_ratio'] * 10, 30)
        elif candle_type == "LONG_LOWER_SHADOW":
            signal_score += min(row['lower_shadow_ratio'] * 10, 30)
        else:  # LONG_UPPER_LOWER_SHADOW
            signal_score += min(row['max_shadow_ratio'] * 15, 40)

        # Объем выше среднего усиливает сигнал
        if row['volume_ratio'] > 1.5:
            signal_score += 10
        elif row['volume_ratio'] > 2.0:
            signal_score += 20

        # Экстремальные значения RSI усиливают сигнал
        if row['rsi_14'] < 30 or row['rsi_14'] > 70:
            signal_score += 10

        # Позиция относительно EMA
        if abs(row['price_vs_ema9']) > 2 or abs(row['price_vs_ema21']) > 3:
            signal_score += 10

        return min(signal_score, 100)

    def analyze_long_shadow_pattern(self, df, candle_info):
        """Анализ паттерна длинной тени и прогноз направления"""
        try:
            current_idx = df[df['timestamp'] == candle_info['timestamp']].index[0]

            # Анализируем контекст
            if current_idx < 2:
                return "NEUTRAL"

            prev_candle = df.iloc[current_idx - 1]
            current_candle = df.iloc[current_idx]

            bullish_signals = 0
            bearish_signals = 0

            # Анализ для длинной нижней тени (возможный разворот вверх)
            if candle_info['candle_type'] in ["LONG_LOWER_SHADOW", "LONG_UPPER_LOWER_SHADOW"]:
                # Проверяем, был ли перед этим нисходящий тренд
                if current_idx >= 2:
                    prev_trend = self.analyze_short_trend(df, current_idx)
                    if prev_trend == "DOWNTREND":
                        bullish_signals += 2

                # Длинная нижняя тень на поддержке
                if current_candle['close'] > current_candle['ema_21']:
                    bullish_signals += 1

                # Подтверждение объемом
                if current_candle['volume_ratio'] > 1.5:
                    bullish_signals += 1

            # Анализ для длинной верхней тени (возможный разворот вниз)
            if candle_info['candle_type'] in ["LONG_UPPER_SHADOW", "LONG_UPPER_LOWER_SHADOW"]:
                # Проверяем, был ли перед этим восходящий тренд
                if current_idx >= 2:
                    prev_trend = self.analyze_short_trend(df, current_idx)
                    if prev_trend == "UPTREND":
                        bearish_signals += 2

                # Длинная верхняя тень на сопротивлении
                if current_candle['close'] < current_candle['ema_21']:
                    bearish_signals += 1

                # Подтверждение объемом
                if current_candle['volume_ratio'] > 1.5:
                    bearish_signals += 1

            # Определяем общее направление
            if bullish_signals > bearish_signals:
                return "BULLISH_REVERSAL"
            elif bearish_signals > bullish_signals:
                return "BEARISH_REVERSAL"
            else:
                return "NEUTRAL"

        except Exception as e:
            print(f"   ⚠️ Ошибка анализа паттерна: {e}")
            return "NEUTRAL"

    def analyze_short_trend(self, df, current_idx, period=3):
        """Анализ краткосрочного тренда"""
        if current_idx < period:
            return "SIDEWAYS"

        prices = df['close'].iloc[current_idx - period:current_idx]
        if len(prices) < period:
            return "SIDEWAYS"

        # Простой анализ тренда по скользящей средней
        price_change = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100

        if price_change > 1.0:
            return "UPTREND"
        elif price_change < -1.0:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"

    def scan_long_shadow_candles(self, symbols, min_shadow_ratio=2.0, min_shadow_percent=40, lookback_periods=5):
        """
        Сканирование символов на наличие свечей с длинными тенями
        """
        print("🎯 СКАНИРОВАНИЕ СВЕЧЕЙ С ДЛИННЫМИ ТЕНЯМИ")
        print("=" * 80)
        print(f"Критерии поиска:")
        print(f"   • Отношение тени к телу ≥ {min_shadow_ratio}x")
        print(f"   • Доля тени от свечи ≥ {min_shadow_percent}%")
        print(f"   • Анализ последних {lookback_periods} свечей")
        print("=" * 80)

        results = []

        for symbol in symbols:
            print(f"\n🔍 Анализ {symbol}...")

            try:
                # Получаем данные
                data = self.get_candles(symbol, "Min60", 100)
                if not data:
                    continue

                df = self.create_dataframe(data)
                if df is None:
                    continue

                # Расчет индикаторов
                df = self.calculate_hourly_indicators(df)

                # Поиск свечей с длинными тенями
                long_shadow_candles = self.find_long_shadow_candles(
                    df,
                    min_shadow_ratio=min_shadow_ratio,
                    min_shadow_percent=min_shadow_percent,
                    lookback_periods=lookback_periods
                )

                for candle in long_shadow_candles:
                    # Анализируем паттерн
                    pattern_analysis = self.analyze_long_shadow_pattern(df, candle)

                    result = {
                        'symbol': symbol,
                        'candle': candle,
                        'pattern_analysis': pattern_analysis,
                        'current_price': df['close'].iloc[-1],
                        'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }

                    results.append(result)

            except Exception as e:
                print(f"   ❌ Ошибка анализа {symbol}: {e}")

            time.sleep(0.3)  # Пауза между запросами

        # Сортируем по силе сигнала
        results.sort(key=lambda x: x['candle']['signal_strength'], reverse=True)

        print(f"\n{'=' * 80}")
        print(f"📊 РЕЗУЛЬТАТЫ СКАНИРОВАНИЯ СВЕЧЕЙ С ДЛИННЫМИ ТЕНЯМИ:")
        print(f"   • Найдено сигналов: {len(results)}")
        print(f"   • Всего проанализировано: {len(symbols)}")
        print(f"   • Время анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return results


def main_long_shadow_scan():
    """Основная функция для поиска свечей с длинными тенями"""
    trader = LongShadowCandleTrader()

    # Получаем список монет с высоким объемом
    symbols_data = get_high_volume_symbols(min_volume=20000000)
    if not symbols_data:
        print("❌ Не удалось получить список монет")
        return []

    # Берем топ-20 самых ликвидных монет
    symbols = [item['symbol'] for item in symbols_data]

    print(f"📊 Анализируем {len(symbols)} монет на наличие свечей с длинными тенями...")

    # Запускаем сканирование
    results = trader.scan_long_shadow_candles(
        symbols,
        min_shadow_ratio=20.0,  # Тень минимум в 2 раза больше тела
        min_shadow_percent=40,  # Тень составляет минимум 40% от свечи
        lookback_periods=2  # Анализ последних 5 свечей
    )

    # Вывод результатов
    if results:
        print(f"\n🎯 СИГНАЛЫ С ДЛИННЫМИ ТЕНЯМИ:")
        print(f"{'=' * 80}")

        for i, result in enumerate(results, 1):
            candle = result['candle']
            pattern = result['pattern_analysis']

            # Определяем иконки и цвета
            if "BULLISH" in pattern:
                direction_icon = "🟢"
                direction_color = "БЫЧИЙ"
            elif "BEARISH" in pattern:
                direction_icon = "🔴"
                direction_color = "МЕДВЕЖИЙ"
            else:
                direction_icon = "⚪"
                direction_color = "НЕЙТРАЛЬНЫЙ"

            # Иконка для типа тени
            if "UPPER" in candle['candle_type'] and "LOWER" in candle['candle_type']:
                shadow_icon = "⬆️⬇️"
            elif "UPPER" in candle['candle_type']:
                shadow_icon = "⬆️"
            else:
                shadow_icon = "⬇️"

            print(f"\n{i}. {direction_icon} {shadow_icon} {result['symbol']}")
            print(f"   📅 Время свечи: {candle['datetime']}")
            print(f"   💰 Цена: {candle['close']} ({candle['direction']})")
            print(f"   🎯 Тип свечи: {candle['candle_type']}")
            print(f"   📏 Верхняя тень: {candle['upper_shadow_ratio']}x ({candle['upper_shadow_percent']}%)")
            print(f"   📏 Нижняя тень: {candle['lower_shadow_ratio']}x ({candle['lower_shadow_percent']}%)")
            print(f"   📊 Объем: {candle['volume_ratio']}x")
            print(f"   📈 RSI: {candle['rsi']}")
            print(f"   💪 Сила сигнала: {candle['signal_strength']}/100")
            print(f"   🔍 Анализ: {pattern} ({direction_color})")

            # Торговые рекомендации
            if pattern in ["BULLISH_REVERSAL", "BEARISH_REVERSAL"]:
                action = "BUY" if "BULLISH" in pattern else "SELL"
                print(f"   💡 СИГНАЛ: ВОЗМОЖЕН РАЗВОРОТ - рассмотрите {action}")
                if candle['is_current']:
                    print(f"   🚀 ДЕЙСТВИЕ: ТЕКУЩАЯ СВЕЧА - вход при подтверждении в следующей свече!")
            else:
                print(f"   💡 СИГНАЛ: Требуется дополнительное подтверждение")

    else:
        print(f"\n❌ Не найдено свечей с длинными тенями по заданным критериям")
        print(f"   💡 Рекомендация: Попробуйте ослабить критерии поиска")

    return results


# Функция для тестирования
def test_long_shadow_strategy():
    """Тестирование стратегии длинных теней"""
    try:
        results = main_long_shadow_scan()
        success = len(results) > 0
        print(
            f"\n{'✅' if success else '⚠️'} Сканирование свечей с длинными тенями завершено! Найдено {len(results)} сигналов")
        return success
    except Exception as e:
        print(f"\n❌ Критическая ошибка в сканировании: {e}")
        return False


# Запуск сканирования
if __name__ == "__main__":
    print("Запуск сканирования свечей с очень длинными тенями...")
    test_long_shadow_strategy()