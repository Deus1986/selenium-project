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


class OneHourCandleTrader:
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

            # Дополнительные индикаторы
            df['price_vs_ema9'] = (close / df['ema_9'] - 1) * 100
            df['price_vs_ema21'] = (close / df['ema_21'] - 1) * 100

            # Анализ свечей
            df['candle_body'] = abs(df['close'] - df['open'])
            df['candle_size'] = df['high'] - df['low']
            df['body_ratio'] = np.where(df['candle_size'] > 0, df['candle_body'] / df['candle_size'], 0)
            df['body_percent'] = (df['candle_body'] / df['close']) * 100
            df['volume_zscore'] = (volume - df['volume_sma']) / df['volume_sma'].std()

            df = df.fillna(method='bfill').fillna(method='ffill')
            print(f"   ✅ Рассчитаны индикаторы")

        except Exception as e:
            print(f"   ❌ Ошибка расчета индикаторов: {e}")
            # Заглушки для индикаторов
            indicator_defaults = {
                'stoch_k': 50, 'stoch_d': 50, 'rsi_14': 50,
                'macd': 0, 'macd_signal': 0, 'atr': df['close'] * 0.02,
                'body_percent': 1.0, 'volume_zscore': 0, 'volume_ratio': 1.0
            }
            for col, default_val in indicator_defaults.items():
                df[col] = default_val

        return df

    def find_current_small_body_high_volume_candle(self, df, min_volume_ratio=2.0, max_body_percent=0.5):
        """
        Поиск СВЕЖЕЙ (последней) свечи с маленьким телом но большим объемом

        Args:
            df: DataFrame с данными свечей
            min_volume_ratio: Минимальное отношение объема к среднему
            max_body_percent: Максимальный размер тела свечи в процентах от цены
        """
        if len(df) < 3:
            print(f"   ⚠️ Недостаточно данных для анализа")
            return None

        # Берем только последнюю (текущую) свечу
        current_row = df.iloc[-1]

        # Проверяем условия для последней свечи
        if (current_row['body_percent'] <= max_body_percent and
                current_row['volume_ratio'] >= min_volume_ratio):

            candle_info = {
                'timestamp': current_row['timestamp'],
                'datetime': current_row['datetime'],
                'open': current_row['open'],
                'high': current_row['high'],
                'low': current_row['low'],
                'close': current_row['close'],
                'volume': current_row['volume'],
                'body_percent': round(current_row['body_percent'], 3),
                'volume_ratio': round(current_row['volume_ratio'], 2),
                'volume_zscore': round(current_row['volume_zscore'], 2),
                'candle_type': 'DOJI' if abs(current_row['close'] - current_row['open']) / current_row[
                    'candle_size'] < 0.1 else 'SMALL_BODY',
                'price_vs_ema9': round(current_row['price_vs_ema9'], 2),
                'price_vs_ema21': round(current_row['price_vs_ema21'], 2),
                'rsi': round(current_row['rsi_14'], 1) if 'rsi_14' in current_row else 50,
                'is_current': True
            }

            print(f"   ✅ Найдена ТЕКУЩАЯ свеча с маленьким телом и большим объемом")
            return candle_info
        else:
            print(
                f"   ❌ Текущая свеча не соответствует критериям (тело: {current_row['body_percent']:.3f}%, объем: {current_row['volume_ratio']:.2f}x)")
            return None

    def analyze_current_small_body_high_volume(self, symbol, min_volume_ratio=2.0, max_body_percent=0.5):
        """
        Анализ ТЕКУЩЕЙ свечи на наличие паттерна маленького тела и большого объема
        """
        print(f"🔍 Поиск ТЕКУЩЕЙ свечи с маленьким телом и большим объемом для {symbol}...")

        try:
            # Получаем данные (берем больше свечей для расчета индикаторов)
            data = self.get_candles(symbol, "Min60", 100)
            if not data:
                print(f"   ❌ Нет данных для {symbol}")
                return None

            df = self.create_dataframe(data)
            if df is None:
                print(f"   ❌ Не удалось создать DataFrame для {symbol}")
                return None

            # Расчет индикаторов
            df = self.calculate_hourly_indicators(df)

            # Поиск ТЕКУЩЕЙ свечи с маленьким телом и большим объемом
            special_candle = self.find_current_small_body_high_volume_candle(
                df,
                min_volume_ratio=min_volume_ratio,
                max_body_percent=max_body_percent
            )

            if special_candle:
                # Определяем потенциальное направление (на основе текущих индикаторов)
                direction = self.analyze_current_direction(df)

                result = {
                    'symbol': symbol,
                    'special_candle_found': True,
                    'current_candle': special_candle,
                    'potential_direction': direction,
                    'current_price': df['close'].iloc[-1],
                    'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                return result
            else:
                return None

        except Exception as e:
            print(f"   ❌ Ошибка анализа {symbol}: {e}")
            return None

    def analyze_current_direction(self, df):
        """
        Анализ потенциального направления для ТЕКУЩЕЙ ситуации
        """
        try:
            current = df.iloc[-1]

            # Собираем бычьи и медвежьи сигналы
            bullish_signals = 0
            bearish_signals = 0

            # Анализ тренда
            if current['close'] > current['ema_9']:
                bullish_signals += 1
            else:
                bearish_signals += 1

            if current['close'] > current['ema_21']:
                bullish_signals += 1
            else:
                bearish_signals += 1

            # Анализ моментума
            if current['rsi_14'] > 50:
                bullish_signals += 1
            else:
                bearish_signals += 1

            if current['macd'] > current['macd_signal']:
                bullish_signals += 1
            else:
                bearish_signals += 1

            # Определяем общее направление
            if bullish_signals > bearish_signals:
                return "BULLISH_BREAKOUT"
            elif bearish_signals > bullish_signals:
                return "BEARISH_BREAKOUT"
            else:
                return "CONSOLIDATION"

        except Exception as e:
            print(f"   ⚠️ Ошибка анализа направления: {e}")
            return "UNKNOWN"

    def scan_current_small_body_high_volume(self, symbols, min_volume_ratio=2.0, max_body_percent=0.5):
        """
        Сканирование ТЕКУЩИХ свечей на наличие паттерна маленького тела и большого объема
        """
        print("🎯 СКАНИРОВАНИЕ ТЕКУЩИХ СВЕЧЕЙ С МАЛЕНЬКИМ ТЕЛОМ И БОЛЬШИМ ОБЪЕМОМ")
        print("=" * 80)
        print(f"Критерии поиска:")
        print(f"   • ОБЪЕМ ≥ {min_volume_ratio}x от среднего")
        print(f"   • ТЕЛО СВЕЧИ ≤ {max_body_percent}%")
        print(f"   • АНАЛИЗ: ТОЛЬКО ТЕКУЩАЯ (ПОСЛЕДНЯЯ) СВЕЧА")
        print("=" * 80)

        results = []

        for symbol in symbols:
            result = self.analyze_current_small_body_high_volume(
                symbol,
                min_volume_ratio=min_volume_ratio,
                max_body_percent=max_body_percent
            )

            if result and result['special_candle_found']:
                results.append(result)

                # Вывод информации о найденной свече
                candle = result['current_candle']
                direction_icon = "🟢" if result['potential_direction'] == "BULLISH_BREAKOUT" else "🔴" if result[
                                                                                                            'potential_direction'] == "BEARISH_BREAKOUT" else "⚪"

                print(f"\n{direction_icon} {symbol} - ТЕКУЩАЯ СВЕЧА")
                print(f"   📅 Время: {candle['datetime']}")
                print(f"   💰 Цена: {candle['close']}")
                print(f"   📏 Тело: {candle['body_percent']}%")
                print(f"   📊 Объем: {candle['volume_ratio']}x (Z-score: {candle['volume_zscore']})")
                print(f"   🎯 Тип: {candle['candle_type']}")
                print(f"   📈 Направление: {result['potential_direction']}")
                print(f"   ⏰ Анализ: {result['analysis_time']}")

            time.sleep(0.3)  # Пауза между запросами

        # Сортируем по Z-score объема (самые аномальные объемы первыми)
        results.sort(key=lambda x: x['current_candle']['volume_zscore'], reverse=True)

        print(f"\n{'=' * 80}")
        print(f"📊 РЕЗУЛЬТАТЫ СКАНИРОВАНИЯ ТЕКУЩИХ СВЕЧЕЙ:")
        print(f"   • Найдено символов с паттерном: {len(results)}")
        print(f"   • Всего проанализировано: {len(symbols)}")
        print(f"   • Время анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return results


def main_current_small_body_high_volume_scan():
    """Основная функция для поиска ТЕКУЩИХ свечей с маленьким телом и большим объемом"""
    trader = OneHourCandleTrader()

    # Получаем список монет с высоким объемом
    symbols_data = get_high_volume_symbols(min_volume=10000000)
    if not symbols_data:
        print("❌ Не удалось получить список монет")
        return []

    # Берем топ-15 самых ликвидных монет
    symbols = [item['symbol'] for item in symbols_data]

    print(f"📊 Анализируем {len(symbols)} монет на наличие ТЕКУЩЕЙ свечи с маленьким телом и большим объемом...")

    # Запускаем сканирование ТЕКУЩИХ свечей
    results = trader.scan_current_small_body_high_volume(
        symbols,
        min_volume_ratio=1.5,
        max_body_percent=0.5
    )

    # Вывод лучших результатов
    if results:
        print(f"\n🎯 ЛУЧШИЕ ТЕКУЩИЕ СИГНАЛЫ:")
        print(f"{'=' * 80}")

        for i, result in enumerate(results, 1):
            candle = result['current_candle']
            direction_icon = "🟢" if result['potential_direction'] == "BULLISH_BREAKOUT" else "🔴" if result[
                                                                                                        'potential_direction'] == "BEARISH_BREAKOUT" else "⚪"

            print(f"\n{i}. {direction_icon} {result['symbol']}")
            print(f"   📅 Время свечи: {candle['datetime']}")
            print(f"   💰 Текущая цена: {candle['close']}")
            print(f"   📏 Размер тела: {candle['body_percent']}%")
            print(f"   📊 Объем: {candle['volume_ratio']}x (Z-score: {candle['volume_zscore']})")
            print(f"   🎯 Прогноз: {result['potential_direction']}")
            print(f"   🔍 Тип свечи: {candle['candle_type']}")

            # Торговые рекомендации для ТЕКУЩЕЙ свечи
            if result['potential_direction'] in ['BULLISH_BREAKOUT', 'BEARISH_BREAKOUT']:
                print(f"   💡 СИГНАЛ: ТЕКУЩАЯ СВЕЧА - возможен пробой в следующей свече!")
                print(f"   🚀 ДЕЙСТВИЕ: Готовьтесь к входу при подтверждении пробоя")
            else:
                print(f"   💡 СИГНАЛ: ТЕКУЩАЯ СВЕЧА - ожидание подтверждения направления")

    else:
        print(f"\n❌ Не найдено ТЕКУЩИХ свечей с паттерном маленького тела и большого объема")
        print(f"   💡 Рекомендация: Проверьте позже или ослабьте критерии поиска")

    return results


# Функция для тестирования
def test_current_small_body_strategy():
    """Тестирование стратегии на ТЕКУЩИХ свечах"""
    try:
        results = main_current_small_body_high_volume_scan()
        success = len(results) > 0
        print(f"\n{'✅' if success else '⚠️'} Сканирование ТЕКУЩИХ свечей завершено! Найдено {len(results)} сигналов")
        return success
    except Exception as e:
        print(f"\n❌ Критическая ошибка в сканировании: {e}")
        return False


# Запуск сканирования ТЕКУЩИХ свечей
if __name__ == "__main__":
    print("Запуск сканирования ТЕКУЩИХ свечей с маленьким телом и большим объемом...")
    test_current_small_body_strategy()