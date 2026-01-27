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
        self.min_confidence = 75  # Минимальная уверенность для сделки
        self.min_profit_ratio = 2.0  # Минимальное соотношение риск/прибыль

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
            # Обрабатываем разные форматы данных от MEXC
            if isinstance(raw_data, dict):
                # Формат: {'time': [1, 2, 3], 'open': [1, 2, 3], ...}
                required_fields = ['time', 'open', 'close', 'high', 'low', 'vol']

                # Проверяем наличие всех полей
                missing_fields = [field for field in required_fields if field not in raw_data]
                if missing_fields:
                    print(f"   ❌ Отсутствуют поля: {missing_fields}")
                    return None

                # Создаем DataFrame из словаря
                df = pd.DataFrame({
                    'timestamp': raw_data['time'],
                    'open': raw_data['open'],
                    'high': raw_data['high'],
                    'low': raw_data['low'],
                    'close': raw_data['close'],
                    'volume': raw_data['vol']
                })

            elif isinstance(raw_data, list):
                # Формат: [['time', 'open', 'high', 'low', 'close', 'volume', ...], ...]
                if len(raw_data) == 0:
                    return None

                # Проверяем структуру первого элемента
                first_item = raw_data[0]
                if isinstance(first_item, list) and len(first_item) >= 6:
                    # Извлекаем данные из списка списков
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

            # Конвертируем timestamp и сортируем
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('datetime').reset_index(drop=True)

            print(f"   ✅ Успешно создан DataFrame с {len(df)} строками")
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

            # Stochastic - исправленная версия
            stoch_result = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            df['stoch_k'] = stoch_result[0]  # %K line
            df['stoch_d'] = stoch_result[1]  # %D line

            # Волатильность
            df['atr'] = talib.ATR(high, low, close, timeperiod=14)
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close, timeperiod=20)

            # Объем
            df['volume_sma'] = talib.SMA(volume, timeperiod=20)
            df['volume_ratio'] = volume / df['volume_sma']

            # Дополнительные индикаторы для одной свечи
            df['price_vs_ema9'] = (close / df['ema_9'] - 1) * 100
            df['price_vs_ema21'] = (close / df['ema_21'] - 1) * 100

            # Анализ текущей свечи
            df['candle_body'] = abs(df['close'] - df['open'])
            df['candle_size'] = df['high'] - df['low']
            df['body_ratio'] = np.where(df['candle_size'] > 0, df['candle_body'] / df['candle_size'], 0)

            # Заполняем NaN значения
            df = df.fillna(method='bfill').fillna(method='ffill')

            print(f"   ✅ Рассчитаны индикаторы")

        except Exception as e:
            print(f"   ❌ Ошибка расчета индикаторов: {e}")
            # Добавляем заглушки для отсутствующих индикаторов
            df['stoch_k'] = 50
            df['stoch_d'] = 50
            df['rsi_14'] = 50
            df['macd'] = 0
            df['macd_signal'] = 0
            df['atr'] = df['close'] * 0.02

        return df

    def analyze_current_candle(self, df):
        """Анализ текущей часовой свечи"""
        if len(df) < 3:
            return None

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # Проверяем наличие всех необходимых колонок
        required_columns = ['close', 'open', 'candle_body', 'candle_size', 'body_ratio',
                            'volume_ratio', 'rsi_14', 'ema_9', 'ema_21', 'macd', 'macd_signal',
                            'stoch_k', 'stoch_d']

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"   ⚠️ Отсутствуют колонки: {missing_columns}")
            # Создаем заглушки для отсутствующих колонок
            for col in missing_columns:
                if col in ['stoch_k', 'stoch_d', 'rsi_14']:
                    df[col] = 50
                elif col in ['macd', 'macd_signal']:
                    df[col] = 0
                elif col in ['ema_9', 'ema_21']:
                    df[col] = df['close']
                else:
                    df[col] = 0

        # Анализ текущей свечи
        candle_analysis = {
            'is_bullish': current['close'] > current['open'],
            'is_bearish': current['close'] < current['open'],
            'body_size': current['candle_body'],
            'total_size': current['candle_size'],
            'body_ratio': current['body_ratio'],
            'volume_ratio': current['volume_ratio'],
            'rsi': current['rsi_14'],
            'above_ema9': current['close'] > current['ema_9'],
            'above_ema21': current['close'] > current['ema_21'],
            'macd_bullish': current['macd'] > current['macd_signal'],
            'stoch_bullish': current['stoch_k'] > current['stoch_d'] and current['stoch_k'] < 80
        }

        return candle_analysis

    def calculate_entry_signals(self, df, candle_analysis):
        """Расчет сигналов для входа в сделку"""
        if len(df) < 10:
            return None

        current = df.iloc[-1]
        signals = {}

        try:
            # Сигналы для LONG
            long_signals = []

            if candle_analysis['is_bullish']:
                long_signals.append(("Бычья свеча", 15))

            if candle_analysis['above_ema9'] and candle_analysis['above_ema21']:
                long_signals.append(("Выше EMA9/21", 20))

            if 30 < candle_analysis['rsi'] < 70:
                long_signals.append(("RSI в норме", 15))
            elif candle_analysis['rsi'] < 30:
                long_signals.append(("RSI перепродан", 10))

            if candle_analysis['macd_bullish']:
                long_signals.append(("MACD бычий", 15))

            if candle_analysis['stoch_bullish']:
                long_signals.append(("Stoch бычий", 10))

            if candle_analysis['volume_ratio'] > 1.2:
                long_signals.append(("Объем выше среднего", 15))

            # Сигналы для SHORT
            short_signals = []

            if candle_analysis['is_bearish']:
                short_signals.append(("Медвежья свеча", 15))

            if not candle_analysis['above_ema9'] and not candle_analysis['above_ema21']:
                short_signals.append(("Ниже EMA9/21", 20))

            if 30 < candle_analysis['rsi'] < 70:
                short_signals.append(("RSI в норме", 15))
            elif candle_analysis['rsi'] > 70:
                short_signals.append(("RSI перекуплен", 10))

            if not candle_analysis['macd_bullish']:
                short_signals.append(("MACD медвежий", 15))

            if not candle_analysis['stoch_bullish'] and current.get('stoch_k', 50) > 20:
                short_signals.append(("Stoch медвежий", 10))

            if candle_analysis['volume_ratio'] > 1.2:
                short_signals.append(("Объем выше среднего", 15))

            # Расчет уверенности
            long_confidence = sum(score for _, score in long_signals)
            short_confidence = sum(score for _, score in short_signals)

            signals = {
                'long_confidence': min(95, long_confidence),
                'short_confidence': min(95, short_confidence),
                'long_signals': long_signals,
                'short_signals': short_signals,
                'current_rsi': candle_analysis['rsi'],
                'current_volume_ratio': candle_analysis['volume_ratio']
            }

        except Exception as e:
            print(f"Ошибка расчета сигналов: {e}")

        return signals

    def calculate_safe_targets(self, current_price, atr, direction):
        """Расчет безопасных целей для одной свечи"""
        # Консервативные цели для одной свечи
        if direction == "LONG":
            target = current_price + (atr * 1.5)  # 1.5 ATR вверх
            stop_loss = current_price - (atr * 0.7)  # 0.7 ATR вниз
        elif direction == "SHORT":
            target = current_price - (atr * 1.5)  # 1.5 ATR вниз
            stop_loss = current_price + (atr * 0.7)  # 0.7 ATR вверх
        else:
            return None

        profit_ratio = abs(target - current_price) / abs(stop_loss - current_price)

        return {
            'target': round(target, 6),
            'stop_loss': round(stop_loss, 6),
            'profit_ratio': round(profit_ratio, 2),
            'potential_profit': round(abs(target - current_price) / current_price * 100, 2),
            'risk_percent': round(abs(stop_loss - current_price) / current_price * 100, 2)
        }

    def generate_hourly_prediction(self, df):
        """Генерация прогноза для одной часовой свечи"""
        if len(df) < 20:
            return None

        try:
            current_price = df['close'].iloc[-1]
            current_atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02

            # Анализ текущей свечи
            candle_analysis = self.analyze_current_candle(df)
            if not candle_analysis:
                return None

            # Расчет сигналов
            signals = self.calculate_entry_signals(df, candle_analysis)
            if not signals:
                return None

            # Определение направления
            direction = "NEUTRAL"
            confidence = 0

            if (signals['long_confidence'] >= self.min_confidence and
                    signals['long_confidence'] > signals['short_confidence']):
                direction = "LONG"
                confidence = signals['long_confidence']
            elif (signals['short_confidence'] >= self.min_confidence and
                  signals['short_confidence'] > signals['long_confidence']):
                direction = "SHORT"
                confidence = signals['short_confidence']

            # Расчет целей
            targets = self.calculate_safe_targets(current_price, current_atr, direction)
            if not targets:
                return None

            # Проверка качества сделки
            is_quality_trade = (
                    direction != "NEUTRAL" and
                    confidence >= self.min_confidence and
                    targets['profit_ratio'] >= self.min_profit_ratio and
                    targets['risk_percent'] <= 2.0  # Максимум 2% риска
            )

            # Генерация причины
            reason = self.generate_trade_reason(direction, signals, targets, is_quality_trade)

            prediction = {
                'direction': direction,
                'confidence': confidence,
                'current_price': round(current_price, 6),
                'target': targets['target'],
                'stop_loss': targets['stop_loss'],
                'profit_ratio': targets['profit_ratio'],
                'potential_profit': targets['potential_profit'],
                'risk_percent': targets['risk_percent'],
                'is_quality_trade': is_quality_trade,
                'timeframe': '1 hour',
                'reason': reason,
                'rsi': round(signals['current_rsi'], 1),
                'volume_ratio': round(signals['current_volume_ratio'], 2),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            return prediction

        except Exception as e:
            print(f"Ошибка генерации прогноза: {e}")
            return None

    def generate_trade_reason(self, direction, signals, targets, is_quality):
        """Генерация обоснования для сделки"""
        reasons = []

        if direction == "LONG":
            reasons.append("🟢 ЧАСОВОЙ ЛОНГ:")
            for signal_name, score in signals['long_signals']:
                reasons.append(signal_name)
        elif direction == "SHORT":
            reasons.append("🔴 ЧАСОВОЙ ШОРТ:")
            for signal_name, score in signals['short_signals']:
                reasons.append(signal_name)
        else:
            reasons.append("⚪ НЕТ СИГНАЛА")

        # Информация о качестве
        if is_quality:
            reasons.append(f"✅ КАЧЕСТВО: {targets['profit_ratio']}:1")
            reasons.append(f"🛡️ РИСК: {targets['risk_percent']}%")
        else:
            reasons.append(f"⚠️ НИЗКОЕ КАЧЕСТВО")

        return " | ".join(reasons[:4])  # Ограничиваем длину

    def analyze_symbol_for_hourly(self, symbol):
        """Анализ символа для часовой сделки"""
        print(f"🔍 Анализ {symbol} для часовой свечи...")

        try:
            # Получаем часовые данные
            data = self.get_candles(symbol, "Min60", 100)
            if not data:
                print(f"   ❌ Нет данных для {symbol}")
                return None

            # Отладочная информация о формате данных
            if 'data' in data:
                raw_data = data['data']
                print(f"   📊 Формат данных: {type(raw_data)}, успех: {data.get('success')}")
                if isinstance(raw_data, dict):
                    print(f"   📊 Ключи словаря: {list(raw_data.keys())}")
                    for key in ['time', 'open', 'close']:
                        if key in raw_data:
                            print(
                                f"   📊 {key}: {type(raw_data[key])}, длина: {len(raw_data[key]) if hasattr(raw_data[key], '__len__') else 'N/A'}")
            else:
                print(f"   ❌ Нет поля 'data' в ответе")

            df = self.create_dataframe(data)
            if df is None:
                print(f"   ❌ Не удалось создать DataFrame для {symbol}")
                return None

            # Расчет индикаторов
            df = self.calculate_hourly_indicators(df)

            # Проверяем наличие необходимых индикаторов
            required_indicators = ['stoch_k', 'stoch_d', 'rsi_14', 'macd', 'atr']
            missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
            if missing_indicators:
                print(f"   ⚠️ Отсутствуют индикаторы: {missing_indicators}")

            # Генерация прогноза
            prediction = self.generate_hourly_prediction(df)

            if prediction:
                prediction['symbol'] = symbol

                # Проверяем качество сигнала
                if prediction['is_quality_trade']:
                    print(
                        f"   ✅ КАЧЕСТВЕННЫЙ СИГНАЛ: {prediction['direction']} (уверенность: {prediction['confidence']}%, профит: {prediction['profit_ratio']}:1)")
                else:
                    print(f"   ⚪ СИГНАЛ: {prediction['direction']} (уверенность: {prediction['confidence']}%)")

                return prediction
            else:
                print(f"   ❌ Нет сигнала для {symbol}")

        except Exception as e:
            print(f"   ❌ Ошибка анализа {symbol}: {e}")

        return None


def main_hourly_trading():
    """Основная функция для торговли по часовым свечам"""
    print("⏰ СТРАТЕГИЯ ТОРГОВЛИ ОДНОЙ ЧАСОВОЙ СВЕЧОЙ")
    print("=" * 80)
    print("🎯 Критерии качественной сделки:")
    print("   • Уверенность ≥ 75%")
    print("   • Соотношение риск/прибыль ≥ 2:1")
    print("   • Риск ≤ 2% от депозита")
    print("   • Сильные сигналы на текущей свече")
    print("=" * 80)

    trader = OneHourCandleTrader()

    # Получаем список монет
    symbols_data = get_high_volume_symbols(min_volume=20000000)
    if not symbols_data:
        print("❌ Не удалось получить список монет")
        return []

    # Берем топ-10 самых ликвидных монет
    symbols = [item['symbol'] for item in symbols_data[:10]]

    print(f"📊 Анализируем {len(symbols)} монет для часовых сделок...")
    print("=" * 80)

    results = []
    quality_trades = []

    for symbol in symbols:
        prediction = trader.analyze_symbol_for_hourly(symbol)
        if prediction:
            results.append(prediction)
            if prediction['is_quality_trade']:
                quality_trades.append(prediction)
        time.sleep(0.5)  # Пауза между запросами

    # Сортируем качественные сделки
    quality_trades.sort(key=lambda x: (x['profit_ratio'], x['confidence']), reverse=True)

    # Вывод результатов
    print(f"\n{'=' * 80}")
    print("🎯 КАЧЕСТВЕННЫЕ ЧАСОВЫЕ СДЕЛКИ:")
    print(f"{'=' * 80}")

    if quality_trades:
        for i, trade in enumerate(quality_trades, 1):
            direction_icon = "🟢" if trade['direction'] == 'LONG' else "🔴"

            print(f"\n{i}. {direction_icon} {trade['symbol']} ⏰")
            print(f"   Направление: {trade['direction']}")
            print(f"   Уверенность: {trade['confidence']}%")
            print(f"   Текущая цена: {trade['current_price']}")
            print(f"   Цель: {trade['target']} (+{trade['potential_profit']}%)")
            print(f"   Стоп-лосс: {trade['stop_loss']} (-{trade['risk_percent']}%)")
            print(f"   Соотношение: {trade['profit_ratio']}:1")
            print(f"   RSI: {trade['rsi']}")
            print(f"   Объем: {trade['volume_ratio']}x")
            print(f"   Причина: {trade['reason']}")

            # Рекомендации по управлению
            print(f"   💡 УПРАВЛЕНИЕ:")
            print(f"      • Вход: рынок")
            print(f"      • Тейк-профит: {trade['target']}")
            print(f"      • Стоп-лосс: {trade['stop_loss']}")
            print(f"      • Срок: 1-2 часа")
    else:
        print("\n❌ Не найдено качественных сделок для часовой торговли")

        # Показываем потенциальные сделки
        potential = [r for r in results if not r['is_quality_trade'] and r['direction'] != 'NEUTRAL']
        if potential:
            print(f"\n⚠️  ПОТЕНЦИАЛЬНЫЕ СДЕЛКИ (требуют осторожности):")
            for trade in potential:
                direction_icon = "🟢" if trade['direction'] == 'LONG' else "🔴"
                print(
                    f"   {direction_icon} {trade['symbol']}: уверенность {trade['confidence']}%, профит {trade['profit_ratio']}:1")

    print(f"\n📈 СТАТИСТИКА:")
    print(f"   • Качественных сделок: {len(quality_trades)}")
    print(f"   • Всего проанализировано: {len(results)}")
    print(f"   • Эффективность: {len(quality_trades) / len(results) * 100:.1f}%" if results else "0%")

    return quality_trades


def test_hourly_strategy():
    """Тестирование часовой стратегии"""
    try:
        results = main_hourly_trading()
        success = len(results) > 0
        print(f"\n{'✅' if success else '⚠️'} Тест завершен! Найдено {len(results)} качественных часовых сделок")
        return success
    except Exception as e:
        print(f"\n❌ Критическая ошибка в тесте: {e}")
        return False


# Запуск часовой стратегии
if __name__ == "__main__":
    test_hourly_strategy()