import time
import numpy as np
import pandas as pd
import requests
import talib
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


def get_high_volume_symbols(min_volume=30000000):
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

        # Фильтруем по объему и волатильности
        return sorted(symbols, key=lambda x: x['volume_24h'], reverse=True)
    except Exception as e:
        print(f"Ошибка получения списка монет: {e}")
        return []


class ImprovedPricePredictor:
    def __init__(self):
        self.historical_predictions = []

    def get_candles(self, symbol, interval="Min30", limit=150):
        """Получение данных с улучшенной обработкой"""
        url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
        params = {"interval": interval, "limit": limit}

        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                return data
            print(f"Ошибка API для {symbol}: {response.status_code}")
        except Exception as e:
            print(f"Ошибка получения данных для {symbol}: {e}")
        return None

    def create_dataframe_from_mexc_data(self, data):
        """Создание DataFrame с правильной обработкой данных MEXC API"""
        if not data or not data.get('success') or not data.get('data'):
            return None

        raw_data = data['data']

        try:
            if isinstance(raw_data, dict):
                required_fields = ['time', 'open', 'close', 'high', 'low', 'vol']
                if not all(field in raw_data for field in required_fields):
                    return None

                # Создаем DataFrame
                df = pd.DataFrame({
                    'timestamp': raw_data['time'],
                    'open': raw_data['open'],
                    'high': raw_data['high'],
                    'low': raw_data['low'],
                    'close': raw_data['close'],
                    'volume': raw_data['vol']
                })

                # Преобразование числовых колонок
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                df = df.dropna()

                if len(df) < 50:  # Увеличили минимальное количество данных
                    return None

                # Добавляем временные метки и сортируем
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('datetime').reset_index(drop=True)

                return df
            else:
                return None

        except Exception as e:
            print(f"Ошибка создания DataFrame: {e}")
            return None

    def get_multiple_timeframes(self, symbol):
        """Получение данных с нескольких таймфреймов для лучшего анализа"""
        timeframes = {
            '1h': 'Min60',
            '4h': 'Hour4',
            '30m': 'Min30'
        }

        multi_data = {}
        for tf_name, tf_api in timeframes.items():
            data = self.get_candles(symbol, tf_api, 100)
            if data and data.get('data'):
                df = self.create_dataframe_from_mexc_data(data)
                if df is not None and len(df) > 30:
                    multi_data[tf_name] = df
            time.sleep(0.3)

        return multi_data

    def calculate_advanced_indicators(self, df):
        """Расчет расширенных технических индикаторов для 5-часового прогноза"""
        if len(df) < 60:
            return df

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        try:
            # Трендовые индикаторы
            df['sma_20'] = talib.SMA(close, timeperiod=20)
            df['sma_50'] = talib.SMA(close, timeperiod=50)
            df['ema_12'] = talib.EMA(close, timeperiod=12)
            df['ema_26'] = talib.EMA(close, timeperiod=26)
            df['ema_50'] = talib.EMA(close, timeperiod=50)

            # MACD с разными настройками
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)

            # RSI на разных периодах
            df['rsi_14'] = talib.RSI(close, timeperiod=14)
            df['rsi_21'] = talib.RSI(close, timeperiod=21)

            # Stochastic
            df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, close)

            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)

            # Волатильность
            df['atr'] = talib.ATR(high, low, close, timeperiod=14)
            df['natr'] = talib.NATR(high, low, close, timeperiod=14)

            # Сила тренда
            df['adx'] = talib.ADX(high, low, close, timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)

            # Объемы
            df['volume_sma'] = talib.SMA(volume, timeperiod=20)
            df['volume_ratio'] = volume / df['volume_sma']
            df['obv'] = talib.OBV(close, volume)

            # Дополнительные осцилляторы для 5-часового прогноза
            df['cci'] = talib.CCI(high, low, close, timeperiod=20)
            df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
            df['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)

            # Price action features
            df['price_change_5h'] = (close / np.roll(close, 10) - 1) * 100  # Примерно 5 часов для 30m TF
            df['volatility_5h'] = talib.STDDEV(close, timeperiod=10) / close * 100

            # Support/Resistance levels
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support'] = df['low'].rolling(window=20).min()

        except Exception as e:
            print(f"Ошибка расчета индикаторов: {e}")

        return df

    def analyze_market_structure(self, df):
        """Анализ структуры рынка для 5-часового прогноза"""
        if len(df) < 50:
            return {'trend': 'NEUTRAL', 'strength': 0, 'momentum': 0, 'rsi': 50, 'adx': 25, 'volatility': 2}

        try:
            close = df['close'].values

            # Мультитаймфреймный анализ тренда
            trend_indicators = 0

            # SMA анализ
            if df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1]:
                trend_indicators += 1
            else:
                trend_indicators -= 1

            # EMA анализ
            if df['ema_12'].iloc[-1] > df['ema_26'].iloc[-1]:
                trend_indicators += 1
            else:
                trend_indicators -= 1

            # MACD анализ
            if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
                trend_indicators += 1
            else:
                trend_indicators -= 1

            # Определение тренда
            if trend_indicators >= 2:
                trend = "STRONG_BULLISH"
            elif trend_indicators >= 1:
                trend = "BULLISH"
            elif trend_indicators <= -2:
                trend = "STRONG_BEARISH"
            elif trend_indicators <= -1:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"

            # Сила тренда
            adx = df['adx'].iloc[-1] if not np.isnan(df['adx'].iloc[-1]) else 25
            strength = min(100, adx * 2)

            # Моментум
            rsi = df['rsi_14'].iloc[-1] if not np.isnan(df['rsi_14'].iloc[-1]) else 50
            macd_hist = df['macd_hist'].iloc[-1] if not np.isnan(df['macd_hist'].iloc[-1]) else 0
            momentum_score = (rsi - 50) / 50 + macd_hist * 10

            # Волатильность для расчета дельты
            volatility = df['volatility_5h'].iloc[-1] if 'volatility_5h' in df.columns else 2

            return {
                'trend': trend,
                'strength': strength,
                'momentum': momentum_score,
                'rsi': rsi,
                'adx': adx,
                'volatility': volatility
            }

        except Exception as e:
            print(f"Ошибка анализа структуры рынка: {e}")
            return {'trend': 'NEUTRAL', 'strength': 0, 'momentum': 0, 'rsi': 50, 'adx': 25, 'volatility': 2}

    def calculate_profit_targets(self, df, direction, current_price, volatility):
        """Расчет целевых уровней для 5-часовой сделки"""
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02

        # Базовые уровни на основе ATR
        if direction == "LONG":
            # Для лонга: цель 3-4 ATR, стоп 1-1.5 ATR
            target_1 = current_price + (atr * 3.0)  # Консервативная цель
            target_2 = current_price + (atr * 4.0)  # Агрессивная цель
            stop_loss = current_price - (atr * 1.2)  # Стоп-лосс

        elif direction == "SHORT":
            # Для шорта: цель 3-4 ATR, стоп 1-1.5 ATR
            target_1 = current_price - (atr * 3.0)  # Консервативная цель
            target_2 = current_price - (atr * 4.0)  # Агрессивная цель
            stop_loss = current_price + (atr * 1.2)  # Стоп-лосс
        else:
            target_1 = target_2 = stop_loss = current_price

        # Корректировка на основе волатильности
        volatility_factor = max(0.5, min(2.0, volatility / 3))
        target_1 = current_price + (target_1 - current_price) * volatility_factor
        target_2 = current_price + (target_2 - current_price) * volatility_factor

        return {
            'target_1': round(target_1, 6),
            'target_2': round(target_2, 6),
            'stop_loss': round(stop_loss, 6),
            'profit_ratio': round(abs(target_1 - current_price) / abs(stop_loss - current_price), 2)
        }

    def collect_trading_signals(self, df):
        """Сбор торговых сигналов для 5-часового прогноза"""
        signals = {}

        try:
            # Трендовые сигналы (вес 40%)
            signals['sma_trend'] = 1 if df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1] else -1
            signals['ema_trend'] = 1 if df['ema_12'].iloc[-1] > df['ema_26'].iloc[-1] else -1
            signals['macd_trend'] = 1 if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else -1

            # Моментум сигналы (вес 30%)
            rsi = df['rsi_14'].iloc[-1]
            if rsi > 65:
                signals['rsi_signal'] = -1  # Перекупленность
            elif rsi < 35:
                signals['rsi_signal'] = 1  # Перепроданность
            else:
                signals['rsi_signal'] = 0

            stoch_k = df['stoch_k'].iloc[-1]
            if stoch_k > 80:
                signals['stoch_signal'] = -1
            elif stoch_k < 20:
                signals['stoch_signal'] = 1
            else:
                signals['stoch_signal'] = 0

            # Сигналы волатильности (вес 20%)
            bb_position = (df['close'].iloc[-1] - df['bb_lower'].iloc[-1]) / (
                        df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1])
            if bb_position > 0.7:
                signals['bb_signal'] = -1  # У верхней границы
            elif bb_position < 0.3:
                signals['bb_signal'] = 1  # У нижней границы
            else:
                signals['bb_signal'] = 0

            # Сигналы объема (вес 10%)
            volume_ratio = df['volume_ratio'].iloc[-1]
            signals['volume_signal'] = 1 if volume_ratio > 1.5 else -1 if volume_ratio < 0.7 else 0

        except Exception as e:
            print(f"Ошибка сбора сигналов: {e}")
            signals = {}

        return signals

    def calculate_signal_strength(self, signals, market_structure):
        """Расчет силы сигнала для 5-часового прогноза"""
        if not signals:
            return 0

        weights = {
            'trend': 0.4,
            'momentum': 0.3,
            'volatility': 0.2,
            'volume': 0.1
        }

        # Трендовые сигналы
        trend_score = (signals.get('sma_trend', 0) +
                       signals.get('ema_trend', 0) +
                       signals.get('macd_trend', 0)) / 3

        # Моментум сигналы
        momentum_score = (signals.get('rsi_signal', 0) +
                          signals.get('stoch_signal', 0)) / 2

        # Волатильность сигналы
        volatility_score = signals.get('bb_signal', 0)

        # Объемные сигналы
        volume_score = signals.get('volume_signal', 0)

        # Итоговый счет
        final_score = (trend_score * weights['trend'] +
                       momentum_score * weights['momentum'] +
                       volatility_score * weights['volatility'] +
                       volume_score * weights['volume'])

        # Усиление за счет силы тренда
        if market_structure['trend'] in ['STRONG_BULLISH', 'STRONG_BEARISH']:
            final_score *= 1.3
        elif market_structure['trend'] in ['BULLISH', 'BEARISH']:
            final_score *= 1.1

        return final_score

    def generate_5h_prediction(self, df, multi_data=None):
        """Генерация прогноза на 5 часов"""
        if len(df) < 60:
            return None

        try:
            current_price = df['close'].iloc[-1]
            market_structure = self.analyze_market_structure(df)
            signals = self.collect_trading_signals(df)

            # Расчет силы сигнала
            signal_strength = self.calculate_signal_strength(signals, market_structure)

            # Определение направления
            if signal_strength > 0.2:
                direction = "LONG"
                confidence = min(95, 60 + (signal_strength * 30))
            elif signal_strength < -0.2:
                direction = "SHORT"
                confidence = min(95, 60 + (abs(signal_strength) * 30))
            else:
                direction = "NEUTRAL"
                confidence = 40

            # Расчет целевых уровней с хорошей дельтой
            targets = self.calculate_profit_targets(df, direction, current_price, market_structure['volatility'])

            # Проверка качества сделки (минимум 1:2 риск/прибыль)
            if targets['profit_ratio'] < 2.0 and direction != "NEUTRAL":
                print(f"   ⚠️  Низкое соотношение прибыли к риску: {targets['profit_ratio']}:1")
                # Увеличиваем цель для улучшения соотношения
                if direction == "LONG":
                    targets['target_1'] = current_price + (targets['target_1'] - current_price) * 1.2
                    targets['target_2'] = current_price + (targets['target_2'] - current_price) * 1.2
                elif direction == "SHORT":
                    targets['target_1'] = current_price - (current_price - targets['target_1']) * 1.2
                    targets['target_2'] = current_price - (current_price - targets['target_2']) * 1.2
                targets['profit_ratio'] = round(
                    abs(targets['target_1'] - current_price) / abs(targets['stop_loss'] - current_price), 2)

            # Генерация причины
            reason = self.generate_trade_reason(direction, signals, market_structure, targets['profit_ratio'])

            prediction = {
                'direction': direction,
                'confidence': round(confidence, 1),
                'current_price': round(current_price, 6),
                'target_1': targets['target_1'],
                'target_2': targets['target_2'],
                'stop_loss': targets['stop_loss'],
                'profit_ratio': targets['profit_ratio'],
                'potential_profit_1': round(abs(targets['target_1'] - current_price) / current_price * 100, 2),
                'potential_profit_2': round(abs(targets['target_2'] - current_price) / current_price * 100, 2),
                'timeframe': '5 hours',
                'reason': reason,
                'signal_strength': round(signal_strength, 3),
                'rsi': round(market_structure['rsi'], 1),
                'trend_strength': round(market_structure['strength'], 1),
                'volatility': round(market_structure['volatility'], 2)
            }

            return prediction

        except Exception as e:
            print(f"Ошибка генерации прогноза: {e}")
            return None

    def generate_trade_reason(self, direction, signals, market_structure, profit_ratio):
        """Генерация обоснования для сделки"""
        reasons = []

        if direction == "LONG":
            reasons.append("📈 Бычьи сигналы:")
            if signals.get('sma_trend') == 1:
                reasons.append("SMA восходящий")
            if signals.get('ema_trend') == 1:
                reasons.append("EMA восходящий")
            if signals.get('rsi_signal') == 1:
                reasons.append("RSI перепродан")
            if market_structure['trend'] in ['BULLISH', 'STRONG_BULLISH']:
                reasons.append(f"Сильный тренд (ADX: {market_structure['strength']}%)")

        elif direction == "SHORT":
            reasons.append("📉 Медвежьи сигналы:")
            if signals.get('sma_trend') == -1:
                reasons.append("SMA нисходящий")
            if signals.get('ema_trend') == -1:
                reasons.append("EMA нисходящий")
            if signals.get('rsi_signal') == -1:
                reasons.append("RSI перекуплен")
            if market_structure['trend'] in ['BEARISH', 'STRONG_BEARISH']:
                reasons.append(f"Сильный тренд (ADX: {market_structure['strength']}%)")
        else:
            reasons.append("⚪ Рынок в консолидации")

        # Добавляем информацию о профитности
        if profit_ratio >= 3:
            reasons.append(f"🔥 Отличное соотношение риска {profit_ratio}:1")
        elif profit_ratio >= 2:
            reasons.append(f"✅ Хорошее соотношение риска {profit_ratio}:1")
        else:
            reasons.append(f"⚠️  Соотношение риска {profit_ratio}:1")

        return " | ".join(reasons)

    def analyze_symbol(self, symbol):
        """Полный анализ символа для 5-часовой сделки"""
        print(f"🔍 Анализируем {symbol} для 5-часовой сделки...")

        try:
            # Получаем данные с нескольких таймфреймов
            multi_data = self.get_multiple_timeframes(symbol)
            if not multi_data or '30m' not in multi_data:
                print(f"   ❌ Недостаточно данных для {symbol}")
                return None

            df = multi_data['30m']

            # Расчет индикаторов
            df = self.calculate_advanced_indicators(df)

            # Генерация прогноза
            prediction = self.generate_5h_prediction(df, multi_data)

            if prediction:
                prediction['symbol'] = symbol
                prediction['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                prediction['volume_ratio'] = round(df['volume_ratio'].iloc[-1],
                                                   2) if 'volume_ratio' in df.columns else 1.0

                # Проверяем качество сигнала
                if prediction['direction'] != "NEUTRAL" and prediction['profit_ratio'] >= 2.0:
                    print(
                        f"   ✅ СИГНАЛ: {prediction['direction']} (уверенность: {prediction['confidence']}%, профит: {prediction['profit_ratio']}:1)")
                else:
                    print(f"   ⚪ НЕЙТРАЛЬНО: {prediction['direction']} (профит: {prediction['profit_ratio']}:1)")

                return prediction
            else:
                print(f"   ❌ Не удалось сгенерировать прогноз для {symbol}")

        except Exception as e:
            print(f"   ❌ Ошибка анализа {symbol}: {e}")

        return None


def main_analysis():
    """Основная функция анализа для 5-часовых сделок"""
    print("🚀 ЗАПУСК АНАЛИЗА ДЛЯ 5-ЧАСОВЫХ СДЕЛОК С ХОРОШЕЙ ДЕЛЬТОЙ")
    print("=" * 80)

    predictor = ImprovedPricePredictor()

    # Получаем список монет с высоким объемом
    symbols_data = get_high_volume_symbols(min_volume=50000000)
    if not symbols_data:
        print("❌ Не удалось получить список монет")
        return []

    # Берем топ-5 монет для анализа
    symbols = [item['symbol'] for item in symbols_data]

    print(f"📊 Анализируем {len(symbols)} монет для 5-часовых сделок...")
    print("🎯 Цель: сделки с соотношением риска не менее 1:2")
    print("=" * 80)

    results = []

    for symbol in symbols:
        prediction = predictor.analyze_symbol(symbol)
        if prediction:
            results.append(prediction)
        time.sleep(1)

    # Сортируем по уверенности и качеству сделки
    results.sort(key=lambda x: (x['direction'] != "NEUTRAL", x['profit_ratio'], x['confidence']), reverse=True)

    # Вывод результатов
    print(f"\n{'=' * 80}")
    print("🎯 ЛУЧШИЕ 5-ЧАСОВЫЕ СДЕЛКИ:")
    print(f"{'=' * 80}")

    profitable_trades = [r for r in results if r['direction'] != "NEUTRAL" and r['profit_ratio'] >= 2.0]

    if profitable_trades:
        for i, result in enumerate(profitable_trades, 1):
            direction_icon = "🟢" if result['direction'] == 'LONG' else "🔴"
            profit_quality = "🔥" if result['profit_ratio'] >= 3 else "✅"

            print(f"\n{i}. {direction_icon} {result['symbol']} {profit_quality}")
            print(f"   Направление: {result['direction']}")
            print(f"   Уверенность: {result['confidence']}%")
            print(f"   Текущая цена: {result['current_price']}")
            print(f"   Цель 1: {result['target_1']} (+{result['potential_profit_1']}%)")
            print(f"   Цель 2: {result['target_2']} (+{result['potential_profit_2']}%)")
            print(f"   Стоп-лосс: {result['stop_loss']}")
            print(f"   Соотношение: {result['profit_ratio']}:1")
            print(f"   RSI: {result['rsi']}")
            print(f"   Волатильность: {result['volatility']}%")
            print(f"   Причина: {result['reason']}")
    else:
        print("\n❌ Не найдено качественных сделок с соотношением риска 1:2")
        # Показываем нейтральные сигналы
        neutral_trades = [r for r in results if r['direction'] == "NEUTRAL"]
        if neutral_trades:
            print(f"\n⚪ Нейтральные сигналы ({len(neutral_trades)}):")
            for trade in neutral_trades:
                print(f"   {trade['symbol']}: {trade['reason']}")

    print(f"\n📈 Статистика: {len(profitable_trades)} прибыльных сделок из {len(results)} проанализированных")

    return profitable_trades


def test_5_hours():
    """Функция для тестирования"""
    try:
        results = main_analysis()
        success = len(results) > 0
        print(f"\n{'✅' if success else '❌'} Тест завершен! Найдено {len(results)} качественных сделок")
        return success
    except Exception as e:
        print(f"\n❌ Критическая ошибка в тесте: {e}")
        return False


# Запуск улучшенного анализа
if __name__ == "__main__":
    test_5_hours()