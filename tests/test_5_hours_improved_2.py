import time
import numpy as np
import pandas as pd
import requests
import talib
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


def get_high_volume_symbols(min_volume=20000000):
    """Получает список монет с высоким объемом и стабильной ценой"""
    url = "https://contract.mexc.com/api/v1/contract/ticker"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        symbols = []

        if 'data' in data:
            for item in data["data"]:
                # Фильтруем по объему и стабильности цены
                if (item["amount24"] > min_volume and
                        abs(float(item['riseFallRate']) * 100) < 15):  # Исключаем слишком волатильные
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


class ConservativePricePredictor:
    def __init__(self):
        self.historical_predictions = []
        self.min_confidence = 70  # Минимальная уверенность для сделки
        self.min_profit_ratio = 2.5  # Минимальное соотношение риск/прибыль

    def get_candles(self, symbol, interval="Min30", limit=200):
        """Получение данных с улучшенной обработкой"""
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

                if len(df) < 80:  # Увеличили минимальное количество данных для надежности
                    return None

                # Добавляем временные метки и сортируем
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('datetime').reset_index(drop=True)

                return df
            return None

        except Exception as e:
            print(f"Ошибка создания DataFrame: {e}")
            return None

    def get_conservative_timeframes(self, symbol):
        """Получение данных с консервативных таймфреймов"""
        timeframes = {
            '4h': 'Hour4',  # Основной для тренда
            '1h': 'Min60',  # Вторичный для подтверждения
            '30m': 'Min30'  # Для точного входа
        }

        multi_data = {}
        for tf_name, tf_api in timeframes.items():
            data = self.get_candles(symbol, tf_api, 150)
            if data and data.get('data'):
                df = self.create_dataframe_from_mexc_data(data)
                if df is not None and len(df) > 50:
                    multi_data[tf_name] = df
            time.sleep(0.5)  # Увеличили паузу для стабильности

        return multi_data

    def calculate_conservative_indicators(self, df):
        """Расчет консервативных технических индикаторов"""
        if len(df) < 80:
            return df

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        try:
            # Консервативные трендовые индикаторы
            df['sma_50'] = talib.SMA(close, timeperiod=50)
            df['sma_100'] = talib.SMA(close, timeperiod=100)
            df['ema_20'] = talib.EMA(close, timeperiod=20)
            df['ema_50'] = talib.EMA(close, timeperiod=50)

            # MACD с консервативными настройками
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close, fastperiod=12, slowperiod=26,
                                                                        signalperiod=9)

            # RSI с фильтрацией
            df['rsi_14'] = talib.RSI(close, timeperiod=14)
            df['rsi_21'] = talib.RSI(close, timeperiod=21)

            # Bollinger Bands с увеличенным периодом
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)

            # Волатильность
            df['atr'] = talib.ATR(high, low, close, timeperiod=14)
            df['natr'] = talib.NATR(high, low, close, timeperiod=14)

            # Сила тренда
            df['adx'] = talib.ADX(high, low, close, timeperiod=14)

            # Объемы
            df['volume_sma'] = talib.SMA(volume, timeperiod=20)
            df['volume_ratio'] = volume / df['volume_sma']

            # Дополнительные консервативные индикаторы
            df['cci'] = talib.CCI(high, low, close, timeperiod=20)

            # Support/Resistance levels
            df['resistance'] = df['high'].rolling(window=25).max()
            df['support'] = df['low'].rolling(window=25).min()

            # Price stability indicators
            df['price_stability'] = talib.STDDEV(close, timeperiod=20) / close * 100

        except Exception as e:
            print(f"Ошибка расчета индикаторов: {e}")

        return df

    def analyze_conservative_trend(self, df_4h, df_1h, df_30m):
        """Консервативный анализ тренда на нескольких таймфреймах"""
        trend_signals = {
            '4h': 'NEUTRAL',
            '1h': 'NEUTRAL',
            '30m': 'NEUTRAL'
        }

        strengths = []

        for tf_name, df in [('4h', df_4h), ('1h', df_1h), ('30m', df_30m)]:
            if df is None or len(df) < 50:
                continue

            # Консервативные условия для тренда
            sma_bullish = df['sma_50'].iloc[-1] > df['sma_100'].iloc[-1]
            ema_bullish = df['ema_20'].iloc[-1] > df['ema_50'].iloc[-1]
            macd_bullish = df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]

            bullish_count = sum([sma_bullish, ema_bullish, macd_bullish])

            if bullish_count >= 2:
                trend_signals[tf_name] = 'BULLISH'
                strengths.append(df['adx'].iloc[-1] if not np.isnan(df['adx'].iloc[-1]) else 0)
            elif bullish_count <= 1:
                trend_signals[tf_name] = 'BEARISH'
                strengths.append(df['adx'].iloc[-1] if not np.isnan(df['adx'].iloc[-1]) else 0)
            else:
                trend_signals[tf_name] = 'NEUTRAL'
                strengths.append(0)

        # Определение общего тренда (требуется согласованность)
        bullish_tf = sum(1 for trend in trend_signals.values() if trend == 'BULLISH')
        bearish_tf = sum(1 for trend in trend_signals.values() if trend == 'BEARISH')

        if bullish_tf >= 2:
            overall_trend = 'BULLISH'
        elif bearish_tf >= 2:
            overall_trend = 'BEARISH'
        else:
            overall_trend = 'NEUTRAL'

        avg_strength = np.mean([s for s in strengths if s > 0]) if strengths else 0

        return {
            'trend': overall_trend,
            'strength': min(100, avg_strength * 2),
            'timeframe_alignment': f"4h:{trend_signals['4h']}, 1h:{trend_signals['1h']}, 30m:{trend_signals['30m']}",
            'rsi_4h': df_4h['rsi_14'].iloc[-1] if df_4h is not None else 50,
            'rsi_1h': df_1h['rsi_14'].iloc[-1] if df_1h is not None else 50
        }

    def calculate_safe_profit_targets(self, df, direction, current_price):
        """Расчет безопасных целевых уровней"""
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.015

        # Консервативные цели: 2-2.5 ATR для цели, 0.8-1 ATR для стопа
        if direction == "LONG":
            target = current_price + (atr * 2.2)  # Консервативная цель
            stop_loss = current_price - (atr * 0.9)  # Близкий стоп-лосс
        elif direction == "SHORT":
            target = current_price - (atr * 2.2)  # Консервативная цель
            stop_loss = current_price + (atr * 0.9)  # Близкий стоп-лосс
        else:
            target = stop_loss = current_price

        profit_ratio = abs(target - current_price) / abs(stop_loss - current_price)

        return {
            'target': round(target, 6),
            'stop_loss': round(stop_loss, 6),
            'profit_ratio': round(profit_ratio, 2),
            'potential_profit': round(abs(target - current_price) / current_price * 100, 2),
            'risk_percent': round(abs(stop_loss - current_price) / current_price * 100, 2)
        }

    def collect_conservative_signals(self, df_4h, df_1h, df_30m):
        """Сбор консервативных торговых сигналов"""
        signals = {}

        try:
            # Используем только 4h и 1h для сигналов (более стабильные)
            for tf_name, df in [('4h', df_4h), ('1h', df_1h)]:
                if df is None:
                    continue

                # Трендовые сигналы
                signals[f'{tf_name}_sma'] = 1 if df['sma_50'].iloc[-1] > df['sma_100'].iloc[-1] else -1
                signals[f'{tf_name}_ema'] = 1 if df['ema_20'].iloc[-1] > df['ema_50'].iloc[-1] else -1
                signals[f'{tf_name}_macd'] = 1 if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else -1

                # Моментум сигналы с консервативными уровнями
                rsi = df['rsi_14'].iloc[-1]
                if rsi > 70:  # Более консервативные уровни
                    signals[f'{tf_name}_rsi'] = -1
                elif rsi < 30:
                    signals[f'{tf_name}_rsi'] = 1
                else:
                    signals[f'{tf_name}_rsi'] = 0

            # Объемные сигналы только с 1h
            if df_1h is not None:
                volume_ratio = df_1h['volume_ratio'].iloc[-1]
                signals['volume'] = 1 if volume_ratio > 1.3 else -1 if volume_ratio < 0.8 else 0

        except Exception as e:
            print(f"Ошибка сбора сигналов: {e}")

        return signals

    def calculate_conservative_confidence(self, signals, trend_analysis):
        """Расчет консервативной уверенности в сделке"""
        if not signals:
            return 0

        # Веса для разных таймфреймов
        weights = {
            '4h': 0.5,  # Наибольший вес для старшего ТФ
            '1h': 0.3,  # Средний вес
            'volume': 0.2  # Меньший вес для объема
        }

        # Считаем согласованность сигналов
        tf_scores = {}
        for tf in ['4h', '1h']:
            tf_signals = [signals.get(f'{tf}_{ind}', 0) for ind in ['sma', 'ema', 'macd', 'rsi']]
            tf_scores[tf] = sum(tf_signals) / len(tf_signals) if tf_signals else 0

        # Итоговый счет с весами
        final_score = (tf_scores.get('4h', 0) * weights['4h'] +
                       tf_scores.get('1h', 0) * weights['1h'] +
                       signals.get('volume', 0) * weights['volume'])

        # Усиление за счет силы тренда
        if trend_analysis['strength'] > 40:  # Только при сильном тренде
            final_score *= 1.2
        elif trend_analysis['strength'] > 25:
            final_score *= 1.1

        return final_score

    def generate_conservative_prediction(self, symbol, multi_data):
        """Генерация консервативного прогноза"""
        if not multi_data or '4h' not in multi_data or '1h' not in multi_data:
            return None

        try:
            df_4h = self.calculate_conservative_indicators(multi_data['4h'])
            df_1h = self.calculate_conservative_indicators(multi_data['1h'])
            df_30m = self.calculate_conservative_indicators(multi_data.get('30m', multi_data['1h']))

            current_price = df_1h['close'].iloc[-1]

            # Консервативный анализ тренда
            trend_analysis = self.analyze_conservative_trend(df_4h, df_1h, df_30m)

            # Сбор сигналов
            signals = self.collect_conservative_signals(df_4h, df_1h, df_30m)

            # Расчет уверенности
            confidence_score = self.calculate_conservative_confidence(signals, trend_analysis)

            # Определение направления (только при высокой уверенности)
            if confidence_score > 0.25 and trend_analysis['trend'] == 'BULLISH':
                direction = "LONG"
                confidence = min(95, 65 + (confidence_score * 25))
            elif confidence_score < -0.25 and trend_analysis['trend'] == 'BEARISH':
                direction = "SHORT"
                confidence = min(95, 65 + (abs(confidence_score) * 25))
            else:
                direction = "NEUTRAL"
                confidence = max(30, abs(confidence_score) * 50)

            # Расчет целевых уровней
            targets = self.calculate_safe_profit_targets(df_1h, direction, current_price)

            # Дополнительные проверки безопасности
            is_safe_trade = (
                    direction != "NEUTRAL" and
                    confidence >= self.min_confidence and
                    targets['profit_ratio'] >= self.min_profit_ratio and
                    targets['risk_percent'] <= 3.0 and  # Максимум 3% риска
                    trend_analysis['rsi_4h'] not in [20, 80] and  # Избегаем экстремумов
                    trend_analysis['rsi_1h'] not in [20, 80]
            )

            # Генерация причины
            reason = self.generate_conservative_reason(direction, signals, trend_analysis, targets, is_safe_trade)

            prediction = {
                'symbol': symbol,
                'direction': direction,
                'confidence': round(confidence, 1),
                'current_price': round(current_price, 6),
                'target': targets['target'],
                'stop_loss': targets['stop_loss'],
                'profit_ratio': targets['profit_ratio'],
                'potential_profit': targets['potential_profit'],
                'risk_percent': targets['risk_percent'],
                'is_safe': is_safe_trade,
                'timeframe': '5 hours',
                'reason': reason,
                'trend_strength': round(trend_analysis['strength'], 1),
                'rsi_4h': round(trend_analysis['rsi_4h'], 1),
                'rsi_1h': round(trend_analysis['rsi_1h'], 1),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            return prediction

        except Exception as e:
            print(f"Ошибка генерации прогноза для {symbol}: {e}")
            return None

    def generate_conservative_reason(self, direction, signals, trend_analysis, targets, is_safe):
        """Генерация обоснования для консервативной сделки"""
        reasons = []

        if direction == "LONG":
            reasons.append("🟢 КОНСЕРВАТИВНЫЙ ЛОНГ:")
            reasons.append("Согласованный бычий тренд на 4h/1h")
        elif direction == "SHORT":
            reasons.append("🔴 КОНСЕРВАТИВНЫЙ ШОРТ:")
            reasons.append("Согласованный медвежий тренд на 4h/1h")
        else:
            reasons.append("⚪ НЕТ ЧЕТКОГО СИГНАЛА:")
            reasons.append("Недостаточная согласованность таймфреймов")

        # Информация о качестве сделки
        if is_safe:
            reasons.append(f"✅ БЕЗОПАСНАЯ СДЕЛКА")
            reasons.append(f"Соотношение {targets['profit_ratio']}:1")
            reasons.append(f"Риск: {targets['risk_percent']}%")
        else:
            reasons.append(f"⚠️  НЕДОСТАТОЧНО КРИТЕРИЕВ")

        reasons.append(f"Тренд: {trend_analysis['strength']}%")

        return " | ".join(reasons)

    def analyze_symbol(self, symbol):
        """Консервативный анализ символа"""
        print(f"🔍 Консервативный анализ {symbol}...")

        try:
            # Получаем данные с консервативных таймфреймов
            multi_data = self.get_conservative_timeframes(symbol)

            # Генерация прогноза
            prediction = self.generate_conservative_prediction(symbol, multi_data)

            if prediction:
                # Проверяем качество сигнала
                if prediction['is_safe']:
                    print(
                        f"   ✅ БЕЗОПАСНЫЙ СИГНАЛ: {prediction['direction']} (уверенность: {prediction['confidence']}%, профит: {prediction['profit_ratio']}:1)")
                else:
                    print(f"   ⚪ НЕДОСТАТОЧНО КРИТЕРИЕВ: {prediction['direction']}")

                return prediction
            else:
                print(f"   ❌ Не удалось сгенерировать прогноз для {symbol}")

        except Exception as e:
            print(f"   ❌ Ошибка анализа {symbol}: {e}")

        return None


def main_conservative_analysis():
    """Основная функция консервативного анализа"""
    print("🛡️  ЗАПУСК КОНСЕРВАТИВНОГО АНАЛИЗА ДЛЯ 5-ЧАСОВЫХ СДЕЛОК")
    print("=" * 80)
    print("🎯 Критерии безопасной сделки:")
    print("   • Уверенность ≥ 70%")
    print("   • Соотношение риск/прибыль ≥ 2.5:1")
    print("   • Риск ≤ 3% от депозита")
    print("   • Согласованный тренд на 4h/1h таймфреймах")
    print("=" * 80)

    predictor = ConservativePricePredictor()

    # Получаем список стабильных монет
    symbols_data = get_high_volume_symbols(min_volume=20000000)  # Больше минимальный объем
    if not symbols_data:
        print("❌ Не удалось получить список монет")
        return []

    # Берем топ-7 самых ликвидных монет
    symbols = [item['symbol'] for item in symbols_data]

    print(f"📊 Анализируем {len(symbols)} самых ликвидных монет...")
    print("=" * 80)

    results = []
    safe_trades = []

    for symbol in symbols:
        prediction = predictor.analyze_symbol(symbol)
        if prediction:
            results.append(prediction)
            if prediction['is_safe']:
                safe_trades.append(prediction)
        time.sleep(1.5)  # Увеличили паузу

    # Сортируем безопасные сделки
    safe_trades.sort(key=lambda x: (x['profit_ratio'], x['confidence']), reverse=True)

    # Вывод результатов
    print(f"\n{'=' * 80}")
    print("🎯 БЕЗОПАСНЫЕ 5-ЧАСОВЫЕ СДЕЛКИ:")
    print(f"{'=' * 80}")

    if safe_trades:
        for i, trade in enumerate(safe_trades, 1):
            direction_icon = "🟢" if trade['direction'] == 'LONG' else "🔴"

            print(f"\n{i}. {direction_icon} {trade['symbol']} 🛡️")
            print(f"   Направление: {trade['direction']}")
            print(f"   Уверенность: {trade['confidence']}%")
            print(f"   Текущая цена: {trade['current_price']}")
            print(f"   Цель: {trade['target']} (+{trade['potential_profit']}%)")
            print(f"   Стоп-лосс: {trade['stop_loss']} (-{trade['risk_percent']}%)")
            print(f"   Соотношение: {trade['profit_ratio']}:1")
            print(f"   RSI 4h/1h: {trade['rsi_4h']}/{trade['rsi_1h']}")
            print(f"   Сила тренда: {trade['trend_strength']}%")
            print(f"   Причина: {trade['reason']}")
    else:
        print("\n❌ Не найдено безопасных сделок, соответствующих критериям")
        # Показываем лучшие из неподходящих
        potential_trades = [r for r in results if not r['is_safe'] and r['direction'] != 'NEUTRAL']
        if potential_trades:
            print(f"\n⚠️  ПОТЕНЦИАЛЬНЫЕ СДЕЛКИ (требуют осторожности):")
            for trade in potential_trades:
                direction_icon = "🟢" if trade['direction'] == 'LONG' else "🔴"
                print(
                    f"   {direction_icon} {trade['symbol']}: уверенность {trade['confidence']}%, профит {trade['profit_ratio']}:1")

    print(f"\n📈 СТАТИСТИКА:")
    print(f"   • Безопасных сделок: {len(safe_trades)}")
    print(f"   • Всего проанализировано: {len(results)}")
    print(f"   • Эффективность: {len(safe_trades) / len(results) * 100:.1f}%" if results else "0%")

    return safe_trades


def test_conservative_strategy():
    """Тестирование консервативной стратегии"""
    try:
        results = main_conservative_analysis()
        success = len(results) > 0
        print(f"\n{'✅' if success else '⚠️'} Тест завершен! Найдено {len(results)} безопасных сделок")
        return success
    except Exception as e:
        print(f"\n❌ Критическая ошибка в тесте: {e}")
        return False


# Запуск консервативного анализа
if __name__ == "__main__":
    test_conservative_strategy()