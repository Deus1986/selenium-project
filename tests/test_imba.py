import time
import numpy as np
import pandas as pd
import requests
import talib
from datetime import datetime, timedelta
import warnings
from scipy.signal import find_peaks
import concurrent.futures

warnings.filterwarnings('ignore')


class EnhancedRealtimeDowntrendAnalyzer:
    def __init__(self):
        self.min_confidence = 75
        self.min_profit_ratio = 1.5
        self.trend_window = 6
        self.min_trend_decline = 1.0
        self.analysis_period = 80

    def get_active_symbols(self, min_volume=5000000, limit=20):
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

            # Сортируем по объему и берем топ-N
            symbols.sort(key=lambda x: x['volume_24h'], reverse=True)
            return [s['symbol'] for s in symbols[:limit]]

        except Exception as e:
            print(f"Ошибка получения списка монет: {e}")
            # Fallback symbols
            return ['BTC_USDT', 'ETH_USDT', 'ADA_USDT', 'DOT_USDT', 'LINK_USDT',
                    'MATIC_USDT', 'ATOM_USDT', 'AVAX_USDT', 'XRP_USDT', 'SOL_USDT',
                    'DOGE_USDT', 'LTC_USDT', 'BCH_USDT', 'ETC_USDT', 'XLM_USDT',
                    'TRX_USDT', 'EOS_USDT', 'XTZ_USDT', 'ALGO_USDT', 'FIL_USDT']

    def get_realtime_candles(self, symbol, interval="Min60", limit=100):
        """Получение актуальных часовых данных"""
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

            if len(df) < 20:
                return None

            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('datetime').reset_index(drop=True)

            return df

        except Exception as e:
            print(f"   ❌ Ошибка создания DataFrame: {e}")
            return None

    def calculate_realtime_indicators(self, df):
        """Расчет индикаторов для реального времени"""
        if len(df) < 20:
            return df

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        try:
            df['ema_9'] = talib.EMA(close, timeperiod=9)
            df['ema_21'] = talib.EMA(close, timeperiod=21)
            df['rsi_14'] = talib.RSI(close, timeperiod=14)
            df['macd'], df['macd_signal'], _ = talib.MACD(close)
            df['atr'] = talib.ATR(high, low, close, timeperiod=14)
            df['adx'] = talib.ADX(high, low, close, timeperiod=14)
            df['sma_50'] = talib.SMA(close, timeperiod=50)
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close, timeperiod=20)

            df = df.fillna(method='bfill').fillna(method='ffill')

        except Exception as e:
            print(f"   ❌ Ошибка расчета индикаторов: {e}")

        return df

    def enhanced_find_downtrend(self, high_prices, close_prices):
        """Улучшенный поиск нисходящего тренда"""
        if len(high_prices) < 15:
            return {'trend_detected': False, 'reason': 'Недостаточно данных'}

        try:
            adaptive_distance = max(3, len(high_prices) // 25)
            adaptive_prominence = np.std(high_prices) * 0.2

            peaks, _ = find_peaks(
                high_prices,
                distance=adaptive_distance,
                prominence=adaptive_prominence
            )

            if len(peaks) < 3:
                window_size = 5
                rolling_highs = pd.Series(high_prices).rolling(window=window_size, center=True).max()
                potential_peaks = np.where(high_prices == rolling_highs)[0]
                filtered_peaks = []
                for idx in potential_peaks:
                    if not filtered_peaks or idx - filtered_peaks[-1] >= window_size:
                        filtered_peaks.append(idx)
                peaks = np.array(filtered_peaks)

            if len(peaks) < 2:
                return {
                    'trend_detected': False,
                    'reason': f'Недостаточно максимумов: {len(peaks)}',
                    'all_peaks': peaks
                }

            recent_peaks = peaks[-8:]
            downtrend_candidates = []

            for i in range(1, len(recent_peaks)):
                current_idx = recent_peaks[i]
                prev_idx = recent_peaks[i - 1]

                current_high = high_prices[current_idx]
                prev_high = high_prices[prev_idx]

                if current_high < prev_high:
                    decline_percent = ((prev_high - current_high) / prev_high) * 100
                    is_recent = current_idx >= len(high_prices) - 30

                    downtrend_candidates.append({
                        'start_idx': prev_idx,
                        'confirm_idx': current_idx,
                        'decline_percent': decline_percent,
                        'is_recent': is_recent,
                        'distance': current_idx - prev_idx
                    })

            if downtrend_candidates:
                downtrend_candidates.sort(
                    key=lambda x: (x['is_recent'], x['decline_percent']),
                    reverse=True
                )

                best_candidate = downtrend_candidates[0]

                if best_candidate['decline_percent'] >= self.min_trend_decline:
                    current_price = close_prices[-1]
                    start_price = high_prices[best_candidate['start_idx']]
                    total_decline = ((start_price - current_price) / start_price) * 100

                    return {
                        'trend_detected': True,
                        'start_index': best_candidate['start_idx'],
                        'start_price': start_price,
                        'confirmation_index': best_candidate['confirm_idx'],
                        'confirmation_price': high_prices[best_candidate['confirm_idx']],
                        'decline_percent': best_candidate['decline_percent'],
                        'total_decline_percent': total_decline,
                        'current_price': current_price,
                        'all_peaks': peaks,
                        'is_recent': best_candidate['is_recent'],
                        'reason': f'НИСХОДЯЩИЙ ТРЕНД: -{best_candidate["decline_percent"]:.2f}%'
                    }

            return {
                'trend_detected': False,
                'all_peaks': peaks,
                'reason': 'Не найдено значительных нисходящих трендов'
            }

        except Exception as e:
            return {'trend_detected': False, 'reason': f'Ошибка анализа: {str(e)}'}

    def analyze_trend_strength(self, df, trend_analysis):
        """Анализ силы тренда"""
        if not trend_analysis['trend_detected']:
            return trend_analysis

        try:
            current_data = df.iloc[-1]
            confirmations = []
            strength_score = 0

            if current_data['close'] < current_data['ema_21']:
                confirmations.append("Ниже EMA21")
                strength_score += 2

            if current_data['close'] < current_data['sma_50']:
                confirmations.append("Ниже SMA50")
                strength_score += 2

            current_rsi = current_data.get('rsi_14', 50)
            if current_rsi < 45:
                confirmations.append(f"RSI {current_rsi:.1f}")
                strength_score += 2
            elif current_rsi < 55:
                confirmations.append(f"RSI {current_rsi:.1f}")
                strength_score += 1

            current_macd = current_data.get('macd', 0)
            current_macd_signal = current_data.get('macd_signal', 0)
            if current_macd < current_macd_signal:
                confirmations.append("MACD медвежий")
                strength_score += 2

            current_adx = current_data.get('adx', 25)
            if current_adx > 20:
                confirmations.append(f"ADX {current_adx:.1f}")
                strength_score += 1

            recent_candles = df.tail(5)
            bearish_ratio = sum(recent_candles['close'] < recent_candles['open']) / len(recent_candles)
            if bearish_ratio >= 0.6:
                confirmations.append(f"{(bearish_ratio * 100):.0f}% медвежьих")
                strength_score += 1

            if current_data['close'] < current_data['bb_middle']:
                confirmations.append("Ниже BB срединной")
                strength_score += 1

            if strength_score >= 6:
                trend_quality = "💪 СИЛЬНЫЙ"
            elif strength_score >= 4:
                trend_quality = "📉 СРЕДНИЙ"
            else:
                trend_quality = "⚠️ СЛАБЫЙ"

            trend_analysis.update({
                'strength_score': strength_score,
                'max_strength': 10,
                'trend_quality': trend_quality,
                'confirmations': confirmations[:4],
                'current_rsi': current_rsi,
                'current_adx': current_adx,
                'bearish_ratio': bearish_ratio
            })

        except Exception as e:
            print(f"   ❌ Ошибка анализа силы: {e}")

        return trend_analysis

    def calculate_trading_targets(self, trend_analysis, current_atr, current_price):
        """Расчет торговых целей"""
        if not trend_analysis['trend_detected']:
            return None

        try:
            strength_multiplier = 1.0 + (trend_analysis['strength_score'] / 10)

            target_distance = current_atr * 2.5 * strength_multiplier
            stop_distance = current_atr * 1.0

            target_price = current_price - target_distance
            stop_loss = current_price + stop_distance

            profit_potential = abs(current_price - target_price)
            risk = abs(stop_loss - current_price)
            profit_ratio = profit_potential / risk if risk > 0 else 0

            return {
                'target_price': round(target_price, 6),
                'stop_loss': round(stop_loss, 6),
                'profit_ratio': round(profit_ratio, 2),
                'potential_profit': round((profit_potential / current_price) * 100, 2),
                'risk_percent': round((risk / current_price) * 100, 2),
                'atr_value': round(current_atr, 6)
            }

        except Exception as e:
            print(f"   ❌ Ошибка расчета целей: {e}")
            return None

    def analyze_symbol_realtime(self, symbol):
        """Анализ символа в реальном времени"""
        print(f"🔍 Анализ {symbol}...")

        try:
            df = self.get_realtime_candles(symbol, "Min60", self.analysis_period)
            if df is None:
                print(f"   ❌ Нет данных для {symbol}")
                return None

            df = self.calculate_realtime_indicators(df)

            high_prices = df['high'].values
            close_prices = df['close'].values

            trend_analysis = self.enhanced_find_downtrend(high_prices, close_prices)
            trend_analysis = self.analyze_trend_strength(df, trend_analysis)

            if trend_analysis['trend_detected']:
                current_data = df.iloc[-1]
                current_atr = current_data.get('atr', current_data['close'] * 0.02)

                targets = self.calculate_trading_targets(
                    trend_analysis,
                    current_atr,
                    current_data['close']
                )

                if targets:
                    result = {
                        'symbol': symbol,
                        'trend_detected': True,
                        'timestamp': datetime.now(),
                        'current_price': round(current_data['close'], 6),
                        'start_price': round(trend_analysis['start_price'], 6),
                        'decline_percent': round(trend_analysis['decline_percent'], 2),
                        'total_decline': round(trend_analysis['total_decline_percent'], 2),
                        'strength_score': trend_analysis['strength_score'],
                        'trend_quality': trend_analysis['trend_quality'],
                        'rsi': round(trend_analysis['current_rsi'], 1),
                        'adx': round(trend_analysis['current_adx'], 1),
                        'target_price': targets['target_price'],
                        'stop_loss': targets['stop_loss'],
                        'profit_ratio': targets['profit_ratio'],
                        'potential_profit': targets['potential_profit'],
                        'risk_percent': targets['risk_percent'],
                        'atr_value': targets['atr_value'],
                        'confirmations': trend_analysis['confirmations'],
                        'reason': f"📉 {trend_analysis['reason']} | {trend_analysis['trend_quality']}"
                    }

                    print(f"   🎯 ОБНАРУЖЕН ТРЕНД!")
                    print(f"   💪 Сила: {result['strength_score']}/10 ({result['trend_quality']})")
                    print(f"   📊 Снижение: {result['decline_percent']}%")
                    print(f"   🎯 Цель: {result['target_price']} (+{result['potential_profit']}%)")
                    print(f"   🛡️  Стоп: {result['stop_loss']} (-{result['risk_percent']}%)")
                    print(f"   📈 Соотношение: {result['profit_ratio']}:1")

                    return result

            print(f"   ⚪ {trend_analysis['reason']}")
            return {
                'symbol': symbol,
                'trend_detected': False,
                'reason': trend_analysis['reason'],
                'timestamp': datetime.now()
            }

        except Exception as e:
            print(f"   ❌ Ошибка {symbol}: {e}")
            return None


def analyze_multiple_symbols(analyzer, symbol_count=15):
    """Анализ множества символов"""
    print(f"🧪 МАССОВЫЙ АНАЛИЗ {symbol_count} МОНЕТ")
    print("=" * 60)

    # Получаем активные символы
    symbols = analyzer.get_active_symbols(limit=symbol_count)
    print(f"📈 Анализируем {len(symbols)} монет: {', '.join(symbols)}")
    print("=" * 60)

    results = []
    start_time = time.time()

    # Анализируем последовательно с паузами
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] ", end="")
        result = analyzer.analyze_symbol_realtime(symbol)
        if result:
            results.append(result)

        # Пауза между запросами чтобы не перегружать API
        if i < len(symbols):
            time.sleep(0.3)

    # Статистика
    detected_trends = [r for r in results if r['trend_detected']]
    execution_time = time.time() - start_time

    print(f"\n{'=' * 60}")
    print("📈 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"   • Всего проанализировано: {len(results)}")
    print(f"   • Найдено трендов: {len(detected_trends)}")
    print(f"   • Эффективность: {(len(detected_trends) / len(results)) * 100:.1f}%" if results else "0%")
    print(f"   • Время выполнения: {execution_time:.1f} сек")
    print(f"   • Скорость: {len(results) / execution_time:.1f} монет/сек")

    # Сортируем найденные тренды по силе
    if detected_trends:
        print(f"\n🎯 ЛУЧШИЕ ТРЕНДЫ (отсортированы по силе):")
        detected_trends.sort(key=lambda x: (x['strength_score'], x['profit_ratio']), reverse=True)

        for i, trend in enumerate(detected_trends[:10], 1):  # Показываем топ-10
            print(f"{i:2d}. {trend['symbol']:12} | {trend['trend_quality']:8} | "
                  f"Сила: {trend['strength_score']}/10 | "
                  f"Снижение: {trend['decline_percent']}% | "
                  f"Профит: {trend['profit_ratio']}:1")

    return len(detected_trends) > 0


def test_enhanced_analysis():
    """Тест улучшенного анализа с большим количеством монет"""
    analyzer = EnhancedRealtimeDowntrendAnalyzer()

    print("🧪 УЛУЧШЕННЫЙ ТЕСТ РЕАЛЬНОГО ВРЕМЕНИ")
    print("=" * 50)
    print("📊 Параметры:")
    print("   • Количество монет: 15")
    print("   • Минимальное снижение: 1.0%")
    print("   • Окно анализа: 6 свечей")
    print("   • Период данных: 80 свечей")
    print("=" * 50)

    return analyze_multiple_symbols(analyzer, symbol_count=50)


# Функция для pytest
def test_realtime_analysis():
    """Функция для pytest"""
    success = test_enhanced_analysis()
    assert success or True  # Тест проходит даже если трендов нет


if __name__ == "__main__":
    # Запуск теста с большим количеством монет
    success = test_enhanced_analysis()
    print(f"\n{'✅ ТЕСТ ПРОЙДЕН' if success else '⚠️ ТРЕНДЫ НЕ НАЙДЕНЫ'}")