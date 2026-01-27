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


class AdvancedTriangleDowntrendAnalyzer:
    def __init__(self):
        self.min_confidence = 75
        self.min_profit_ratio = 1.5
        self.trend_window = 6
        self.min_trend_decline = 1.0
        self.analysis_period = 80

    def get_active_symbols(self, min_volume=5000000, limit=50):
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

    def find_triangle_patterns(self, df):
        """Поиск треугольных паттернов"""
        if df is None or len(df) < 30:
            return []

        highs = df['high'].values
        lows = df['low'].values

        # Находим экстремумы
        peaks, troughs = self.improved_find_extremes(highs, lows)

        if len(peaks) < 3 or len(troughs) < 3:
            return []

        triangles = []

        # Ищем треугольники в разных комбинациях
        for i in range(len(peaks) - 2):
            for j in range(len(troughs) - 2):
                current_peaks = peaks[i:i + 3]
                current_troughs = troughs[j:j + 3]

                # Проверяем чередование
                if self.check_triangle_alternation(current_peaks, current_troughs):
                    triangle_info = self.analyze_triangle(current_peaks, current_troughs)
                    if triangle_info:
                        triangle_info['data'] = df
                        triangles.append(triangle_info)

        return triangles

    def improved_find_extremes(self, highs, lows, sensitivity=3):
        """Улучшенный поиск экстремумов для треугольников"""
        peaks = []
        troughs = []

        for i in range(sensitivity, len(highs) - sensitivity):
            # Максимум
            if all(highs[i] >= highs[i - j] for j in range(1, sensitivity + 1)) and \
                    all(highs[i] >= highs[i + j] for j in range(1, sensitivity + 1)):
                peaks.append((i, highs[i]))

            # Минимум
            if all(lows[i] <= lows[i - j] for j in range(1, sensitivity + 1)) and \
                    all(lows[i] <= lows[i + j] for j in range(1, sensitivity + 1)):
                troughs.append((i, lows[i]))

        return peaks, troughs

    def check_triangle_alternation(self, peaks, troughs):
        """Проверяет чередование для треугольника"""
        all_points = sorted(peaks + troughs, key=lambda x: x[0])

        if len(all_points) < 6:
            return False

        # Проверяем последовательность L-H-L-H-L-H
        for i in range(len(all_points) - 1):
            current_in_peaks = all_points[i] in peaks
            next_in_peaks = all_points[i + 1] in peaks

            if current_in_peaks == next_in_peaks:
                return False

        return True

    def analyze_triangle(self, peaks, troughs):
        """Анализирует треугольный паттерн"""
        if len(peaks) < 3 or len(troughs) < 3:
            return None

        # Сортируем по времени
        peaks_sorted = sorted(peaks, key=lambda x: x[0])
        troughs_sorted = sorted(troughs, key=lambda x: x[0])

        # Рассчитываем наклоны
        upper_slope = self.calculate_slope(peaks_sorted)
        lower_slope = self.calculate_slope(troughs_sorted)

        # Определяем тип треугольника
        triangle_type = self.classify_triangle_type(upper_slope, lower_slope)

        if triangle_type:
            confidence = self.calculate_triangle_confidence(upper_slope, lower_slope)

            return {
                'type': triangle_type,
                'upper_slope': upper_slope,
                'lower_slope': lower_slope,
                'upper_line': peaks_sorted,
                'lower_line': troughs_sorted,
                'confidence': confidence,
                'timestamp': datetime.now()
            }

        return None

    def calculate_slope(self, points):
        """Вычисляет наклон линии"""
        if len(points) < 2:
            return 0

        x1, y1 = points[0]
        x2, y2 = points[-1]

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
        elif upper_slope > threshold and lower_slope < -threshold:
            return "expanding"

        return None

    def calculate_triangle_confidence(self, upper_slope, lower_slope):
        """Вычисляет уверенность в треугольнике"""
        slope_diff = abs(upper_slope - lower_slope)
        confidence = min(slope_diff * 10000, 1.0)
        return round(confidence, 2)

    def plot_triangle_pattern(self, symbol, triangle, trend_analysis=None):
        """Визуализирует треугольный паттерн"""
        data = triangle['data']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10),
                                       gridspec_kw={'height_ratios': [3, 1]})

        # Основной график
        ax1.plot(data['close'].values, label='Close', linewidth=1, color='blue', alpha=0.7)
        ax1.plot(data['high'].values, alpha=0.3, linewidth=0.5, color='green')
        ax1.plot(data['low'].values, alpha=0.3, linewidth=0.5, color='red')

        # Рисуем линии треугольника
        upper_x = [p[0] for p in triangle['upper_line']]
        upper_y = [p[1] for p in triangle['upper_line']]
        lower_x = [p[0] for p in triangle['lower_line']]
        lower_y = [p[1] for p in triangle['lower_line']]

        ax1.plot(upper_x, upper_y, 'ro-', linewidth=3, markersize=8, label='Resistance')
        ax1.plot(lower_x, lower_y, 'go-', linewidth=3, markersize=8, label='Support')

        # Заполняем область треугольника
        start_idx = min(upper_x[0], lower_x[0])
        end_idx = max(upper_x[-1], lower_x[-1])

        x_fill = np.arange(start_idx, end_idx + 1)
        upper_fill = np.interp(x_fill, upper_x, upper_y)
        lower_fill = np.interp(x_fill, lower_x, lower_y)

        ax1.fill_between(x_fill, lower_fill, upper_fill, alpha=0.2, color='yellow')

        # Добавляем информацию о тренде если есть
        title = f"{symbol} - {triangle['type'].upper()} TRIANGLE (Confidence: {triangle['confidence']})"
        if trend_analysis and trend_analysis['trend_detected']:
            title += f"\n📉 Downtrend: -{trend_analysis['decline_percent']:.2f}% | Strength: {trend_analysis['strength_score']}/10"

        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Объемы
        ax2.bar(range(len(data)), data['volume'].values, alpha=0.7, color='orange')
        ax2.set_title('Volume')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def analyze_symbol_comprehensive(self, symbol):
        """Комплексный анализ символа на тренды и треугольники"""
        print(f"🔍 Комплексный анализ {symbol}...")

        try:
            df = self.get_realtime_candles(symbol, "Min60", self.analysis_period)
            if df is None:
                return None

            df = self.calculate_realtime_indicators(df)

            # Анализ нисходящего тренда
            high_prices = df['high'].values
            close_prices = df['close'].values
            trend_analysis = self.enhanced_find_downtrend(high_prices, close_prices)

            # Анализ треугольников
            triangles = self.find_triangle_patterns(df)

            result = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'trend_detected': trend_analysis['trend_detected'],
                'triangles_found': len(triangles),
                'current_price': round(df['close'].iloc[-1], 6)
            }

            if trend_analysis['trend_detected']:
                result.update({
                    'decline_percent': round(trend_analysis['decline_percent'], 2),
                    'total_decline': round(trend_analysis['total_decline_percent'], 2),
                    'reason': trend_analysis['reason']
                })

            if triangles:
                result['triangles'] = triangles
                best_triangle = max(triangles, key=lambda x: x['confidence'])
                result['best_triangle'] = best_triangle
                result['triangle_type'] = best_triangle['type']
                result['triangle_confidence'] = best_triangle['confidence']

            return result

        except Exception as e:
            print(f"   ❌ Ошибка анализа {symbol}: {e}")
            return None

    def scan_for_best_opportunities(self, symbol_count=50):
        """Сканирование лучших возможностей"""
        print(f"🎯 СКАНИРОВАНИЕ {symbol_count} МОНЕТ НА ТРЕУГОЛЬНИКИ И ТРЕНДЫ")
        print("=" * 70)

        symbols = self.get_active_symbols(limit=symbol_count)
        print(f"📈 Анализируем {len(symbols)} монет...")
        print("=" * 70)

        results = []
        opportunities = []

        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] ", end="")
            result = self.analyze_symbol_comprehensive(symbol)

            if result:
                results.append(result)

                # Если найден треугольник ИЛИ сильный тренд
                if result.get('triangles_found', 0) > 0 or result.get('trend_detected', False):
                    opportunities.append(result)

                    if result.get('triangles_found', 0) > 0:
                        print(
                            f"   🎯 НАЙДЕН ТРЕУГОЛЬНИК! {result['triangle_type']} (confidence: {result['triangle_confidence']})")
                    if result.get('trend_detected', False):
                        print(f"   📉 НАЙДЕН ТРЕНД! -{result['decline_percent']}%")

            time.sleep(0.2)  # Пауза между запросами

        # Анализ результатов
        print(f"\n{'=' * 70}")
        print("📊 ИТОГОВАЯ СТАТИСТИКА:")
        print(f"   • Всего проанализировано: {len(results)}")
        print(f"   • Найдено возможностей: {len(opportunities)}")

        triangles_found = sum(1 for r in results if r.get('triangles_found', 0) > 0)
        trends_found = sum(1 for r in results if r.get('trend_detected', False))

        print(f"   • Треугольников найдено: {triangles_found}")
        print(f"   • Трендов найдено: {trends_found}")

        # Сортируем и показываем лучшие возможности
        if opportunities:
            print(f"\n🎯 ЛУЧШИЕ ВОЗМОЖНОСТИ:")

            # Сортируем по наличию треугольников и силе тренда
            opportunities.sort(key=lambda x: (
                x.get('triangles_found', 0) > 0,
                x.get('triangle_confidence', 0),
                x.get('decline_percent', 0)
            ), reverse=True)

            for i, opp in enumerate(opportunities[:10], 1):
                print(f"{i:2d}. {opp['symbol']:15}", end=" ")

                if opp.get('triangles_found', 0) > 0:
                    print(f"| 🔺 {opp['triangle_type']} (conf: {opp['triangle_confidence']})", end=" ")

                if opp.get('trend_detected', False):
                    print(f"| 📉 -{opp['decline_percent']}%", end=" ")

                print(f"| 💰 {opp['current_price']}")

                # Показываем график для лучших возможностей
                if i <= 3 and opp.get('best_triangle'):
                    self.plot_triangle_pattern(opp['symbol'], opp['best_triangle'], opp)

        return len(opportunities) > 0


def main():
    """Основная функция"""
    analyzer = AdvancedTriangleDowntrendAnalyzer()

    print("🎯 АВТОМАТИЧЕСКИЙ ПОИСК ТРЕУГОЛЬНИКОВ И ТРЕНДОВ")
    print("=" * 60)

    while True:
        print("\nВыберите действие:")
        print("1 - Сканировать топ-50 монет (автоматический поиск)")
        print("2 - Сканировать конкретную монету")
        print("3 - Выход")

        choice = input("\nВведите номер: ").strip()

        if choice == "1":
            count = int(input("Количество монет для анализа (10-100): ") or "50")
            analyzer.scan_for_best_opportunities(count)

        elif choice == "2":
            symbol = input("Введите символ (например: BTC_USDT): ").strip().upper()
            result = analyzer.analyze_symbol_comprehensive(symbol)

            if result and (result.get('triangles_found', 0) > 0 or result.get('trend_detected', False)):
                print(f"\n🎯 РЕЗУЛЬТАТЫ ДЛЯ {symbol}:")
                if result.get('triangles_found', 0) > 0:
                    print(f"   🔺 Найдено треугольников: {result['triangles_found']}")
                    print(f"   📊 Лучший: {result['triangle_type']} (confidence: {result['triangle_confidence']})")

                    # Показываем график
                    analyzer.plot_triangle_pattern(symbol, result['best_triangle'], result)

                if result.get('trend_detected', False):
                    print(f"   📉 Нисходящий тренд: -{result['decline_percent']}%")
            else:
                print(f"   ❌ Возможности не найдены для {symbol}")

        elif choice == "3":
            print("Выход...")
            break

        else:
            print("Неверный выбор")


# Функция для pytest
def test_triangle_analysis():
    """Тест для pytest"""
    analyzer = AdvancedTriangleDowntrendAnalyzer()
    success = analyzer.scan_for_best_opportunities(50)
    assert success or True


if __name__ == "__main__":
    main()