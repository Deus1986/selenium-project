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


class OverboughtScanner:
    def __init__(self, rsi_threshold=90, stoch_threshold=90):
        self.rsi_threshold = rsi_threshold
        self.stoch_threshold = stoch_threshold

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
            return None

        raw_data = data['data']

        try:
            if isinstance(raw_data, dict):
                required_fields = ['time', 'open', 'close', 'high', 'low', 'vol']

                missing_fields = [field for field in required_fields if field not in raw_data]
                if missing_fields:
                    return None

                df = pd.DataFrame({
                    'timestamp': raw_data['time'],
                    'open': raw_data['open'],
                    'high': raw_data['high'],
                    'low': raw_data['low'],
                    'close': raw_data['close'],
                    'volume': raw_data['vol']
                })

                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                df = df.dropna()

                if len(df) < 50:
                    return None

                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('datetime').reset_index(drop=True)

                return df
            return None

        except Exception as e:
            print(f"Ошибка создания DataFrame: {e}")
            return None

    def calculate_indicators(self, df):
        """Расчет индикаторов перекупленности"""
        if len(df) < 20:
            return df

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        try:
            # RSI - основной индикатор перекупленности
            df['rsi_14'] = talib.RSI(close, timeperiod=14)

            # Stochastic
            stoch_result = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            df['stoch_k'] = stoch_result[0]
            df['stoch_d'] = stoch_result[1]

            # MACD для дополнительной информации
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)

            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close, timeperiod=20)

            # Процент от верхней полосы Боллинджера
            df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100

            # Заполняем NaN значения
            df = df.fillna(method='bfill').fillna(method='ffill')

        except Exception as e:
            print(f"Ошибка расчета индикаторов: {e}")

        return df

    def analyze_overbought_conditions(self, df, symbol):
        """Анализ условий перекупленности"""
        if len(df) < 20:
            return None

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # Проверяем наличие необходимых индикаторов
        required_indicators = ['rsi_14', 'stoch_k', 'stoch_d', 'bb_position']
        for indicator in required_indicators:
            if indicator not in df.columns:
                return None

        # Анализ перекупленности
        overbought_signals = []

        # RSI перекупленность
        rsi_overbought = current['rsi_14'] > self.rsi_threshold
        if rsi_overbought:
            overbought_signals.append(f"RSI: {current['rsi_14']:.1f}")

        # Stochastic перекупленность
        stoch_overbought = current['stoch_k'] > self.stoch_threshold
        if stoch_overbought:
            overbought_signals.append(f"Stoch: {current['stoch_k']:.1f}")

        # Bollinger Bands перекупленность (цена близко к верхней полосе)
        bb_overbought = current['bb_position'] > 80
        if bb_overbought:
            overbought_signals.append(f"BB: {current['bb_position']:.1f}%")

        # MACD замедление (возможный разворот)
        macd_slowing = current['macd'] < prev['macd'] and current['macd'] > current['macd_signal']

        # Сила перекупленности
        overbought_strength = len(overbought_signals)

        # Определяем уровень опасности
        if overbought_strength >= 2:
            danger_level = "🔴 ВЫСОКАЯ"
        elif overbought_strength == 1:
            danger_level = "🟡 СРЕДНЯЯ"
        else:
            danger_level = "🟢 НИЗКАЯ"

        analysis = {
            'symbol': symbol,
            'current_price': round(current['close'], 6),
            'rsi': round(current['rsi_14'], 2),
            'stoch_k': round(current['stoch_k'], 2),
            'stoch_d': round(current['stoch_d'], 2),
            'bb_position': round(current['bb_position'], 2),
            'macd': round(current['macd'], 6),
            'macd_hist': round(current['macd_hist'], 6),
            'is_overbought': overbought_strength > 0,
            'overbought_signals': overbought_signals,
            'overbought_strength': overbought_strength,
            'danger_level': danger_level,
            'macd_slowing': macd_slowing,
            'price_change_1h': round((current['close'] - prev['close']) / prev['close'] * 100, 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return analysis

    def scan_symbol(self, symbol):
        """Сканирование символа на перекупленность"""
        try:
            # Получаем данные
            data = self.get_candles(symbol, "Min60", 100)
            if not data:
                return None

            df = self.create_dataframe(data)
            if df is None:
                return None

            # Расчет индикаторов
            df = self.calculate_indicators(df)

            # Анализ перекупленности
            analysis = self.analyze_overbought_conditions(df, symbol)

            return analysis

        except Exception as e:
            print(f"Ошибка анализа {symbol}: {e}")
            return None


def main_overbought_scanner():
    """Основная функция сканера перекупленности"""
    print("🎯 СКАНЕР ПЕРЕКУПЛЕННОСТИ - RSI > 90%")
    print("=" * 80)
    print("📊 Критерии перекупленности:")
    print("   • RSI > 90% - сильная перекупленность")
    print("   • Stochastic K > 90% - моментум перекупленности")
    print("   • Bollinger Bands > 80% - цена у верхней границы")
    print("=" * 80)

    scanner = OverboughtScanner(rsi_threshold=90, stoch_threshold=90)

    # Получаем список монет
    symbols_data = get_high_volume_symbols(min_volume=20000000)
    if not symbols_data:
        print("❌ Не удалось получить список монет")
        return []

    # Берем топ-50 самых ликвидных монет
    symbols = [item['symbol'] for item in symbols_data[:50]]

    print(f"🔍 Сканируем {len(symbols)} монет на перекупленность...")
    print("=" * 80)

    overbought_symbols = []
    all_results = []

    for i, symbol in enumerate(symbols, 1):
        print(f"📊 Анализ {i}/{len(symbols)}: {symbol}...")

        analysis = scanner.scan_symbol(symbol)
        if analysis:
            all_results.append(analysis)

            if analysis['is_overbought']:
                overbought_symbols.append(analysis)
                print(f"   ✅ ПЕРЕКУПЛЕННОСТЬ: {analysis['danger_level']}")
                print(f"      RSI: {analysis['rsi']}, Stoch: {analysis['stoch_k']}")
            else:
                print(f"   ⚪ Норма: RSI {analysis['rsi']}")

        time.sleep(0.3)  # Пауза между запросами

    # Сортируем по силе перекупленности
    overbought_symbols.sort(key=lambda x: (x['overbought_strength'], x['rsi']), reverse=True)

    # Вывод результатов
    print(f"\n{'=' * 80}")
    print("🎯 НАЙДЕНЫ ПЕРЕКУПЛЕННЫЕ АКТИВЫ:")
    print(f"{'=' * 80}")

    if overbought_symbols:
        for i, asset in enumerate(overbought_symbols, 1):
            print(f"\n{i}. {asset['danger_level']} {asset['symbol']}")
            print(f"   💰 Цена: {asset['current_price']}")
            print(f"   📈 RSI: {asset['rsi']} (порог: 90)")
            print(f"   🎯 Stoch K: {asset['stoch_k']} (порог: 90)")
            print(f"   📊 BB позиция: {asset['bb_position']}%")
            print(f"   🔍 Сигналы: {', '.join(asset['overbought_signals'])}")
            print(f"   📉 Изменение за час: {asset['price_change_1h']}%")

            # Рекомендации
            if asset['overbought_strength'] >= 2:
                print(f"   ⚠️  РЕКОМЕНДАЦИЯ: Сильная перекупленность - возможен разворот")
                print(f"      💡 Рассмотреть SHORT позицию с тейк-профитом 2-3%")
            else:
                print(f"   💡 РЕКОМЕНДАЦИЯ: Умеренная перекупленность - наблюдать")

            if asset['macd_slowing']:
                print(f"   📉 MACD замедляется - подтверждение возможного разворота")

    else:
        print("\n✅ Перекупленных активов не найдено")

        # Показываем ближайшие к перекупленности
        near_overbought = [r for r in all_results if r['rsi'] > 70]
        if near_overbought:
            near_overbought.sort(key=lambda x: x['rsi'], reverse=True)
            print(f"\n⚠️  БЛИЗКО К ПЕРЕКУПЛЕННОСТИ (RSI > 70):")
            for asset in near_overbought:
                print(f"   {asset['symbol']}: RSI {asset['rsi']}")

    print(f"\n📈 СТАТИСТИКА СКАНЕРА:")
    print(f"   • Перекупленных активов: {len(overbought_symbols)}")
    print(f"   • Всего проанализировано: {len(all_results)}")
    print(
        f"   • Эффективность сканирования: {len(overbought_symbols) / len(all_results) * 100:.1f}%" if all_results else "0%")

    if overbought_symbols:
        avg_rsi = sum(asset['rsi'] for asset in overbought_symbols) / len(overbought_symbols)
        max_rsi = max(asset['rsi'] for asset in overbought_symbols)
        print(f"   • Средний RSI перекупленных: {avg_rsi:.1f}")
        print(f"   • Максимальный RSI: {max_rsi:.1f}")

    return overbought_symbols


def test_overbought_scanner():
    """Тестирование сканера перекупленности"""
    try:
        results = main_overbought_scanner()
        print(f"\n{'✅' if results else '⚠️'} Сканирование завершено! Найдено {len(results)} перекупленных активов")
        return len(results) > 0
    except Exception as e:
        print(f"\n❌ Критическая ошибка в сканере: {e}")
        return False


# Дополнительная функция для мониторинга в реальном времени
def continuous_monitoring(interval_minutes=5):
    """Непрерывный мониторинг перекупленности"""
    print("🔄 ЗАПУСК НЕПРЕРЫВНОГО МОНИТОРИНГА ПЕРЕКУПЛЕННОСТИ")
    print(f"📊 Интервал проверки: {interval_minutes} минут")
    print("=" * 80)

    while True:
        print(f"\n🕒 Проверка: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        results = main_overbought_scanner()

        if results:
            print(f"\n🚨 ОБНАРУЖЕНЫ ПЕРЕКУПЛЕННЫЕ АКТИВЫ!")
            for asset in results:
                print(f"   {asset['symbol']} - RSI: {asset['rsi']}")

        print(f"\n⏳ Ожидание {interval_minutes} минут до следующей проверки...")
        time.sleep(interval_minutes * 60)


# Запуск сканера
if __name__ == "__main__":
    print("Выберите режим работы:")
    print("1 - Однократное сканирование")
    print("2 - Непрерывный мониторинг")

    choice = input("Введите номер режима (1 или 2): ").strip()

    if choice == "2":
        interval = int(input("Введите интервал проверки в минутах (по умолчанию 5): ") or "5")
        continuous_monitoring(interval)
    else:
        test_overbought_scanner()