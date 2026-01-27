import time
import pandas as pd
import requests
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


def get_high_volume_symbols(min_volume=15000000):
    """Получает список монет с высоким объемом"""
    url = "https://contract.mexc.com/api/v1/contract/ticker"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        symbols = []

        if 'data' in data:
            for item in data["data"]:
                if item["amount24"] > min_volume and item['symbol'].endswith('_USDT'):
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


class VolumeSpikeScanner:
    def __init__(self):
        self.min_volume = 20000000
        self.timeframe = "Min60"
        self.volume_threshold = 3.0
        self.candles_to_analyze = 3  # Анализируем последние 3 свечи

    def get_candles_simple(self, symbol, interval="Min60", limit=50):
        """Упрощенное получение данных свечей"""
        url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
        params = {"interval": interval, "limit": limit}

        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                return None
        except Exception as e:
            return None

    def create_dataframe_from_dict(self, data):
        """Создание DataFrame из словарного формата MEXC"""
        if not data or not data.get('data'):
            return None

        raw_data = data['data']

        try:
            # Если данные в формате словаря
            if isinstance(raw_data, dict):
                # Проверяем наличие необходимых ключей
                required_keys = ['time', 'open', 'high', 'low', 'close', 'vol']
                if not all(key in raw_data for key in required_keys):
                    return None

                # Создаем DataFrame из словаря
                df_data = []
                for i in range(len(raw_data['time'])):
                    df_data.append({
                        'timestamp': raw_data['time'][i],
                        'open': float(raw_data['open'][i]),
                        'high': float(raw_data['high'][i]),
                        'low': float(raw_data['low'][i]),
                        'close': float(raw_data['close'][i]),
                        'volume': float(raw_data['vol'][i])
                    })

                df = pd.DataFrame(df_data)
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('datetime').reset_index(drop=True)

                return df

            # Если данные в формате списка (старый формат)
            elif isinstance(raw_data, list) and len(raw_data) > 0:
                rows = []
                for row in raw_data:
                    if isinstance(row, list) and len(row) >= 6:
                        rows.append({
                            'timestamp': row[0],
                            'open': float(row[1]),
                            'high': float(row[2]),
                            'low': float(row[3]),
                            'close': float(row[4]),
                            'volume': float(row[5])
                        })

                if rows:
                    df = pd.DataFrame(rows)
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.sort_values('datetime').reset_index(drop=True)
                    return df

            return None

        except Exception as e:
            return None

    def analyze_volume_spike(self, df):
        """Анализ всплесков объема в последних 3 свечах"""
        try:
            if len(df) < 20:
                return None

            # Берем только последние 3 свечи для анализа
            recent_data = df.tail(self.candles_to_analyze)

            # Средний объем за предыдущие 20 свечей (исключая последние 3)
            historical_data = df.head(-self.candles_to_analyze)
            if len(historical_data) < 10:
                historical_data = df.head(-self.candles_to_analyze)

            avg_volume = historical_data['volume'].tail(20).mean()

            if avg_volume == 0:
                return None

            results = []

            # Анализируем каждую из последних 3 свечей
            for i in range(len(recent_data)):
                candle = recent_data.iloc[i]
                volume_ratio = candle['volume'] / avg_volume

                # Если объем превышает порог
                if volume_ratio >= self.volume_threshold:
                    # Определяем тип свечи
                    if candle['close'] > candle['open']:
                        candle_type = "🟢 БЫЧЬЯ"
                        trend = "BULLISH"
                    elif candle['close'] < candle['open']:
                        candle_type = "🔴 МЕДВЕЖЬЯ"
                        trend = "BEARISH"
                    else:
                        candle_type = "⚪ ДОДЖ"
                        trend = "NEUTRAL"

                    # Размер тела свечи
                    body_size = abs(candle['close'] - candle['open'])
                    body_percent = (body_size / candle['open']) * 100 if candle['open'] > 0 else 0

                    # Вычисляем тени
                    upper_shadow = candle['high'] - max(candle['open'], candle['close'])
                    lower_shadow = min(candle['open'], candle['close']) - candle['low']
                    total_range = candle['high'] - candle['low']

                    upper_shadow_percent = (upper_shadow / total_range * 100) if total_range > 0 else 0
                    lower_shadow_percent = (lower_shadow / total_range * 100) if total_range > 0 else 0

                    candle_info = {
                        'datetime': candle['datetime'],
                        'open': candle['open'],
                        'high': candle['high'],
                        'low': candle['low'],
                        'close': candle['close'],
                        'volume': candle['volume'],
                        'volume_ratio': round(volume_ratio, 2),
                        'candle_type': candle_type,
                        'trend': trend,
                        'body_percent': round(body_percent, 2),
                        'upper_shadow_percent': round(upper_shadow_percent, 1),
                        'lower_shadow_percent': round(lower_shadow_percent, 1),
                        'age_hours': len(recent_data) - i - 1,  # 0 = текущая, 1 = предыдущая, 2 = позапрошлая
                        'is_current': (i == len(recent_data) - 1)  # Является ли текущей свечой
                    }

                    results.append(candle_info)

            return results if results else None

        except Exception as e:
            return None

    def analyze_symbol(self, symbol):
        """Анализ символа на всплески объема в последних 3 свечах"""
        print(f"🔍 Анализ {symbol}...", end=" ")

        try:
            data = self.get_candles_simple(symbol, self.timeframe, 50)

            if not data:
                print("❌ Нет данных")
                return None

            # Проверяем код ответа
            if data.get('code') != 0:
                print("❌ Ошибка API")
                return None

            df = self.create_dataframe_from_dict(data)
            if df is None:
                print("❌ Не удалось создать DataFrame")
                return None

            volume_spikes = self.analyze_volume_spike(df)

            if volume_spikes:
                print(f"✅ Найдено {len(volume_spikes)} всплесков")
                return {
                    'symbol': symbol,
                    'volume_spikes': volume_spikes,
                    'current_price': df['close'].iloc[-1],
                    'avg_volume': df['volume'].tail(20).mean(),
                    'data': df
                }
            else:
                print("❌ Всплесков не найдено")
                return None

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return None

    def scan_for_volume_spikes(self, symbol_count=20):
        """Сканирование символов на всплески объема в последних 3 свечах"""
        print("🎯 СКАНЕР ВСПЛЕСКОВ ОБЪЕМА (ПОСЛЕДНИЕ 3 СВЕЧИ)")
        print("=" * 80)
        print(f"🔍 Параметры сканирования:")
        print(f"   • Анализируемые свечи: последние {self.candles_to_analyze}")
        print(f"   • Минимальный коэффициент объема: {self.volume_threshold}x")
        print(f"   • Таймфрейм: {self.timeframe} (часовой)")
        print("=" * 80)

        symbols_data = get_high_volume_symbols(min_volume=self.min_volume)
        symbols = [item['symbol'] for item in symbols_data[:symbol_count]]

        print(f"📊 Анализируем {len(symbols)} монет...")
        print("=" * 80)

        results = []

        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] ", end="")
            result = self.analyze_symbol(symbol)

            if result:
                results.append(result)

            time.sleep(0.3)

        # Сортировка по силе всплеска
        if results:
            results.sort(key=lambda x: max(spike['volume_ratio'] for spike in x['volume_spikes']), reverse=True)

        print(f"\n{'=' * 80}")
        print("📊 РЕЗУЛЬТАТЫ СКАНИРОВАНИЯ:")
        print(f"   • Найдено символов с всплесками: {len(results)}")
        print(f"   • Анализировались последние {self.candles_to_analyze} свечи")

        if results:
            print(f"\n🎯 САМЫЕ СИЛЬНЫЕ ВСПЛЕСКИ ОБЪЕМА:")

            for i, result in enumerate(results[:15], 1):
                symbol = result['symbol']
                spikes = result['volume_spikes']
                current_price = result['current_price']

                # Сортируем всплески по силе
                spikes_sorted = sorted(spikes, key=lambda x: x['volume_ratio'], reverse=True)
                strongest_spike = spikes_sorted[0]

                # Определяем возраст свечи
                if strongest_spike['age_hours'] == 0:
                    age_text = "ТЕКУЩАЯ свеча"
                elif strongest_spike['age_hours'] == 1:
                    age_text = "1 час назад"
                else:
                    age_text = f"{strongest_spike['age_hours']} часа назад"

                print(f"\n{i}. **{symbol}**")
                print(f"   💥 {strongest_spike['candle_type']} - x{strongest_spike['volume_ratio']} ({age_text})")
                print(f"   📊 Тело: {strongest_spike['body_percent']}%")
                print(
                    f"   📍 Тени: ↑{strongest_spike['upper_shadow_percent']}% ↓{strongest_spike['lower_shadow_percent']}%")
                print(f"   💰 Текущая цена: {current_price:.6f}")

                # Показываем все всплески для этого символа
                if len(spikes) > 1:
                    other_spikes = []
                    for spike in spikes_sorted[1:]:
                        if spike['age_hours'] == 0:
                            age = "текущая"
                        else:
                            age = f"{spike['age_hours']}ч назад"
                        other_spikes.append(f"{spike['candle_type']} x{spike['volume_ratio']}")

                    print(f"   🔥 Также: {', '.join(other_spikes)}")

        else:
            print("\n❌ Всплесков объема не найдено в последних 3 свечах")

        return results

    def get_detailed_analysis(self, symbol):
        """Детальный анализ символа"""
        print(f"\n🔍 Детальный анализ {symbol} (последние {self.candles_to_analyze} свечи)...")

        data = self.get_candles_simple(symbol, self.timeframe, 50)
        if not data or data.get('code') != 0:
            print("❌ Нет данных или ошибка API")
            return

        df = self.create_dataframe_from_dict(data)
        if df is None:
            print("❌ Не удалось создать DataFrame")
            return

        volume_spikes = self.analyze_volume_spike(df)

        if not volume_spikes:
            print("❌ Всплесков объема не найдено в последних 3 свечах")
            return

        print(f"\n📊 ДЕТАЛЬНЫЙ АНАЛИЗ {symbol}:")
        print(f"   • Текущая цена: {df['close'].iloc[-1]:.6f}")
        print(f"   • Средний объем (20 свечей): {df['volume'].tail(20).mean():.0f}")
        print(f"   • Анализируются последние {self.candles_to_analyze} свечи")
        print(f"   • Всплесков найдено: {len(volume_spikes)}")

        print(f"\n🎯 ВСПЛЕСКИ ОБЪЕМА:")

        # Сортируем по времени (от самой старой к самой новой)
        volume_spikes_sorted = sorted(volume_spikes, key=lambda x: x['datetime'])

        for i, spike in enumerate(volume_spikes_sorted, 1):
            if spike['age_hours'] == 0:
                age_text = "ТЕКУЩАЯ СВЕЧА 🔥"
            elif spike['age_hours'] == 1:
                age_text = "1 час назад"
            else:
                age_text = f"{spike['age_hours']} часа назад"

            print(f"\n   {i}. {spike['candle_type']} свеча ({age_text})")
            print(f"      • Объем: x{spike['volume_ratio']} от среднего")
            print(f"      • Время: {spike['datetime'].strftime('%H:%M %d.%m')}")
            print(f"      • Цены: {spike['open']:.6f} → {spike['close']:.6f}")
            print(f"      • Тело: {spike['body_percent']}%")
            print(f"      • Тени: верхняя {spike['upper_shadow_percent']}%, нижняя {spike['lower_shadow_percent']}%")

            # Анализируем движение если это не текущая свеча
            if spike['age_hours'] > 0:
                current_price = df['close'].iloc[-1]
                spike_price = spike['close']
                change_since_spike = ((current_price - spike_price) / spike_price) * 100
                direction = "📈" if change_since_spike > 0 else "📉"
                print(f"      • Изменение: {direction} {change_since_spike:+.2f}%")


def main():
    """Основная функция"""
    scanner = VolumeSpikeScanner()

    print("🎯 СКАНЕР ВСПЛЕСКОВ ОБЪЕМА")
    print("=" * 70)
    print("🔍 Анализ последних 3 свечей на часовом таймфрейме")
    print("=" * 70)

    while True:
        print("\nВыберите действие:")
        print("1 - Сканировать топ-монеты (последние 3 свечи)")
        print("2 - Детальный анализ монеты")
        print("3 - Настроить параметры")
        print("4 - Выход")

        choice = input("\nВведите номер: ").strip()

        if choice == "1":
            count = int(input("Количество монет (10-50): ") or "20")
            scanner.scan_for_volume_spikes(count)

        elif choice == "2":
            symbol = input("Введите символ (например: BTC_USDT): ").strip().upper()
            scanner.get_detailed_analysis(symbol)

        elif choice == "3":
            print("\nНастройка параметров:")
            new_threshold = float(input(f"Коэффициент объема (текущий: {scanner.volume_threshold}): ") or "3.0")
            scanner.volume_threshold = new_threshold

            new_candles = int(input(f"Количество свечей для анализа (текущее: {scanner.candles_to_analyze}): ") or "3")
            scanner.candles_to_analyze = new_candles

            print(f"✅ Параметры установлены: {scanner.volume_threshold}x, {scanner.candles_to_analyze} свечи")

        elif choice == "4":
            print("Выход...")
            break

        else:
            print("Неверный выбор")


if __name__ == "__main__":
    main()