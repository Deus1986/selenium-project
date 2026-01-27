import time
import numpy as np
import pandas as pd
import requests
import talib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def get_candles(symbol, interval="Min30", limit=100):
    """Получение свечных данных с правильной обработкой структуры MEXC"""
    url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
    params = {
        "interval": interval,
        "limit": limit
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"DEBUG: Структура данных для {symbol}: {list(data.keys()) if data else 'No data'}")
            if data and 'data' in data:
                print(
                    f"DEBUG: Ключи в data: {list(data['data'].keys()) if isinstance(data['data'], dict) else 'Not dict'}")
            return data
        else:
            print(f"Ошибка API для {symbol}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Ошибка получения данных для {symbol}: {e}")
        return None


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
                        'price_change_percent': float(item['riseFallRate']) * 100
                    })

        return sorted(symbols, key=lambda x: x['volume_24h'], reverse=True)
    except Exception as e:
        print(f"Ошибка получения списка монет: {e}")
        return []


def create_dataframe_from_mexc_data(data):
    """Создает DataFrame из данных MEXC API"""
    if not data or not data.get('success') or not data.get('data'):
        return None

    raw_data = data['data']

    # Проверяем структуру данных
    if isinstance(raw_data, list):
        # Данные в формате списка списков
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                   'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
        df = pd.DataFrame(raw_data, columns=columns)
    elif isinstance(raw_data, dict):
        # Данные в формате словаря с массивами
        df = pd.DataFrame(raw_data)
    else:
        print(f"Неизвестный формат данных: {type(raw_data)}")
        return None

    # Преобразуем числовые колонки
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Удаляем строки с NaN
    df = df.dropna()

    return df


def calculate_technical_indicators(df):
    """Расчет технических индикаторов"""
    # Берем последние 100 свечей для расчетов
    df = df.tail(100).reset_index(drop=True)

    # Цены для расчетов
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values if 'volume' in df.columns else np.ones(len(close))

    try:
        # Трендовые индикаторы
        df['sma_20'] = talib.SMA(close, timeperiod=20)
        df['sma_50'] = talib.SMA(close, timeperiod=50)
        df['ema_12'] = talib.EMA(close, timeperiod=12)
        df['ema_26'] = talib.EMA(close, timeperiod=26)

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)

        # RSI
        df['rsi'] = talib.RSI(close, timeperiod=14)

        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, close)

        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close, timeperiod=20)

        # Volume indicators
        if 'volume' in df.columns:
            df['volume_sma'] = talib.SMA(volume, timeperiod=20)
            df['volume_ratio'] = volume / df['volume_sma']
        else:
            df['volume_ratio'] = 1.0

        # ATR (Average True Range) - волатильность
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)

        # ADX (Average Directional Index) - сила тренда
        df['adx'] = talib.ADX(high, low, close, timeperiod=14)

    except Exception as e:
        print(f"Ошибка расчета индикаторов: {e}")
        # Заполняем NaN значениями по умолчанию
        for col in ['sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd', 'macd_signal',
                    'macd_hist', 'rsi', 'stoch_k', 'stoch_d', 'bb_upper', 'bb_middle',
                    'bb_lower', 'atr', 'adx']:
            df[col] = 0 if col in ['rsi', 'stoch_k', 'stoch_d', 'adx'] else close[-1]

    return df


def analyze_price_action(df):
    """Анализ ценового действия"""
    if len(df) < 2:
        return None

    current_price = df['close'].iloc[-1]

    try:
        # Сигналы тренда
        trend_signals = {
            'sma_trend': 1 if df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1] else -1,
            'ema_trend': 1 if df['ema_12'].iloc[-1] > df['ema_26'].iloc[-1] else -1,
            'macd_trend': 1 if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else -1,
            'price_vs_sma': 1 if current_price > df['sma_20'].iloc[-1] else -1
        }

        # Сигналы перекупленности/перепроданности
        momentum_signals = {
            'rsi_signal': 0,
            'stoch_signal': 0
        }

        rsi = df['rsi'].iloc[-1]
        if not np.isnan(rsi):
            if rsi > 70:
                momentum_signals['rsi_signal'] = -1  # Перекупленность
            elif rsi < 30:
                momentum_signals['rsi_signal'] = 1  # Перепроданность

        stoch_k = df['stoch_k'].iloc[-1]
        if not np.isnan(stoch_k):
            if stoch_k > 80:
                momentum_signals['stoch_signal'] = -1
            elif stoch_k < 20:
                momentum_signals['stoch_signal'] = 1

        # Анализ волатильности
        volatility = {
            'atr_percent': (df['atr'].iloc[-1] / current_price) * 100 if df['atr'].iloc[-1] > 0 else 1,
            'bb_position': (current_price - df['bb_lower'].iloc[-1]) / (
                        df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]) if (df['bb_upper'].iloc[-1] -
                                                                               df['bb_lower'].iloc[-1]) > 0 else 0.5
        }

        # Сила тренда
        trend_strength = df['adx'].iloc[-1] if not np.isnan(df['adx'].iloc[-1]) else 25

        return {
            'trend_signals': trend_signals,
            'momentum_signals': momentum_signals,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'current_price': current_price
        }

    except Exception as e:
        print(f"Ошибка анализа ценового действия: {e}")
        return None


def predict_next_5_hours(df, analysis):
    """Прогноз движения на 5 часов"""
    if not analysis:
        return None

    # Собираем все сигналы
    trend_score = sum(analysis['trend_signals'].values())
    momentum_score = sum(analysis['momentum_signals'].values())

    # Весовые коэффициенты
    weights = {
        'trend': 0.4,
        'momentum': 0.3,
        'volatility': 0.2,
        'trend_strength': 0.1
    }

    # Базовый прогноз на основе тренда
    base_direction = 1 if trend_score > 0 else -1

    # Корректировка на основе моментума
    if momentum_score < 0 and base_direction == 1:
        base_direction = 0  # Нейтрально/коррекция
    elif momentum_score > 0 and base_direction == -1:
        base_direction = 0  # Нейтрально/отскок

    # Учет волатильности
    volatility_factor = analysis['volatility']['atr_percent'] / 2  # Нормализация

    # Учет силы тренда
    trend_strength_factor = analysis['trend_strength'] / 50  # Нормализация ADX

    # Итоговый счет
    final_score = (trend_score * weights['trend'] +
                   momentum_score * weights['momentum'] +
                   volatility_factor * weights['volatility'] +
                   trend_strength_factor * weights['trend_strength'])

    # Определение направления и уверенности
    if final_score > 0.3:
        direction = "LONG"
        confidence = min(90, (final_score + 0.3) * 25)
        reason = "Сильный восходящий тренд с поддержкой индикаторов"
    elif final_score < -0.3:
        direction = "SHORT"
        confidence = min(90, (abs(final_score) + 0.3) * 25)
        reason = "Сильный нисходящий тренд с подтверждением индикаторов"
    else:
        direction = "NEUTRAL"
        confidence = 40
        reason = "Неопределенность на рынке, ожидайте боковое движение"

    # Расчет целевых уровней
    atr = df['atr'].iloc[-1] if not np.isnan(df['atr'].iloc[-1]) else analysis['current_price'] * 0.02
    current_price = analysis['current_price']

    if direction == "LONG":
        target_price = current_price + (atr * 2.0)  # 2.0 ATR вверх
        stop_loss = current_price - (atr * 1.0)  # 1.0 ATR вниз
    elif direction == "SHORT":
        target_price = current_price - (atr * 2.0)  # 2.0 ATR вниз
        stop_loss = current_price + (atr * 1.0)  # 1.0 ATR вверх
    else:
        target_price = current_price
        stop_loss = current_price

    return {
        'direction': direction,
        'confidence': round(confidence, 1),
        'target_price': round(target_price, 6),
        'stop_loss': round(stop_loss, 6),
        'current_price': round(current_price, 6),
        'potential_profit_percent': round(abs(target_price - current_price) / current_price * 100, 2),
        'reason': reason,
        'final_score': round(final_score, 2)
    }


def analyze_symbol(symbol):
    """Полный анализ символа"""
    print(f"🔍 Анализируем {symbol}...")

    # Получаем данные
    data = get_candles(symbol, "Min30", 200)

    if not data or not data.get('success'):
        print(f"   ❌ Нет данных для {symbol}")
        return None

    # Создаем DataFrame
    df = create_dataframe_from_mexc_data(data)

    if df is None or len(df) < 50:
        print(f"   ❌ Недостаточно данных для анализа {symbol}")
        return None

    print(f"   ✅ Получено {len(df)} свечей")

    try:
        # Рассчитываем индикаторы
        df = calculate_technical_indicators(df)

        # Анализируем ценовое действие
        price_analysis = analyze_price_action(df)

        if not price_analysis:
            print(f"   ❌ Ошибка анализа ценового действия для {symbol}")
            return None

        # Делаем прогноз
        prediction = predict_next_5_hours(df, price_analysis)

        if not prediction:
            print(f"   ❌ Ошибка прогноза для {symbol}")
            return None

        # Дополнительная информация
        prediction['symbol'] = symbol
        prediction['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        prediction['rsi'] = round(df['rsi'].iloc[-1], 2) if not np.isnan(df['rsi'].iloc[-1]) else 50
        prediction['volume_ratio'] = round(df['volume_ratio'].iloc[-1], 2) if 'volume_ratio' in df.columns else 1.0

        print(f"   ✅ Прогноз: {prediction['direction']} (уверенность: {prediction['confidence']}%)")

        return prediction

    except Exception as e:
        print(f"   ❌ Ошибка анализа {symbol}: {e}")
        return None


def main_analysis():
    """Основная функция анализа"""
    print("🚀 ЗАПУСК АНАЛИЗА ЦЕНЫ НА 5 ЧАСОВ ВПЕРЕД")
    print("=" * 80)

    # Получаем список монет с высоким объемом
    symbols_data = get_high_volume_symbols(min_volume=20000000)

    if not symbols_data:
        print("❌ Не удалось получить список монет")
        return []

    symbols = [item['symbol'] for item in symbols_data]

    print(f"📊 Анализируем {len(symbols)} монет с высоким объемом...")
    print("⏰ Прогноз на ближайшие 5 часов")
    print("=" * 80)

    results = []

    for symbol_info in symbols_data:  # Анализируем топ-10
        symbol = symbol_info['symbol']

        prediction = analyze_symbol(symbol)

        if prediction:
            results.append(prediction)

        time.sleep(0.3)  # Пауза между запросами

    # Сортируем результаты по уверенности
    results.sort(key=lambda x: x['confidence'], reverse=True)

    # Вывод итогов
    print(f"\n{'=' * 80}")
    print("🎯 ЛУЧШИЕ СИГНАЛЫ ДЛЯ ТОРГОВЛИ:")
    print(f"{'=' * 80}")

    if results:
        for i, result in enumerate(results, 1):
            direction_icon = "🟢" if result['direction'] == 'LONG' else "🔴" if result['direction'] == 'SHORT' else "⚪"
            print(f"\n{i}. {direction_icon} {result['symbol']}")
            print(f"   Направление: {result['direction']}")
            print(f"   Уверенность: {result['confidence']}%")
            print(f"   Текущая цена: {result['current_price']}")
            print(f"   Цель: {result['target_price']} ({result['potential_profit_percent']}%)")
            print(f"   Стоп-лосс: {result['stop_loss']}")
            print(f"   RSI: {result['rsi']}")
            print(f"   Объем: {result['volume_ratio']}x от среднего")
            print(f"   Причина: {result['reason']}")
    else:
        print("\n❌ Не найдено подходящих сигналов для торговли")

    return results

def test_5_hours():
    # Основной анализ
    results = main_analysis()

    # Для мониторинга конкретного символа раскомментируйте:
    # if results:
    #     monitor_specific_symbol(results[0]['symbol'])