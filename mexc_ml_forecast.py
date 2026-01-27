import os
import time
import argparse
import numpy as np
import pandas as pd
import requests
import talib
from sklearn.ensemble import RandomForestClassifier
import joblib


def get_candles(symbol, interval="Min30", limit=100):
    url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
    params = {"interval": interval, "limit": limit}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("data"):
                return data
        return None
    except Exception as e:
        print(f"⚠️ {symbol}: {e}")
        return None


def get_active_symbols(min_volume=20_000_000):
    url = "https://contract.mexc.com/api/v1/contract/ticker"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        symbols = []
        if 'data' in data:
            for item in data["data"]:
                symbol = item.get("symbol")
                vol = item.get("amount24", 0)
                if symbol and symbol.endswith("_USDT") and vol >= min_volume:
                    test_data = get_candles(symbol, "Min30", 5)
                    if test_data is not None:
                        symbols.append(symbol)
                        if len(symbols) >= 12:
                            break
        return symbols
    except Exception as e:
        print(f"❌ Ticker error: {e}")
        return []


def create_dataframe_from_mexc_data(data):
    raw = data['data']
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(raw, columns=cols[:len(raw[0])])
    else:
        return None
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df.dropna().reset_index(drop=True)


def add_features(df):
    if len(df) < 50:
        return None
    h, l, c, v = df['high'].values, df['low'].values, df['close'].values, df['volume'].values

    df = df.copy()
    df['rsi'] = talib.RSI(c, 14)
    df['macd'], df['macd_signal'], _ = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    df['adx'] = talib.ADX(h, l, c, 14)
    df['atr'] = talib.ATR(h, l, c, 14)
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = talib.BBANDS(c, 20)
    df['volume_sma'] = talib.SMA(v, 20)
    df['volume_ratio'] = v / df['volume_sma'].replace(0, np.nan)
    df['price_vs_sma20'] = c / talib.SMA(c, 20)
    df['target'] = np.nan
    return df


def define_target(df, future_bars=5, threshold=0.01):
    future = df['close'].shift(-future_bars)
    ret = (future - df['close']) / df['close']
    df = df.copy()
    df['target'] = np.nan
    df.loc[ret >= threshold, 'target'] = 1
    df.loc[ret <= -threshold, 'target'] = 0
    return df


def collect_historical_data():
    print("🔍 Получение активных символов...")
    symbols = get_active_symbols(min_volume=20_000_000)
    if not symbols:
        print("❌ Нет ликвидных пар")
        return

    print(f"✅ Найдено: {', '.join(symbols)}")
    all_dfs = []

    for symbol in symbols:
        print(f"  → {symbol}")
        data = get_candles(symbol, "Min30", 100)
        if not data:
            continue
        df = create_dataframe_from_mexc_data(data)
        if df is None or len(df) < 50:
            continue
        df = add_features(df)
        if df is None:
            continue
        df = define_target(df, future_bars=5, threshold=0.01)
        df = df.dropna(subset=['target'])
        if len(df) < 10:
            continue
        df['symbol'] = symbol
        all_dfs.append(df)
        time.sleep(0.3)

    if not all_dfs:
        print("❌ Нет валидных данных")
        return

    full = pd.concat(all_dfs, ignore_index=True)
    full.to_csv("mexc_dataset.csv", index=False)
    print(f"✅ Сохранено {len(full)} строк в mexc_dataset.csv")


FEATURES = ['rsi', 'macd', 'macd_signal', 'adx', 'atr', 'volume_ratio', 'price_vs_sma20']


def train_model():
    if not os.path.exists("mexc_dataset.csv"):
        print("❌ Нет датасета. Сначала запустите: --mode collect")
        return None
    df = pd.read_csv("mexc_dataset.csv")
    df = df.dropna(subset=FEATURES + ['target'])
    if len(df) < 50:
        print("❌ Недостаточно данных для обучения")
        return None
    X, y = df[FEATURES], df['target'].astype(int)
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X, y)
    joblib.dump(model, "mexc_rf_model.pkl")
    print(f"✅ Модель обучена на {len(df)} примерах")
    return model


def predict_symbol(symbol, model):
    data = get_candles(symbol, "Min30", 100)
    if not data:
        return None
    df = create_dataframe_from_mexc_data(data)
    if df is None or len(df) < 50:
        return None
    df = add_features(df)
    if df is None:
        return None
    latest = df.iloc[-1]
    if latest[FEATURES].isnull().any():
        return None
    X = latest[FEATURES].values.reshape(1, -1)
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    current = latest['close']
    atr = latest['atr'] if not pd.isna(latest['atr']) else current * 0.015
    direction = "LONG" if pred == 1 else "SHORT"
    target = current + atr * 2 if pred == 1 else current - atr * 2
    stop = current - atr if pred == 1 else current + atr
    return {
        'symbol': symbol,
        'direction': direction,
        'confidence': round(max(proba) * 100, 1),
        'current_price': round(current, 6),
        'target_price': round(target, 6),
        'stop_loss': round(stop, 6),
        'potential_profit_percent': round(abs(target - current) / current * 100, 2),
        'rsi': round(latest['rsi'], 2) if not pd.isna(latest['rsi']) else 50
    }


def run_prediction():
    symbols = get_active_symbols(min_volume=10_000_000)
    if not symbols:
        print("❌ Нет пар для прогноза")
        return
    if os.path.exists("mexc_rf_model.pkl"):
        model = joblib.load("mexc_rf_model.pkl")
        print("🧠 Модель загружена")
    else:
        print("🔄 Обучение модели...")
        model = train_model()
        if model is None:
            return
    results = []
    for s in symbols[:10]:
        p = predict_symbol(s, model)
        if p:
            results.append(p)
        time.sleep(0.3)
    results.sort(key=lambda x: x['confidence'], reverse=True)
    print(f"\n{'='*60}\n🤖 ТОП ML-СИГНАЛЫ:\n{'='*60}")
    for i, r in enumerate(results, 1):
        icon = "🟢" if r['direction'] == 'LONG' else "🔴"
        print(f"\n{i}. {icon} {r['symbol']}")
        print(f"   Уверенность: {r['confidence']}% | Цена: {r['current_price']}")
        print(f"   Цель: {r['target_price']} ({r['potential_profit_percent']}%) | Стоп: {r['stop_loss']}")
        print(f"   RSI: {r['rsi']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["collect", "predict"], default="predict")
    args = parser.parse_args()
    if args.mode == "collect":
        collect_historical_data()
    else:
        run_prediction()