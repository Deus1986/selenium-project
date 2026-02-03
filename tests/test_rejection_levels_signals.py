import time
import os
import json
import winsound
import numpy as np
import pandas as pd
import requests
import talib
from datetime import datetime


STATE_FILE = "rejection_levels_state.json"
MIN_PROFIT_PCT = 0.05  # минимальная прибыль 5% от входа (3R >= 5%)


def format_price(value):
    """Формат цены без научной записи."""
    if value is None or value == "":
        return ""
    if value < 0.0001:
        return f"{value:.8f}"
    if value < 0.001:
        return f"{value:.7f}"
    if value < 0.01:
        return f"{value:.6f}"
    if value < 1:
        return f"{value:.5f}"
    if value < 100:
        return f"{value:.4f}"
    return f"{value:.2f}"


def get_high_volume_symbols(min_volume=10_000_000):
    """Фьючерсные монеты MEXC с высоким объемом."""
    url = "https://contract.mexc.com/api/v1/contract/ticker"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        symbols = []
        if "data" in data:
            for item in data["data"]:
                if item.get("amount24", 0) > min_volume:
                    symbols.append({
                        "symbol": item["symbol"],
                        "volume_24h": item["amount24"],
                    })
        return sorted(symbols, key=lambda x: x["volume_24h"], reverse=True)
    except Exception as e:
        print(f"Ошибка получения списка монет: {e}")
        return []


def get_candles(symbol, interval="Min5", limit=800):
    url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
    params = {"interval": interval, "limit": limit}
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"   ⚠️ Ошибка получения данных для {symbol}: {e}")
    return None


def create_dataframe(data):
    if not data or not data.get("success") or not data.get("data"):
        return None
    raw = data["data"]
    try:
        if isinstance(raw, dict):
            required = ["time", "open", "high", "low", "close", "vol"]
            if not all(k in raw for k in required):
                return None
            length = min(len(raw["time"]), len(raw["open"]), len(raw["high"]),
                         len(raw["low"]), len(raw["close"]), len(raw["vol"]))
            if length == 0:
                return None
            df = pd.DataFrame({
                "timestamp": raw["time"][:length],
                "open": raw["open"][:length],
                "high": raw["high"][:length],
                "low": raw["low"][:length],
                "close": raw["close"][:length],
                "volume": raw["vol"][:length],
            })
        elif isinstance(raw, list):
            cleaned = []
            for row in raw:
                if isinstance(row, (list, tuple)) and len(row) >= 6:
                    cleaned.append(row[:6])
            if not cleaned:
                return None
            df = pd.DataFrame(cleaned, columns=["timestamp", "open", "high", "low", "close", "volume"])
        else:
            return None

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()
        if len(df) < 200:
            return None

        ts = df["timestamp"].iloc[0]
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms" if ts > 1e12 else "s")
        df = df.sort_values("datetime").reset_index(drop=True)
        return df
    except Exception:
        return None


def add_indicators(df):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values
    df["atr"] = talib.ATR(high, low, close, timeperiod=14)
    df["vol_sma"] = talib.SMA(volume, timeperiod=20)
    df["vol_ratio"] = volume / df["vol_sma"]
    return df


def find_strong_levels(df, lookback=150, vol_threshold=1.8, min_touches=2):
    """Ищем сильные уровни по всплеску объема на экстремумах + повторные касания."""
    window = df.tail(lookback)
    levels = []

    for i in range(2, len(window) - 2):
        row = window.iloc[i]
        if row["vol_ratio"] < vol_threshold:
            continue

        # Swing high
        if row["high"] > window.iloc[i - 1]["high"] and row["high"] > window.iloc[i + 1]["high"]:
            levels.append(("RES", row["high"], row["vol_ratio"]))

        # Swing low
        if row["low"] < window.iloc[i - 1]["low"] and row["low"] < window.iloc[i + 1]["low"]:
            levels.append(("SUP", row["low"], row["vol_ratio"]))

    # Подтверждаем уровни количеством касаний
    atr = df["atr"].iloc[-2] if not pd.isna(df["atr"].iloc[-2]) else df["close"].iloc[-2] * 0.003
    tolerance = max(atr * 0.4, df["close"].iloc[-2] * 0.002)
    confirmed = []
    for lvl_type, level, lvl_vol in levels:
        if lvl_type == "RES":
            touches = (abs(window["high"] - level) <= tolerance).sum()
        else:
            touches = (abs(window["low"] - level) <= tolerance).sum()
        if touches >= min_touches:
            confirmed.append((lvl_type, level, lvl_vol, touches))

    # Сортируем по объему (сила уровня)
    confirmed.sort(key=lambda x: (x[2], x[3]), reverse=True)
    return confirmed[:5]  # берём только самые сильные уровни


def evaluate_rejection(df):
    """
    Сценарии:
    - Resistance Rejection SHORT
    - Support Rejection LONG
    Требования:
    - касание сильного уровня
    - откат и закрытие обратно
    - подтверждение следующей свечой
    - объем выше среднего
    """
    signal = df.iloc[-2]
    confirm = df.iloc[-1]

    if np.isnan(signal["atr"]) or signal["atr"] <= 0:
        return None

    levels = find_strong_levels(df)
    if not levels:
        return None

    tolerance = max(signal["atr"] * 0.3, signal["close"] * 0.002)
    vol_ok = signal["vol_ratio"] > 1.4

    # Форма свечи (отбой должен иметь длинную тень)
    body = abs(signal["close"] - signal["open"])
    upper_wick = signal["high"] - max(signal["close"], signal["open"])
    lower_wick = min(signal["close"], signal["open"]) - signal["low"]
    total_range = max(signal["high"] - signal["low"], 1e-9)
    upper_wick_ratio = upper_wick / total_range
    lower_wick_ratio = lower_wick / total_range

    for lvl_type, level, lvl_vol, touches in levels:
        if lvl_type == "RES":
            touched = abs(signal["high"] - level) <= tolerance
            rejected = signal["close"] < level and signal["close"] < signal["open"]
            confirmed = confirm["close"] < signal["close"]
            strong_wick = upper_wick_ratio >= 0.5 and lower_wick_ratio < 0.4
            if touched and rejected and confirmed and vol_ok and strong_wick:
                entry = confirm["close"]
                stop = level + signal["atr"] * 0.6
                take = entry - (stop - entry) * 3.0
                profit_pct = (entry - take) / entry
                if profit_pct < MIN_PROFIT_PCT:
                    continue
                return {
                    "enter_now": "Да",
                    "direction": "SHORT",
                    "entry": entry,
                    "stop": stop,
                    "take_profit": take,
                    "rr": round(abs((entry - take) / (stop - entry)), 2),
                    "reasons": f"Отбой от сопротивления {round(level,6)} (объем {round(lvl_vol,2)}x, касаний {touches})",
                    "signal_time": signal["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
                }

        if lvl_type == "SUP":
            touched = abs(signal["low"] - level) <= tolerance
            rejected = signal["close"] > level and signal["close"] > signal["open"]
            confirmed = confirm["close"] > signal["close"]
            strong_wick = lower_wick_ratio >= 0.5 and upper_wick_ratio < 0.4
            if touched and rejected and confirmed and vol_ok and strong_wick:
                entry = confirm["close"]
                stop = level - signal["atr"] * 0.6
                take = entry + (entry - stop) * 3.0
                profit_pct = (take - entry) / entry
                if profit_pct < MIN_PROFIT_PCT:
                    continue
                return {
                    "enter_now": "Да",
                    "direction": "LONG",
                    "entry": entry,
                    "stop": stop,
                    "take_profit": take,
                    "rr": round(abs((take - entry) / (entry - stop)), 2),
                    "reasons": f"Отбой от поддержки {round(level,6)} (объем {round(lvl_vol,2)}x, касаний {touches})",
                    "signal_time": signal["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
                }

    return None


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def save_state(state_set):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(list(state_set)), f, ensure_ascii=False, indent=2)


def wait_until_next_5min():
    now = datetime.now()
    next_min = ((now.minute // 5) + 1) * 5
    if next_min == 60:
        next_time = now.replace(minute=0, second=1, microsecond=0) + pd.Timedelta(hours=1)
    else:
        next_time = now.replace(minute=next_min, second=1, microsecond=0)
    sleep_seconds = (next_time - now).total_seconds()
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)


def run_once(state_set):
    print("🚀 ОТБОЙ ОТ СИЛЬНЫХ УРОВНЕЙ (MEXC, 5m)")
    print("Только сильные уровни по объему. R/R ≥ 1:3")

    symbols = get_high_volume_symbols(min_volume=10_000_000)
    if not symbols:
        print("❌ Нет монет по объёму")
        return state_set

    rows = []
    new_entries = []

    for s in symbols:
        symbol = s["symbol"]
        data = get_candles(symbol, "Min5", 800)
        df = create_dataframe(data)
        if df is None:
            continue
        df = add_indicators(df)
        signal = evaluate_rejection(df)
        if not signal:
            continue
        row = {
            "Символ": symbol,
            "Вход сейчас": signal["enter_now"],
            "Направление": signal["direction"],
            "Время сигнала": signal["signal_time"],
            "Цена входа": format_price(signal["entry"]),
            "Стоп-лосс": format_price(signal["stop"]),
            "Тейк-профит (R≥3)": format_price(signal["take_profit"]),
            "R/R": signal["rr"],
            "Причины/условия": signal["reasons"],
        }
        rows.append(row)

        key = f"{symbol}|{row['Направление']}|{row['Время сигнала']}"
        if key not in state_set:
            new_entries.append(row)
            state_set.add(key)

        time.sleep(0.25)

    out_file = "rejection_levels_signals.xlsx"
    temp_file = "rejection_levels_tmp.xlsx"
    with pd.ExcelWriter(temp_file, engine="openpyxl", mode="w") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Сигналы", index=False)
    try:
        os.replace(temp_file, out_file)
    except PermissionError:
        ts_name = f"rejection_levels_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        os.replace(temp_file, ts_name)

    if new_entries:
        print("\n🔔 НОВЫЕ ТОЧКИ ВХОДА:")
        for n in new_entries:
            print(f"   {n['Символ']} {n['Направление']} | Вход {n['Цена входа']} | Стоп {n['Стоп-лосс']} | Тейк {n['Тейк-профит (R≥3)']}")
        try:
            winsound.MessageBeep(winsound.MB_ICONHAND)
        except Exception:
            pass
        for _ in range(3):
            winsound.Beep(1200, 600)
            winsound.Beep(800, 400)
            time.sleep(0.1)

    return state_set


def run_loop():
    state_set = load_state()
    print("⏱️ Запуск: каждые 5 минут на 01-й секунде")
    while True:
        wait_until_next_5min()
        print(f"\n⏰ Запуск проверки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            state_set = run_once(state_set)
            save_state(state_set)
        except Exception as e:
            print(f"⚠️ Ошибка цикла: {e}")


if __name__ == "__main__":
    run_loop()
