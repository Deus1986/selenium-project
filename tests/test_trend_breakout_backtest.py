import time
import os
import numpy as np
import pandas as pd
import requests
import talib
from datetime import datetime


def get_high_volume_symbols(min_volume=10_000_000):
    """Получает список фьючерсных монет с высоким объемом на MEXC."""
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


def get_candles(symbol, interval="Min30", limit=500):
    """Свечи с MEXC (Min30 по умолчанию)."""
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
    """Нормализует ответ API в DataFrame."""
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

    df["ema_fast"] = talib.EMA(close, timeperiod=21)
    df["ema_slow"] = talib.EMA(close, timeperiod=55)
    df["rsi"] = talib.RSI(close, timeperiod=14)
    df["atr"] = talib.ATR(high, low, close, timeperiod=14)
    df["vol_sma"] = talib.SMA(volume, timeperiod=20)
    df["vol_ratio"] = volume / df["vol_sma"]
    df["range"] = df["high"] - df["low"]
    return df


def evaluate_current_signal(df, lookback=20):
    """
    Возвращает текущую рекомендацию: входить или нет.
    Сигнальная свеча: последняя ЗАКРЫТАЯ (iloc[-2]).
    Вход: по текущей цене (close последней свечи).
    """
    if len(df) < lookback + 5:
        return None

    signal = df.iloc[-2]
    current = df.iloc[-1]
    window = df.iloc[-(lookback + 2):-2]

    if np.isnan(signal["ema_fast"]) or np.isnan(signal["ema_slow"]) or np.isnan(signal["atr"]):
        return None

    local_high = window["high"].max()
    local_low = window["low"].min()

    long_conditions = [
        ("EMA21 > EMA55", signal["ema_fast"] > signal["ema_slow"]),
        ("RSI > 55", signal["rsi"] > 55),
        ("Объем > 1.3x", signal["vol_ratio"] > 1.3),
        ("Пробой локального high", signal["close"] > local_high),
    ]

    short_conditions = [
        ("EMA21 < EMA55", signal["ema_fast"] < signal["ema_slow"]),
        ("RSI < 45", signal["rsi"] < 45),
        ("Объем > 1.3x", signal["vol_ratio"] > 1.3),
        ("Пробой локального low", signal["close"] < local_low),
    ]

    long_ok = all(c[1] for c in long_conditions)
    short_ok = all(c[1] for c in short_conditions)

    direction = ""
    enter_now = "Нет"
    reasons = []

    if long_ok:
        direction = "LONG"
        enter_now = "Да"
        reasons = [c[0] for c in long_conditions]
    elif short_ok:
        direction = "SHORT"
        enter_now = "Да"
        reasons = [c[0] for c in short_conditions]
    else:
        # Показываем, чего не хватает
        missing_long = [c[0] for c in long_conditions if not c[1]]
        missing_short = [c[0] for c in short_conditions if not c[1]]
        reasons.append("LONG не подходит: " + ", ".join(missing_long))
        reasons.append("SHORT не подходит: " + ", ".join(missing_short))

    entry = current["close"]
    atr = signal["atr"]
    if np.isnan(atr) or atr == 0:
        atr = entry * 0.01

    if direction == "LONG":
        stop = entry - atr
        tp1 = entry + atr * 2
        rr = (tp1 - entry) / (entry - stop) if entry - stop > 0 else 0
    elif direction == "SHORT":
        stop = entry + atr
        tp1 = entry - atr * 2
        rr = (entry - tp1) / (stop - entry) if stop - entry > 0 else 0
    else:
        stop = None
        tp1 = None
        rr = None

    return {
        "enter_now": enter_now,
        "direction": direction or "-",
        "entry": round(entry, 6),
        "stop": round(stop, 6) if stop else "",
        "take_profit": round(tp1, 6) if tp1 else "",
        "rr": round(rr, 2) if rr is not None else "",
        "reasons": " | ".join(reasons),
        "signal_time": signal["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
    }


def generate_signals(df):
    """
    Тренд + импульс + объём:
    LONG: EMA21>EMA55, RSI>55, объём > 1.3x, пробой локального high.
    SHORT: EMA21<EMA55, RSI<45, объём > 1.3x, пробой локального low.
    """
    signals = []
    lookback = 20
    for i in range(lookback, len(df) - 1):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        window = df.iloc[i - lookback:i]

        if np.isnan(row["ema_fast"]) or np.isnan(row["ema_slow"]) or np.isnan(row["atr"]):
            continue

        local_high = window["high"].max()
        local_low = window["low"].min()

        # LONG
        if (
            row["ema_fast"] > row["ema_slow"]
            and row["rsi"] > 55
            and row["vol_ratio"] > 1.3
            and row["close"] > local_high
        ):
            signals.append(("LONG", i))

        # SHORT
        if (
            row["ema_fast"] < row["ema_slow"]
            and row["rsi"] < 45
            and row["vol_ratio"] > 1.3
            and row["close"] < local_low
        ):
            signals.append(("SHORT", i))

    return signals


def backtest(df, max_hold_candles=12, rr_target=2.0):
    """
    Простой бэктест:
    - вход по сигналу
    - стоп = 1 ATR
    - цель = 2R
    - time stop = max_hold_candles
    """
    trades = []
    signals = generate_signals(df)

    for direction, idx in signals:
        entry = df.iloc[idx + 1]["open"]  # вход на следующей свече
        atr = df.iloc[idx]["atr"]
        if np.isnan(atr) or atr == 0:
            continue

        if direction == "LONG":
            stop = entry - atr
            target = entry + atr * rr_target
        else:
            stop = entry + atr
            target = entry - atr * rr_target

        exit_price = None
        exit_reason = None
        exit_index = None

        for j in range(idx + 1, min(idx + 1 + max_hold_candles, len(df))):
            high = df.iloc[j]["high"]
            low = df.iloc[j]["low"]

            if direction == "LONG":
                if low <= stop:
                    exit_price = stop
                    exit_reason = "STOP"
                    exit_index = j
                    break
                if high >= target:
                    exit_price = target
                    exit_reason = "TARGET"
                    exit_index = j
                    break
            else:
                if high >= stop:
                    exit_price = stop
                    exit_reason = "STOP"
                    exit_index = j
                    break
                if low <= target:
                    exit_price = target
                    exit_reason = "TARGET"
                    exit_index = j
                    break

        # time stop
        if exit_price is None:
            exit_index = min(idx + max_hold_candles, len(df) - 1)
            exit_price = df.iloc[exit_index]["close"]
            exit_reason = "TIME"

        pnl = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
        pnl_pct = pnl / entry * 100

        trades.append({
            "direction": direction,
            "entry_time": df.iloc[idx + 1]["datetime"],
            "entry": round(entry, 6),
            "stop": round(stop, 6),
            "target": round(target, 6),
            "exit_time": df.iloc[exit_index]["datetime"],
            "exit": round(exit_price, 6),
            "exit_reason": exit_reason,
            "pnl_pct": round(pnl_pct, 2),
        })

    return trades


def summarize_trades(trades):
    if not trades:
        return {"total": 0, "win_rate": 0, "avg_pnl": 0}
    wins = [t for t in trades if t["pnl_pct"] > 0]
    win_rate = len(wins) / len(trades) * 100
    avg_pnl = sum(t["pnl_pct"] for t in trades) / len(trades)
    return {
        "total": len(trades),
        "win_rate": round(win_rate, 2),
        "avg_pnl": round(avg_pnl, 2),
    }


def run_strategy():
    print("🚀 СИГНАЛЫ НА СЕЙЧАС (MEXC, 30m)")
    print("Условия: EMA21/EMA55 + RSI + объём > 1.3x + пробой локальных уровней")
    print("Выход: стоп 1 ATR, цель 2R")

    symbols = get_high_volume_symbols(min_volume=10_000_000)
    if not symbols:
        print("❌ Нет монет по объёму")
        return

    rows = []
    for s in symbols:
        symbol = s["symbol"]
        data = get_candles(symbol, "Min30", 500)
        df = create_dataframe(data)
        if df is None:
            continue
        df = add_indicators(df)
        signal = evaluate_current_signal(df)
        if not signal:
            continue
        rows.append({
            "Символ": symbol,
            "Вход сейчас": signal["enter_now"],
            "Направление": signal["direction"],
            "Время сигнала": signal["signal_time"],
            "Цена входа": signal["entry"],
            "Стоп-лосс": signal["stop"],
            "Тейк-профит": signal["take_profit"],
            "R/R": signal["rr"],
            "Причины/условия": signal["reasons"],
        })
        time.sleep(0.25)

    # Сортировка: сначала "Да", потом "Нет"
    rows = sorted(rows, key=lambda x: (x["Вход сейчас"] != "Да", x["Символ"]))
    out_file = "trend_breakout_signals.xlsx"
    with pd.ExcelWriter(out_file, engine="openpyxl", mode="w") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Сигналы", index=False)

    print(f"✅ Готово: {os.path.abspath(out_file)}")


if __name__ == "__main__":
    run_strategy()
