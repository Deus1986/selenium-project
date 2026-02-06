"""
Пересечение EMA 10 и EMA 60 на таймфрейме 1H.
Запуск вручную, при сигнале — звук; уже отданные не повторяются (состояние в JSON).
Монеты: объём > 10M, без ограничения по топу — все такие монеты.
Запуск на 1-й секунде каждого часа (HH:00:01).

Запуск:  python test-scripts/trade_ema_cross_1h_live.py
"""
import sys
import os
import json
import time
import winsound
import numpy as np
from datetime import datetime, timedelta

_tests = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")
sys.path.insert(0, os.path.abspath(_tests))
from test_trend_change_signals_1h import get_candles, create_dataframe, add_indicators, format_price, get_high_volume_symbols
import talib

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ema_cross_1h_live_state.json")
CANDLES_LIMIT = 400
INTERVAL = "Min60"
MIN_VOLUME = 10_000_000
MIN_BARS = 70  # чтобы EMA 60 успела построиться


def load_state():
    try:
        if os.path.isfile(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
    except Exception:
        pass
    return set()


def save_state(state_set):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(list(state_set)), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("  Ошибка сохранения состояния:", e)


def wait_until_next_hour():
    now = datetime.now()
    next_hour = (now.replace(minute=0, second=1, microsecond=0) + timedelta(hours=1))
    sleep_seconds = (next_hour - now).total_seconds()
    if sleep_seconds > 0:
        print(f"  Ожидание до следующего часа: {int(sleep_seconds)} сек.")
        time.sleep(sleep_seconds)


def check_ema_cross(df):
    """
    Пересечение EMA 10 и EMA 60 на последней закрытой свече.
    Возвращает "LONG", "SHORT" или None, плюс entry, signal_time.
    """
    if df is None or len(df) < MIN_BARS:
        return None
    df = df.copy()
    c = df["close"].values
    if "ema_10" not in df.columns:
        df["ema_10"] = talib.EMA(c, timeperiod=10)
    df["ema_60"] = talib.EMA(c, timeperiod=60)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if np.isnan(last["ema_10"]) or np.isnan(last["ema_60"]) or np.isnan(prev["ema_10"]) or np.isnan(prev["ema_60"]):
        return None
    # Кросс вверх: было EMA10 < EMA60, стало EMA10 > EMA60
    if prev["ema_10"] < prev["ema_60"] and last["ema_10"] > last["ema_60"]:
        return {
            "direction": "LONG",
            "entry": float(last["close"]),
            "ema_10": float(last["ema_10"]),
            "ema_60": float(last["ema_60"]),
            "signal_time": last["datetime"].strftime("%Y-%m-%d %H:%M") if hasattr(last["datetime"], "strftime") else str(last["datetime"]),
        }
    # Кросс вниз: было EMA10 > EMA60, стало EMA10 < EMA60
    if prev["ema_10"] > prev["ema_60"] and last["ema_10"] < last["ema_60"]:
        return {
            "direction": "SHORT",
            "entry": float(last["close"]),
            "ema_10": float(last["ema_10"]),
            "ema_60": float(last["ema_60"]),
            "signal_time": last["datetime"].strftime("%Y-%m-%d %H:%M") if hasattr(last["datetime"], "strftime") else str(last["datetime"]),
        }
    return None


def run_cycle(state_set):
    # Все монеты с объёмом > 10M, без ограничения топ-N
    symbols = [s["symbol"] for s in get_high_volume_symbols(MIN_VOLUME)]
    new_signals = []
    for symbol in symbols:
        try:
            data = get_candles(symbol, INTERVAL, CANDLES_LIMIT)
            df = create_dataframe(data)
            if df is None or len(df) < MIN_BARS:
                continue
            df = add_indicators(df)
            res = check_ema_cross(df)
            if not res:
                continue
            key = f"{symbol}|{res['direction']}|{res['signal_time']}"
            if key in state_set:
                continue
            state_set.add(key)
            new_signals.append({
                "symbol": symbol,
                "direction": res["direction"],
                "signal_time": res["signal_time"],
                "entry": res["entry"],
                "ema_10": res["ema_10"],
                "ema_60": res["ema_60"],
            })
        except Exception as e:
            print(f"  {symbol}: {e}")
        time.sleep(0.15)
    return state_set, new_signals


def main():
    print()
    print("=" * 60)
    print("  EMA 10 / EMA 60 CROSS 1H LIVE — СКРИПТ ЗАПУЩЕН")
    print("  Пересечения на 1H, монеты с объёмом > 10M (все без ограничений).")
    print("  Старт:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("  Режим: проверка на 1-й секунде каждого часа (HH:00:01).")
    print("=" * 60)
    print()

    state_set = load_state()
    while True:
        wait_until_next_hour()
        print(f"\n⏰ Запуск проверки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        state_set, new_signals = run_cycle(state_set)
        if new_signals:
            save_state(state_set)
            for s in new_signals:
                print(f"[СИГНАЛ] {s['symbol']}  {s['direction']}  Вход: {format_price(s['entry'])}  EMA10: {format_price(s['ema_10'])}  EMA60: {format_price(s['ema_60'])}  | {s['signal_time']}")
            try:
                winsound.Beep(1200, 500)
                winsound.Beep(800, 400)
            except Exception:
                pass
        else:
            print("  Пересечений в этом часу нет.")


if __name__ == "__main__":
    main()
