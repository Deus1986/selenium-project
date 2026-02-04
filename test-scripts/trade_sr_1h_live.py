"""
S/R 1H — торговля по сигналам на реальной бирже MEXC.
При появлении сигнала — звук; уже отданные позиции не повторяются (состояние в JSON).

Запуск:  python test-scripts/trade_sr_1h_live.py
Цикл: проверка всех монет с объёмом > 10M, затем пауза 60 сек.
"""
import sys
import os
import json
import time
import winsound
from datetime import datetime

_tests = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")
sys.path.insert(0, os.path.abspath(_tests))
from test_trend_change_signals_1h import get_candles, create_dataframe, add_indicators, format_price, get_high_volume_symbols
from sr_algorithm_1h import evaluate_sr_1h

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr_1h_live_state.json")
CANDLES_LIMIT = 400
INTERVAL = "Min60"
MIN_VOLUME = 10_000_000
PAUSE_SEC = 60


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


def run_cycle(state_set):
    symbols = [s["symbol"] for s in get_high_volume_symbols(MIN_VOLUME)]
    new_signals = []
    for symbol in symbols:
        try:
            data = get_candles(symbol, INTERVAL, CANDLES_LIMIT)
            df = create_dataframe(data)
            if df is None or len(df) < 100:
                continue
            df = add_indicators(df)
            res = evaluate_sr_1h(df)
            if not res or res.get("enter_now") != "Да":
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
                "stop": res["stop"],
                "take_profit": res["take_profit"],
                "reasons": res.get("reasons", ""),
            })
        except Exception as e:
            print(f"  {symbol}: {e}")
        time.sleep(0.15)
    return state_set, new_signals


def main():
    print()
    print("=" * 60)
    print("  S/R 1H LIVE — СКРИПТ ЗАПУЩЕН")
    print("  Торговля по сигналам (MEXC), реальные монеты.")
    print("  Старт:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    print()

    state_set = load_state()
    while True:
        state_set, new_signals = run_cycle(state_set)
        if new_signals:
            save_state(state_set)
            for s in new_signals:
                print(f"[СИГНАЛ] {s['symbol']}  {s['direction']}  Вход: {format_price(s['entry'])}  Стоп: {format_price(s['stop'])}  Тейк: {format_price(s['take_profit'])}  | {s['signal_time']}")
                if s.get("reasons"):
                    print(f"         Причина: {s['reasons']}")
            try:
                winsound.Beep(1200, 500)
                winsound.Beep(800, 400)
            except Exception:
                pass
        else:
            print(f"  Проверка завершена. Следующий цикл через {PAUSE_SEC} сек.")
        time.sleep(PAUSE_SEC)


if __name__ == "__main__":
    main()
