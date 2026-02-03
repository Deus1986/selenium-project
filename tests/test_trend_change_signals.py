import time
import os
import json
import winsound
import numpy as np
import pandas as pd
import requests
import talib
from datetime import datetime


MIN_RR = 3.0
MIN_PROFIT_PCT = 0.05  # минимальная прибыль 5% от входа


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
    """Список фьючерсных монет MEXC с высоким объемом."""
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
        if len(df) < 250:
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

    df["ema_fast"] = talib.EMA(close, timeperiod=50)
    df["ema_slow"] = talib.EMA(close, timeperiod=200)
    df["atr"] = talib.ATR(high, low, close, timeperiod=14)
    df["vol_sma"] = talib.SMA(volume, timeperiod=20)
    df["vol_ratio"] = volume / df["vol_sma"]
    df["swing_high"] = df["high"].rolling(20).max().shift(1)
    df["swing_low"] = df["low"].rolling(20).min().shift(1)
    return df


def evaluate_trend_change(df):
    """
    Поиск разворота тренда.
    Сигнальная свеча: последняя закрытая (iloc[-2]).
    Вход: текущая цена (close последней свечи).
    """
    if len(df) < 300:
        return None

    signal = df.iloc[-2]
    confirm = df.iloc[-1]
    current = df.iloc[-1]

    if np.isnan(signal["ema_fast"]) or np.isnan(signal["ema_slow"]) or np.isnan(signal["atr"]):
        return None

    ema_fast_prev = df["ema_fast"].iloc[-3]
    ema_slow_prev = df["ema_slow"].iloc[-3]
    swing_high = signal["swing_high"]
    swing_low = signal["swing_low"]
    vol_ok = signal["vol_ratio"] > 1.4

    # Ищем "ранний" сигнал: недавний кросс + пробой/ретест
    recent_cross_long = (ema_fast_prev <= ema_slow_prev and signal["ema_fast"] > signal["ema_slow"])
    recent_cross_short = (ema_fast_prev >= ema_slow_prev and signal["ema_fast"] < signal["ema_slow"])

    # Пробой структуры
    bos_long = signal["close"] > swing_high if not pd.isna(swing_high) else False
    bos_short = signal["close"] < swing_low if not pd.isna(swing_low) else False

    # Ретест: цена рядом с уровнем после пробоя
    retest_band = signal["atr"] * 0.5 if signal["atr"] > 0 else signal["close"] * 0.003
    retest_long = abs(signal["close"] - swing_high) <= retest_band if not pd.isna(swing_high) else False
    retest_short = abs(signal["close"] - swing_low) <= retest_band if not pd.isna(swing_low) else False

    # Фильтр "слишком поздно": если цена далеко ушла от пробоя
    distance_from_level_long = abs(current["close"] - swing_high) if not pd.isna(swing_high) else 0
    distance_from_level_short = abs(current["close"] - swing_low) if not pd.isna(swing_low) else 0
    too_late_long = distance_from_level_long > signal["atr"] * 1.2
    too_late_short = distance_from_level_short > signal["atr"] * 1.2

    long_trigger = (recent_cross_long and (bos_long or retest_long)) and not too_late_long
    short_trigger = (recent_cross_short and (bos_short or retest_short)) and not too_late_short
    confirm_long = confirm["close"] > signal["close"]
    confirm_short = confirm["close"] < signal["close"]

    direction = ""
    reasons = []
    if long_trigger and vol_ok and confirm_long:
        direction = "LONG"
        reasons.append("Ранний сигнал LONG: EMA50 пересек EMA200 + пробой/ретест уровня")
        reasons.append("Объем подтверждает импульс (>1.2x)")
        reasons.append("Подтверждение свечой: следующая закрылась выше")
    elif short_trigger and vol_ok and confirm_short:
        direction = "SHORT"
        reasons.append("Ранний сигнал SHORT: EMA50 пересек EMA200 + пробой/ретест уровня")
        reasons.append("Объем подтверждает импульс (>1.2x)")
        reasons.append("Подтверждение свечой: следующая закрылась ниже")
    else:
        missing = []
        if not vol_ok:
            missing.append("мало объема")
        if not (long_trigger or short_trigger):
            missing.append("нет раннего разворота (кросс + пробой/ретест)")
        if (recent_cross_long or recent_cross_short) and (too_late_long or too_late_short):
            missing.append("поздний вход — цена уже далеко от уровня")
        if (long_trigger and not confirm_long) or (short_trigger and not confirm_short):
            missing.append("нет подтверждения следующей свечой")
        return {
            "enter_now": "Нет",
            "direction": "-",
            "entry": round(current["close"], 6),
            "stop": "",
            "take_profit": "",
            "rr": "",
            "reasons": "Не входить: " + ", ".join(missing),
            "signal_time": signal["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
        }

    entry = current["close"]
    atr = signal["atr"] if signal["atr"] > 0 else entry * 0.01

    # Динамический стоп: там, где тренд «сломается»
    if direction == "LONG":
        base_stop = swing_low if not pd.isna(swing_low) else entry - atr * 1.2
        stop = min(base_stop, entry - atr * 0.8) - atr * 0.3  # небольшой запас
        # Динамический тейк: ожидаем движение к следующему уровню
        recent_range = (swing_high - swing_low) if (not pd.isna(swing_high) and not pd.isna(swing_low)) else atr * 3
        take_profit = entry + max(atr * 2.0, recent_range * 0.7)
    else:
        base_stop = swing_high if not pd.isna(swing_high) else entry + atr * 1.2
        stop = max(base_stop, entry + atr * 0.8) + atr * 0.3
        recent_range = (swing_high - swing_low) if (not pd.isna(swing_high) and not pd.isna(swing_low)) else atr * 3
        take_profit = entry - max(atr * 2.0, recent_range * 0.7)

    rr = abs((take_profit - entry) / (entry - stop)) if (entry - stop) != 0 else 0
    profit_pct = abs(take_profit - entry) / entry

    # Фильтр: R/R ≥ 1:3, прибыль ≥ 5%
    if rr < MIN_RR or profit_pct < MIN_PROFIT_PCT:
        return None

    return {
        "enter_now": "Да",
        "direction": direction,
        "entry": entry,
        "stop": stop,
        "take_profit": take_profit,
        "rr": round(rr, 2),
        "reasons": " | ".join(reasons),
        "signal_time": signal["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
    }


STATE_FILE = "trend_change_state.json"


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


def cleanup_old_files(prefix="trend_change_signals_", keep_last=3):
    """Удаляет старые файлы сигналов, оставляет последние N."""
    files = []
    for name in os.listdir("."):
        if name.startswith(prefix) and name.endswith(".xlsx"):
            files.append(name)
    files.sort(reverse=True)
    for old in files[keep_last:]:
        try:
            os.remove(old)
        except Exception:
            pass


def wait_until_next_5min():
    """Ожидание до 01 секунды каждой 5-й минуты."""
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
    print("🚀 ПОИСК СМЕНЫ ТРЕНДА (MEXC, 5m)")
    print("Условия: EMA50/EMA200 + пробой swing уровней + объем")
    print("R/R минимум 1:3")

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
        signal = evaluate_trend_change(df)
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

        # Новые входы: только "Да", не повторяем старые
        if row["Вход сейчас"] == "Да":
            key = f"{symbol}|{row['Направление']}|{row['Время сигнала']}"
            if key not in state_set:
                new_entries.append(row)
                state_set.add(key)
        time.sleep(0.25)

    rows = sorted(rows, key=lambda x: (x["Вход сейчас"] != "Да", x["Символ"]))
    out_file = "trend_change_signals.xlsx"
    temp_file = "trend_change_signals_tmp.xlsx"
    try:
        with pd.ExcelWriter(temp_file, engine="openpyxl", mode="w") as writer:
            pd.DataFrame(rows).to_excel(writer, sheet_name="Сигналы", index=False)
        # Пытаемся заменить основной файл (может быть открыт)
        try:
            os.replace(temp_file, out_file)
            print(f"✅ Готово: {os.path.abspath(out_file)}")
        except PermissionError:
            ts_name = f"trend_change_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            os.replace(temp_file, ts_name)
            print("⚠️ Файл trend_change_signals.xlsx открыт. Сохранено в новый файл:")
            print(f"✅ {os.path.abspath(ts_name)}")
            cleanup_old_files()
    except PermissionError:
        print("❌ Нет доступа к файлу Excel. Закройте Excel и попробуйте снова.")

    if new_entries:
        print("\n🔔 НОВЫЕ ТОЧКИ ВХОДА:")
        for n in new_entries:
            print(f"   {n['Символ']} {n['Направление']} | Вход {n['Цена входа']} | Стоп {n['Стоп-лосс']} | Тейк {n['Тейк-профит (R≥3)']}")
        # Громкий звуковой сигнал (несколько коротких импульсов)
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
    print("⏱️ Запуск по расписанию: каждые 5 минут на 01-й секунде")
    while True:
        wait_until_next_5min()
        print(f"\n⏰ Запуск проверки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            state_set = run_once(state_set)
            save_state(state_set)
        except Exception as e:
            print(f"⚠️ Ошибка цикла: {e}")


if __name__ == "__main__":
    # По умолчанию запускаем по расписанию
    run_loop()
