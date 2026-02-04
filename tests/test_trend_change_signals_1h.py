"""
Смена тренда на ЧАСОВОМ таймфрейме (1H). Паттерны: EMA кросс, консолидация, глобальный пик, локальный пик.
Время на графике MEXC и в отчётах скрипта — UTC. У вас UTC+3: ваше время = UTC + 3 часа.
Параметры: глобальный пик 24 бар (~1 день), локальный пик — свеча выше соседей (2-й/3-й пики в зоне), консолидация 6 баров.
"""
import time
import os
import json
import winsound
import numpy as np
import pandas as pd
import requests
import talib
from datetime import datetime

INTERVAL_1H = "Min60"
LIMIT = 800
MIN_BARS = 200


def format_price(value):
    if value is None or value == "":
        return ""
    if value < 0.0001:
        return f"{value:.8f}"
    if value < 0.01:
        return f"{value:.6f}"
    if value < 1:
        return f"{value:.5f}"
    if value < 100:
        return f"{value:.4f}"
    return f"{value:.2f}"


def get_high_volume_symbols(min_volume=10_000_000):
    url = "https://contract.mexc.com/api/v1/contract/ticker"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        symbols = []
        if "data" in data:
            for item in data["data"]:
                if item.get("amount24", 0) > min_volume:
                    symbols.append({"symbol": item["symbol"], "volume_24h": item["amount24"]})
        return sorted(symbols, key=lambda x: x["volume_24h"], reverse=True)
    except Exception as e:
        print(f"Ошибка: {e}")
        return []


def get_candles(symbol, interval=INTERVAL_1H, limit=LIMIT):
    url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
    try:
        r = requests.get(url, params={"interval": interval, "limit": limit}, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"   Ошибка {symbol}: {e}")
    return None


def create_dataframe(data):
    if not data or not data.get("success") or not data.get("data"):
        return None
    raw = data["data"]
    try:
        if isinstance(raw, dict):
            req = ["time", "open", "high", "low", "close", "vol"]
            if not all(k in raw for k in req):
                return None
            n = min(len(raw[k]) for k in req)
            if n == 0:
                return None
            df = pd.DataFrame({k: raw[k][:n] for k in ["time", "open", "high", "low", "close"]})
            df["volume"] = raw["vol"][:n]
            df = df.rename(columns={"time": "timestamp"})
        else:
            rows = [row[:6] for row in raw if isinstance(row, (list, tuple)) and len(row) >= 6]
            if not rows:
                return None
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()
        if len(df) < MIN_BARS:
            return None
        ts = df["timestamp"].iloc[0]
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms" if ts > 1e12 else "s")
        df = df.sort_values("datetime").reset_index(drop=True)
        return df
    except Exception:
        return None


def add_indicators(df):
    c, h, l, v = df["close"].values, df["high"].values, df["low"].values, df["volume"].values
    df["ema_fast"] = talib.EMA(c, timeperiod=50)
    df["ema_slow"] = talib.EMA(c, timeperiod=200)
    df["ema_10"] = talib.EMA(c, timeperiod=10)
    df["ema_30"] = talib.EMA(c, timeperiod=30)
    df["atr"] = talib.ATR(h, l, c, timeperiod=14)
    df["vol_sma"] = talib.SMA(v, timeperiod=20)
    df["vol_ratio"] = v / df["vol_sma"]
    df["swing_high"] = df["high"].rolling(20).max().shift(1)
    df["swing_low"] = df["low"].rolling(20).min().shift(1)
    df["body_size"] = (df["close"] - df["open"]).abs()
    df["candle_range"] = (df["high"] - df["low"]).replace(0, 1e-9)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["rsi"] = talib.RSI(c, timeperiod=14)
    # MACD: сдвиг импульса (фильтр ложных сигналов)
    macd, macd_signal, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    # ADX: сила тренда (вход только при ADX > 20 — был тренд перед разворотом)
    df["adx"] = talib.ADX(h, l, c, timeperiod=14)
    df["plus_di"] = talib.PLUS_DI(h, l, c, timeperiod=14)
    df["minus_di"] = talib.MINUS_DI(h, l, c, timeperiod=14)
    return df


def evaluate_trend_change_1h(df, strict_peaks=False, peak_stop_atr_mult=0.2, strict_rsi_only=False,
                            use_confluence=False, min_confluence=3):
    """
    Оценка смены тренда на 1H.
    use_confluence: вход только при совпадении нескольких индикаторов (MACD, RSI, объём, ADX) — выше % прибыльных.
    min_confluence: минимум подтверждений из 4 (паттерн уже есть; RSI; MACD; объём; ADX).
    """
    if len(df) < 210:
        return None
    signal = df.iloc[-2]
    confirm = df.iloc[-1]
    current = df.iloc[-1]
    if np.isnan(signal["ema_fast"]) or np.isnan(signal["ema_slow"]) or np.isnan(signal["atr"]):
        return None

    swing_high = signal["swing_high"]
    swing_low = signal["swing_low"]
    vol_ok = signal["vol_ratio"] > 1.4
    ema_fast_prev = df["ema_fast"].iloc[-3]
    ema_slow_prev = df["ema_slow"].iloc[-3]
    recent_cross_long = ema_fast_prev <= ema_slow_prev and signal["ema_fast"] > signal["ema_slow"]
    recent_cross_short = ema_fast_prev >= ema_slow_prev and signal["ema_fast"] < signal["ema_slow"]
    bos_long = signal["close"] > swing_high if not pd.isna(swing_high) else False
    bos_short = signal["close"] < swing_low if not pd.isna(swing_low) else False
    atr_band = signal["atr"] * 0.5 if signal["atr"] > 0 else signal["close"] * 0.003
    retest_long = abs(signal["close"] - swing_high) <= atr_band if not pd.isna(swing_high) else False
    retest_short = abs(signal["close"] - swing_low) <= atr_band if not pd.isna(swing_low) else False
    too_late_long = abs(current["close"] - swing_high) > signal["atr"] * 1.2 if not pd.isna(swing_high) else False
    too_late_short = abs(current["close"] - swing_low) > signal["atr"] * 1.2 if not pd.isna(swing_low) else False
    long_trigger = recent_cross_long and (bos_long or retest_long) and not too_late_long
    short_trigger = recent_cross_short and (bos_short or retest_short) and not too_late_short
    confirm_long = confirm["close"] > signal["close"]
    confirm_short = confirm["close"] < signal["close"]

    consol_len = 6
    if len(df) >= 210 + consol_len:
        consol_high = df["high"].iloc[-consol_len - 2 : -2].max()
        consol_low = df["low"].iloc[-consol_len - 2 : -2].min()
        range_c = consol_high - consol_low if consol_high > consol_low else 0
        atr_val = signal["atr"] if signal["atr"] > 0 else signal["close"] * 0.01
        is_consol = range_c <= atr_val * 1.0 and range_c > 0
        vol_ok_c = signal["vol_ratio"] > 1.0
        low_old = df["low"].iloc[-consol_len - 2 - 20 : -consol_len - 2].min() if len(df) >= consol_len + 22 else None
        had_drop = low_old is not None and not pd.isna(low_old) and consol_low <= low_old * 1.005
        breakout_up = (confirm["close"] > consol_high) and (confirm["close"] > signal["close"])
        long_consol = is_consol and had_drop and breakout_up and vol_ok_c
        breakout_down = (confirm["close"] < consol_low) and (confirm["close"] < signal["close"])
        ema30 = confirm["ema_30"]
        trend_broken = (not pd.isna(ema30)) and (confirm["close"] < ema30)
        recent_h = df["high"].iloc[-30:-2].max() if len(df) >= 30 else consol_high
        was_near_top = consol_high >= recent_h * 0.99
        short_consol = is_consol and was_near_top and breakout_down and trend_broken and vol_ok_c
    else:
        long_consol = short_consol = False
        consol_high = consol_low = np.nan

    # LONG: старт восходящего импульса — отбой от EMA30 или пробой неширокого диапазона (ловим начало движений)
    lookback_bounce = 6
    if len(df) >= lookback_bounce + 2:
        atr_b = signal["atr"] if signal["atr"] > 0 else signal["close"] * 0.01
        # Откат к EMA30 в последних 3 барах (свежий отбой), не 6 — чтобы не ловить ложные отбои после падения
        recent_lows_3 = df["low"].iloc[-5 : -2].values  # 3 бара: -5,-4,-3
        recent_ema30_3 = df["ema_30"].iloc[-5 : -2].values
        pullback_near_ema = False
        for j in range(len(recent_lows_3)):
            if not pd.isna(recent_ema30_3[j]) and abs(recent_lows_3[j] - recent_ema30_3[j]) <= atr_b * 0.6:
                pullback_near_ema = True
                break
        # Не входим в лонг, если только что было сильное падение (лоу последних 6 бар сильно ниже 12 бар назад)
        low_6 = df["low"].iloc[-lookback_bounce - 2 : -2].min()
        no_crash = True
        if len(df) >= 14:
            close_12_ago = df["close"].iloc[-14]
            no_crash = low_6 >= close_12_ago * 0.98
        confirm_bullish = confirm["close"] > confirm["open"]
        confirm_above_signal_high = confirm["close"] > signal["high"]
        confirm_above_ema30 = (not pd.isna(confirm["ema_30"])) and (confirm["close"] > confirm["ema_30"])
        # Только первый бар пробоя: сигнальная свеча ещё не была выше хайка 3 баров назад (не дублируем вход в одном движении)
        signal_not_breakout_yet = signal["close"] <= (df["high"].iloc[-4] if len(df) >= 5 else 0) * 1.003
        # В восходящем микротренде (EMA10 > EMA30) — не ловим отбой после уже начавшегося падения
        uptrend_short = (not pd.isna(signal["ema_10"])) and (not pd.isna(signal["ema_30"])) and (signal["ema_10"] > signal["ema_30"])
        # Не покупаем отбой, если сигнальная свеча уже перекуплена (RSI >= 65) или слабый отбой (RSI < 55)
        rsi_not_overbought = pd.isna(signal["rsi"]) or signal["rsi"] < 65
        rsi_not_weak = pd.isna(signal["rsi"]) or signal["rsi"] >= 55
        # Тренд вверх по DMI и достаточная сила (ADX) — меньше ложных отбоев в боковике
        plus_di = signal.get("plus_di")
        minus_di = signal.get("minus_di")
        dmi_bull = (pd.isna(plus_di) or pd.isna(minus_di)) or (plus_di > minus_di)
        adx_ok = (pd.isna(signal.get("adx"))) or (signal["adx"] >= 18)
        long_bounce = (
            pullback_near_ema and no_crash and uptrend_short and rsi_not_overbought and rsi_not_weak and confirm_bullish
            and confirm_above_signal_high and confirm_above_ema30
            and signal["vol_ratio"] > 0.9 and signal_not_breakout_yet and dmi_bull and adx_ok
        )
        # Пробой диапазона вверх (без требования "после падения") — старт движения из флэта
        range_high = df["high"].iloc[-lookback_bounce - 2 : -2].max()
        range_low = df["low"].iloc[-lookback_bounce - 2 : -2].min()
        range_size = range_high - range_low if range_high > range_low else 0
        is_tight_range = 0 < range_size <= atr_b * 1.3
        # Первый бар пробоя: сигнальная свеча внутри диапазона
        signal_inside_range = signal["high"] <= range_high * 1.002
        breakout_range_up = confirm["close"] > range_high and confirm["close"] > signal["close"] and signal["vol_ratio"] > 0.9
        long_range_breakout = is_tight_range and breakout_range_up and (not long_consol) and signal_inside_range
    else:
        long_bounce = long_range_breakout = False
        range_high = range_low = np.nan

    lookback_peak = 24
    need_bars = lookback_peak + 2
    if len(df) >= need_bars:
        recent_high_peak = df["high"].iloc[-need_bars:-2].max()
        low_before = df["low"].iloc[-need_bars:-2].min()
        signal_is_peak = signal["high"] >= recent_high_peak * 0.998
        run_up_pct = (signal["high"] - low_before) / low_before if low_before > 0 else 0
        significant_run_up = run_up_pct >= 0.28  # 28% рост за 24 бара — отсекает первый мелкий пик 27.01, оставляет 28.01
    else:
        signal_is_peak = significant_run_up = False
    break_below = (confirm["close"] < signal["low"]) and (confirm["close"] < signal["close"])
    vol_ok_peak = signal["vol_ratio"] > 1.0
    # Пиковая свеча медвежья (close < open) — разворот уже на пике, меньше ложных первых откатов
    peak_candle_bearish = signal["close"] < signal["open"]
    short_peak = signal_is_peak and significant_run_up and break_below and vol_ok_peak and peak_candle_bearish

    # Локальный пик: свеча-максимум относительно соседей (левый и правый бар) — ловит 2-й и 3-й пики в зоне
    if len(df) >= 4 and not short_peak:
        prev_high = df["high"].iloc[-3]
        next_high = confirm["high"]
        signal_is_swing_high = signal["high"] >= prev_high * 0.998 and signal["high"] >= next_high * 0.998
        # Бар до пика был медвежий — пик как отбой, а не продолжение роста (меньше ложных SHORT)
        prev_bar_bearish = df.iloc[-3]["close"] < df.iloc[-3]["open"]
        # небольшой рост перед пиком: минимум за последние 6 баров до сигнальной
        lookback_run = 6
        if len(df) >= lookback_run + 2:
            low_before = df["low"].iloc[-lookback_run - 2 : -2].min()
            run_up_local = (signal["high"] - low_before) / low_before if low_before > 0 else 0
            significant_run_local = run_up_local >= 0.005  # 0.5%
        else:
            significant_run_local = False
        short_local_peak = (
            signal_is_swing_high and significant_run_local and break_below and vol_ok_peak and prev_bar_bearish
        )
    else:
        short_local_peak = False
        prev_bar_bearish = False

    sl_wick = signal["lower_wick"] / signal["candle_range"] if signal["candle_range"] > 0 else 0
    cl_wick = confirm["lower_wick"] / confirm["candle_range"] if confirm["candle_range"] > 0 else 0
    strong_bottom = sl_wick > 0.60 or cl_wick > 0.60
    if short_peak and strong_bottom:
        short_peak = False
    # Strict: только пики с перекупленностью и явным разворотом — отсекаем сделки, которые уходят в SL
    if strict_peaks or strict_rsi_only:
        rsi_min = 55 if strict_peaks else 60
        rsi_ob = (not pd.isna(signal["rsi"])) and signal["rsi"] >= rsi_min
        if strict_peaks:
            confirm_below_ema30 = (not pd.isna(confirm["ema_30"])) and (confirm["close"] < confirm["ema_30"])
            confirm_bearish = confirm["close"] < confirm["open"]
            strict_ok = rsi_ob and confirm_below_ema30 and confirm_bearish
        else:
            strict_ok = rsi_ob  # strict_rsi_only: только RSI
        if short_peak and not strict_ok:
            short_peak = False
        if short_local_peak and not strict_ok:
            short_local_peak = False
    # для локального пика сильный низ не отменяем — иначе теряем второй/третий пики
    su_wick = signal["upper_wick"] / signal["candle_range"] if signal["candle_range"] > 0 else 0
    cu_wick = confirm["upper_wick"] / confirm["candle_range"] if confirm["candle_range"] > 0 else 0
    at_top = su_wick > 0.45 or cu_wick > 0.45
    if long_consol and at_top:
        long_consol = False
    if short_consol and strong_bottom:
        short_consol = False

    direction = ""
    reasons = []
    from_consol = from_peak = from_bounce = False
    use_range_stop = False
    if long_consol:
        direction = "LONG"
        from_consol = True
        reasons.append("LONG 1H: консолидация после падения, выход вверх")
    elif long_bounce:
        direction = "LONG"
        from_bounce = True
        reasons.append("LONG 1H: отбой от EMA30, старт восходящего импульса")
    elif long_range_breakout:
        direction = "LONG"
        from_bounce = True
        use_range_stop = True
        reasons.append("LONG 1H: пробой диапазона вверх, старт движения")
    elif short_consol:
        direction = "SHORT"
        from_consol = True
        reasons.append("SHORT 1H: консолидация у верха, пробой вниз")
    elif short_peak:
        direction = "SHORT"
        from_peak = True
        reasons.append("SHORT 1H: глобальный пик (24ч макс, рост >=28%), закрытие ниже низа пиковой")
    elif short_local_peak:
        direction = "SHORT"
        from_peak = True
        reasons.append("SHORT 1H: локальный пик (макс по соседним барам, рост >=0.5%), пробой вниз")
    elif long_trigger and vol_ok and confirm_long:
        direction = "LONG"
        reasons.append("LONG 1H: EMA50/200 кросс + пробой/ретест")
    elif short_trigger and vol_ok and confirm_short:
        direction = "SHORT"
        reasons.append("SHORT 1H: EMA50/200 кросс + пробой/ретест")
    if not direction:
        return {
            "enter_now": "Нет", "direction": "-", "entry": round(current["close"], 6),
            "stop": "", "take_profit": "", "rr": "",
            "reasons": "Не входить", "signal_time": signal["datetime"].strftime("%Y-%m-%d %H:%M"),
        }

    # Конфлюэнс: вход только при совпадении нескольких индикаторов (MACD, RSI, объём, ADX, DMI)
    if use_confluence:
        conf = 1  # паттерн уже совпал
        rsi_ok = not pd.isna(signal["rsi"])
        macd_ok = not pd.isna(confirm.get("macd")) and not pd.isna(confirm.get("macd_signal"))
        vol_ok_cf = (not pd.isna(signal["vol_ratio"])) and signal["vol_ratio"] >= 1.2
        adx_ok = (not pd.isna(signal["adx"])) and signal["adx"] >= 18
        plus_di = signal.get("plus_di")
        minus_di = signal.get("minus_di")
        dmi_ok = not pd.isna(plus_di) and not pd.isna(minus_di)
        if direction == "SHORT":
            if rsi_ok and signal["rsi"] >= 62:
                conf += 1
            if macd_ok and confirm["macd"] <= confirm["macd_signal"]:
                conf += 1
            if vol_ok_cf:
                conf += 1
            if adx_ok:
                conf += 1
            if dmi_ok and minus_di > plus_di:
                conf += 1
        else:
            if rsi_ok and signal["rsi"] < 70:
                conf += 1
            if macd_ok and confirm["macd"] >= confirm["macd_signal"]:
                conf += 1
            if vol_ok_cf:
                conf += 1
            if adx_ok:
                conf += 1
            if dmi_ok and plus_di > minus_di:
                conf += 1
            elif dmi_ok and minus_di >= plus_di:
                conf = 0
        if conf < min_confluence:
            reasons.append(f"[конфлюэнс {conf}<{min_confluence}]")
            return {
                "enter_now": "Нет", "direction": direction, "entry": round(current["close"], 6),
                "stop": "", "take_profit": "", "rr": "",
                "reasons": " | ".join(reasons), "signal_time": signal["datetime"].strftime("%Y-%m-%d %H:%M"),
            }
        reasons.append(f"[конфлюэнс {conf}]")

    entry = current["close"]
    atr = signal["atr"] if signal["atr"] > 0 else entry * 0.01
    if from_peak:
        stop = signal["high"] + atr * peak_stop_atr_mult
        take_profit = entry - (stop - entry) * 2.5
    elif from_consol and not (pd.isna(consol_high) or pd.isna(consol_low)):
        if direction == "LONG":
            stop = consol_low - atr * 0.25
            take_profit = entry + (entry - stop) * 2.5
        else:
            stop = consol_high + atr * 0.25
            take_profit = entry - (stop - entry) * 2.5
    elif from_bounce and direction == "LONG":
        if use_range_stop and not (pd.isna(range_low)):
            stop = range_low - atr * 0.25
        else:
            stop = df["low"].iloc[-lookback_bounce - 2 : -2].min() - atr * 0.3
        stop = min(stop, entry - atr * 0.5)
        take_profit = entry + (entry - stop) * 2.5
    elif direction == "LONG":
        base = swing_low if not pd.isna(swing_low) else entry - atr * 1.2
        stop = min(base, entry - atr * 0.8) - atr * 0.3
        rng = (swing_high - swing_low) if (not pd.isna(swing_high) and not pd.isna(swing_low)) else atr * 3
        take_profit = entry + max(atr * 2.0, rng * 0.7)
    else:
        base = swing_high if not pd.isna(swing_high) else entry + atr * 1.2
        stop = max(base, entry + atr * 0.8) + atr * 0.3
        rng = (swing_high - swing_low) if (not pd.isna(swing_high) and not pd.isna(swing_low)) else atr * 3
        take_profit = entry - max(atr * 2.0, rng * 0.7)
    rr = abs((take_profit - entry) / (entry - stop)) if (entry - stop) != 0 else 0
    return {
        "enter_now": "Да", "direction": direction, "entry": entry, "stop": stop,
        "take_profit": take_profit, "rr": round(rr, 2), "reasons": " | ".join(reasons),
        "signal_time": signal["datetime"].strftime("%Y-%m-%d %H:%M"),
    }


def evaluate_levels_1h(df):
    """
    Режим «по уровням»: вход только от уровней (свинг high/low) и зоны Fib 0.618.
    Цель — 7–9 сделок с только TP: 2–3 LONG (отбой от уровня/Fib), 5–6 SHORT (от уровня-пика).
    Время на графике — UTC.
    """
    if len(df) < 210:
        return None
    signal = df.iloc[-2]
    confirm = df.iloc[-1]
    current = df.iloc[-1]
    atr = signal["atr"] if signal["atr"] > 0 else signal["close"] * 0.01
    # Уровни: свинг за 24 и 50 баров (для Fib — последний импульс вверх)
    look = 26
    swing_high_24 = df["high"].iloc[-look:-2].max()
    swing_low_24 = df["low"].iloc[-look:-2].min()
    look50 = 52
    if len(df) >= look50:
        seg_high = df["high"].iloc[-look50:-2].max()
        seg_low = df["low"].iloc[-look50:-2].min()
        range_up = seg_high - seg_low
        fib_0618 = seg_high - 0.618 * range_up if range_up > 0 else seg_low
        fib_0786 = seg_high - 0.786 * range_up if range_up > 0 else seg_low
    else:
        fib_0618 = fib_0786 = np.nan
    # --- LONG: отбой от уровня или от зоны Fib 0.618 / 0.5 ---
    touch_zone = False
    if not pd.isna(fib_0618):
        recent_lows = df["low"].iloc[-6:-2].values
        for low_val in recent_lows:
            if low_val <= fib_0618 * 1.012 and low_val >= fib_0618 * 0.988:
                touch_zone = True
                break
    if not touch_zone and len(df) >= look50 and range_up > 0:
        fib_05 = seg_high - 0.5 * range_up
        for low_val in df["low"].iloc[-6:-2].values:
            if low_val <= fib_05 * 1.012 and low_val >= fib_05 * 0.988:
                touch_zone = True
                break
    touch_support = (df["low"].iloc[-6:-2].min() <= swing_low_24 * 1.01) if swing_low_24 > 0 else False
    confirm_bull = confirm["close"] > confirm["open"] and confirm["close"] > signal["high"]
    vol_ok = (not pd.isna(signal["vol_ratio"])) and signal["vol_ratio"] >= 1.0
    rsi_ok_long = (pd.isna(signal["rsi"])) or (signal["rsi"] < 68)
    macd_bull = (not pd.isna(confirm.get("macd")) and not pd.isna(confirm.get("macd_signal")) and
                 confirm["macd"] >= confirm["macd_signal"])
    plus_di = signal.get("plus_di")
    minus_di = signal.get("minus_di")
    trend_ok = (pd.isna(plus_di) or pd.isna(minus_di)) or (plus_di > minus_di)
    long_fib = touch_zone and confirm_bull and vol_ok and rsi_ok_long and trend_ok
    long_support = touch_support and confirm_bull and vol_ok and rsi_ok_long and trend_ok
    long_level = long_fib or long_support
    # LONG: отбой от EMA30 (как в паттернах тренда, но с уровнями стопа)
    recent_lows_3 = df["low"].iloc[-5:-2].values
    recent_ema30_3 = df["ema_30"].iloc[-5:-2].values
    pullback_near_ema = False
    for j in range(len(recent_lows_3)):
        if not pd.isna(recent_ema30_3[j]) and abs(recent_lows_3[j] - recent_ema30_3[j]) <= atr * 0.6:
            pullback_near_ema = True
            break
    ema10_gt_30 = (not pd.isna(signal["ema_10"])) and (not pd.isna(signal["ema_30"])) and (signal["ema_10"] > signal["ema_30"])
    rsi_55_68 = (pd.isna(signal["rsi"])) or (55 <= signal["rsi"] < 68)
    long_ema_bounce = (pullback_near_ema and ema10_gt_30 and rsi_55_68 and confirm_bull and vol_ok and
                        trend_ok and (pd.isna(signal.get("adx")) or signal["adx"] >= 18))
    long_level = long_level or long_ema_bounce
    # --- SHORT: от уровня (свинг-хай), пробой вниз, пиковая свеча медвежья ---
    at_high = signal["high"] >= swing_high_24 * 0.998
    run_low = df["low"].iloc[-look:-2].min()
    run_up_pct = (signal["high"] - run_low) / run_low if run_low > 0 else 0
    break_down = confirm["close"] < signal["low"]  # закрытие ниже низа сигнальной свечи = пробой вниз
    rsi_ok_short = (not pd.isna(signal["rsi"])) and signal["rsi"] >= 60
    below_ema30 = (not pd.isna(confirm["ema_30"])) and (confirm["close"] < confirm["ema_30"])
    peak_bearish = signal["close"] < signal["open"]
    short_level = (at_high and break_down and run_up_pct >= 0.08 and vol_ok and rsi_ok_short and
                   below_ema30 and peak_bearish)
    # Локальный пик (второй/третий пик в зоне): свеча выше соседей, пиковая свеча медвежья
    if not short_level and len(df) >= 4:
        prev_high = df["high"].iloc[-3]
        next_high = confirm["high"]
        is_swing = signal["high"] >= prev_high * 0.998 and signal["high"] >= next_high * 0.998
        run_6 = df["low"].iloc[-8:-2].min()
        run_local = (signal["high"] - run_6) / run_6 if run_6 > 0 else 0
        short_local = (is_swing and run_local >= 0.008 and break_down and vol_ok and
                       rsi_ok_short and below_ema30 and peak_bearish)
    else:
        short_local = False
    direction = ""
    reasons = []
    if long_level:
        direction = "LONG"
        reasons.append("LONG 1H: отбой от EMA30 (уровни)" if long_ema_bounce and not (long_fib or long_support) else "LONG 1H: отбой от уровня/Fib 0.618, подтверждение объём+MACD")
    elif short_level:
        direction = "SHORT"
        reasons.append("SHORT 1H: от уровня (свинг-хай), пробой вниз, закрытие ниже EMA30")
    elif short_local:
        direction = "SHORT"
        reasons.append("SHORT 1H: локальный пик у уровня, пробой вниз")
    if not direction:
        return {
            "enter_now": "Нет", "direction": "-", "entry": round(current["close"], 6),
            "stop": "", "take_profit": "", "rr": "",
            "reasons": "Уровень не совпал", "signal_time": signal["datetime"].strftime("%Y-%m-%d %H:%M"),
        }
    entry = current["close"]
    if direction == "LONG":
        stop = (min(df["low"].iloc[-8:-2].min(), fib_0618) - atr * 0.3) if not pd.isna(fib_0618) else df["low"].iloc[-8:-2].min() - atr * 0.3
        stop = min(stop, entry - atr * 0.5)
        take_profit = entry + (entry - stop) * 2.5
    else:
        stop = signal["high"] + atr * 0.4
        take_profit = entry - (stop - entry) * 2.5
    rr = abs((take_profit - entry) / (entry - stop)) if (entry - stop) != 0 else 0
    return {
        "enter_now": "Да", "direction": direction, "entry": entry, "stop": stop,
        "take_profit": take_profit, "rr": round(rr, 2), "reasons": " | ".join(reasons),
        "signal_time": signal["datetime"].strftime("%Y-%m-%d %H:%M"),
    }


STATE_FILE_1H = "trend_change_state_1h.json"


def load_state():
    if os.path.exists(STATE_FILE_1H):
        try:
            with open(STATE_FILE_1H, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()


def save_state(state_set):
    with open(STATE_FILE_1H, "w", encoding="utf-8") as f:
        json.dump(sorted(list(state_set)), f, ensure_ascii=False, indent=2)


def run_once(state_set, symbols_list=None):
    print("Смена тренда 1H (MEXC)")
    symbols = symbols_list or [s["symbol"] for s in get_high_volume_symbols(10_000_000)[:10]]
    rows = []
    new_entries = []
    for symbol in symbols:
        data = get_candles(symbol, INTERVAL_1H, LIMIT)
        df = create_dataframe(data)
        if df is None:
            continue
        df = add_indicators(df)
        res = evaluate_trend_change_1h(df)
        if not res or res.get("enter_now") != "Да":
            continue
        row = {
            "Символ": symbol,
            "Направление": res["direction"],
            "Время сигнала": res["signal_time"],
            "Вход": format_price(res["entry"]),
            "Стоп": format_price(res["stop"]),
            "Тейк": format_price(res["take_profit"]),
            "R/R": res["rr"],
            "Причины": res["reasons"],
        }
        rows.append(row)
        key = f"{symbol}|{res['direction']}|{res['signal_time']}"
        if key not in state_set:
            new_entries.append(row)
            state_set.add(key)
        time.sleep(0.2)
    out = "trend_change_signals_1h.xlsx"
    try:
        pd.DataFrame(rows).to_excel(out, index=False, sheet_name="1H")
        print(f"Сохранено: {os.path.abspath(out)}")
    except Exception as e:
        print(f"Excel: {e}")
    if new_entries:
        for n in new_entries:
            print(f"  {n['Символ']} {n['Направление']} | {n['Вход']} | {n['Стоп']} | {n['Тейк']}")
        try:
            winsound.Beep(1200, 400)
        except Exception:
            pass
    return state_set


if __name__ == "__main__":
    state = load_state()
    run_once(state)
    save_state(state)
