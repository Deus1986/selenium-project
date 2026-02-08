"""
MEXC 5m backtest. Multiple entry strategies (CONFIG["STRATEGY"]):
  trend_change  - EMA 20/50 cross + BOS/retest + volume + RSI (fewer entries, was +6 R on LINK 60d).
  pullback_ema  - Pullback to EMA20 in trend; many entries (2000+ on LINK 60d), needs filters.
  breakout_10   - 10-bar high/low breakout; enter next open. Many entries.
  ema9_21       - EMA 9/21 cross + confirm + RSI + volume; more crosses than 20/50.
R:R 1:3. Results written to backtest_5m_strategy_results.txt when using 60 STRATEGIES.

Run:  python test-scripts/backtest_5m_trend_mexc.py [DAYS] [SYMBOL]
      60 LINKUSDT | 60 ALL | 60 TUNE | 60 TUNE QUICK | 60 STRATEGIES [max_coins]
      60 LINKUSDT TIMEFRAMES  -> compare 1m, 5m, 15m, 30m, 1h on one symbol
"""
TARGET_R = 500
import sys
import os
import time
import numpy as np
import pandas as pd
import requests
import talib
from datetime import datetime, timedelta

INTERVAL = "Min5"
INTERVAL_MINUTES = {"Min1": 1, "Min5": 5, "Min15": 15, "Min30": 30, "Min60": 60}
RR_RATIO = 3.0   # take-profit at 3R; SL = 0.5R (CONFIG["SL_R"])
MIN_BARS = 220
CHUNK_SIZE = 2000
API_TIMEOUT = 120
API_RETRIES = 3
EMA_FAST = 20
EMA_SLOW = 50

# Config. STRATEGY: "hybrid" | "trend_change" | "pullback_ema" | "breakout_10" | "ema9_21"
# Stricter = fewer deals, fewer stop-losses. (More strict: VOL 1.55, COOLDOWN 20 -> very few trades.)
CONFIG = {
    "STRATEGY": "hybrid",
    "VOL_MIN": 1.45,
    "RSI_LONG_MAX": 55,
    "RSI_SHORT_MIN": 45,
    "USE_ADX": False,
    "ADX_MIN": 20,
    "STOP_ATR_MUL": 1.2,
    "COOLDOWN_BARS": 16,
    "STRONG_BAR_PCT": 0.55,
    "BOS_ONLY": False,
    "SL_R": 0.3,
    "SKIP_WEEKENDS": False,  # trade all days (set True to exclude Sat/Sun)
}
# SL_R: loss in R per stop (1.0 = -1R per SL). 0.5 = each SL = -0.5 R (tighter stop accounting).

# Temporary exclude: min order > $5 (for $5 position / 5x). Remove from list to re-enable.
EXCLUDED_SYMBOLS = {
    "PEPE_USDT", "ETH_USDT", "ETH_USDC", "XRP_USD", "DOGE_USDT", "SOL_USDT", "SOL_USDC",
    "BTC_USDT", "BTC_USDC", "BNB_USDT", "NIGHT_USDT", "BCH_USDT",
    "CHZ_USDT", "ZEC_USDT", "XMR_USDT",  # many SL, few TP (backtest 30d)
}


def format_price(v):
    if v is None or v == "": return ""
    if v < 0.01: return f"{v:.6f}"
    if v < 1: return f"{v:.5f}"
    if v < 100: return f"{v:.4f}"
    return f"{v:.2f}"


def get_high_volume_symbols(min_volume=10_000_000):
    url = "https://contract.mexc.com/api/v1/contract/ticker"
    for attempt in range(API_RETRIES):
        try:
            r = requests.get(url, timeout=API_TIMEOUT)
            data = r.json()
            symbols = []
            if "data" in data:
                for item in data["data"]:
                    if item.get("amount24", 0) > min_volume:
                        sym = item["symbol"]
                        if sym not in EXCLUDED_SYMBOLS:
                            symbols.append({"symbol": sym, "volume_24h": item["amount24"]})
            return sorted(symbols, key=lambda x: x["volume_24h"], reverse=True)
        except Exception as e:
            if attempt < API_RETRIES - 1:
                time.sleep(5)
                continue
            return []
    return []


def get_candles_chunk(symbol, interval, end_ts_sec=None, limit=2000):
    url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
    params = {"interval": interval, "limit": limit}
    if end_ts_sec is not None:
        params["end"] = int(end_ts_sec)
    for attempt in range(API_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=API_TIMEOUT)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            if attempt < API_RETRIES - 1:
                time.sleep(5)
                continue
            print(f"  API error: {e}")
    return None


def fetch_history(symbol, days, interval=None):
    """Fetch candles for last `days` days. interval: Min1, Min5, Min15, Min30, Min60. MEXC max 2000 per request."""
    interval = interval or INTERVAL
    mins = INTERVAL_MINUTES.get(interval, 5)
    need_bars = days * 24 * (60 // mins)
    all_data = {"time": [], "open": [], "high": [], "low": [], "close": [], "vol": []}
    end_ts = None
    total = 0
    while total < need_bars:
        data = get_candles_chunk(symbol, interval, end_ts_sec=end_ts, limit=CHUNK_SIZE)
        if not data or not data.get("success") or not data.get("data"):
            break
        raw = data["data"]
        if isinstance(raw, dict) and "time" in raw:
            n = min(len(raw["time"]), len(raw["open"]), len(raw["high"]), len(raw["low"]), len(raw["close"]), len(raw["vol"]))
            if n == 0:
                break
            all_data["time"].extend(raw["time"][:n])
            all_data["open"].extend(raw["open"][:n])
            all_data["high"].extend(raw["high"][:n])
            all_data["low"].extend(raw["low"][:n])
            all_data["close"].extend(raw["close"][:n])
            all_data["vol"].extend(raw["vol"][:n])
            total += n
            if n < CHUNK_SIZE:
                break
            first_ts = raw["time"][0]
            if isinstance(first_ts, (int, float)) and first_ts > 1e12:
                first_ts = first_ts // 1000
            end_ts = int(first_ts) - 1
        else:
            break
        time.sleep(0.2)
    if not all_data["time"]:
        return None
    df = pd.DataFrame(all_data)
    df = df.rename(columns={"vol": "volume"})
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    ts = df["time"].iloc[0]
    df["datetime"] = pd.to_datetime(df["time"], unit="ms" if ts > 1e12 else "s")
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)
    return df


def add_indicators(df):
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    v = df["volume"].values
    df["ema_fast"] = talib.EMA(c, timeperiod=EMA_FAST)
    df["ema_slow"] = talib.EMA(c, timeperiod=EMA_SLOW)
    df["atr"] = talib.ATR(h, l, c, timeperiod=14)
    df["vol_sma"] = talib.SMA(v, timeperiod=20)
    df["vol_ratio"] = np.where(df["vol_sma"] > 0, v / df["vol_sma"].values, 1.0)
    df["swing_high"] = df["high"].rolling(20).max().shift(1)
    df["swing_low"] = df["low"].rolling(20).min().shift(1)
    df["rsi"] = talib.RSI(c, timeperiod=14)
    df["ema_9"] = talib.EMA(c, timeperiod=9)
    df["ema_21"] = talib.EMA(c, timeperiod=21)
    df["ema_20"] = talib.EMA(c, timeperiod=20)
    if CONFIG.get("USE_ADX", False):
        df["adx"] = talib.ADX(h, l, c, timeperiod=14)
    # Monday=0 .. Sunday=6; Saturday=5, Sunday=6
    df["is_weekend"] = pd.to_datetime(df["datetime"]).dt.weekday >= 5
    return df


def _make_signal(i, direction, entry, stop, tp, signal_time):
    return {"bar_index": i, "direction": direction, "entry": entry, "stop": stop, "take_profit": tp, "signal_time": signal_time}


def get_signals_pullback_ema(df):
    """More entries: pullback to EMA20 in trend. Long when close>EMA20, low touched EMA20, bullish bar. Enter next open. R:R 1:3."""
    if df is None or len(df) < 80:
        return []
    signals = []
    last_long, last_short = -99, -99
    cooldown = CONFIG.get("COOLDOWN_BARS", 5)
    atr_mul = CONFIG.get("STOP_ATR_MUL", 1.2)
    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        ema = row["ema_20"]
        if np.isnan(ema) or row["atr"] <= 0:
            continue
        atr = float(row["atr"])
        open_next = float(df["open"].iloc[i + 1])
        dt = df["datetime"].iloc[i]
        signal_time = dt.strftime("%Y-%m-%d %H:%M") if hasattr(dt, "strftime") else str(dt)
        # Long: close > EMA20, low <= EMA20 + 0.2% (touched), bullish candle
        if row["close"] > ema and row["low"] <= ema * 1.002 and row["close"] > row["open"]:
            if i - last_long < cooldown:
                continue
            stop = float(row["low"])
            if open_next <= stop:
                continue
            risk = open_next - stop
            if risk < atr * 0.3:
                continue
            if CONFIG.get("SKIP_WEEKENDS") and i + 1 < len(df) and df["is_weekend"].iloc[i + 1]:
                continue
            tp = open_next + risk * RR_RATIO
            signals.append(_make_signal(i, "LONG", open_next, stop, tp, signal_time))
            last_long = i
        # Short: close < EMA20, high >= EMA20 - 0.2%, bearish candle
        elif row["close"] < ema and row["high"] >= ema * 0.998 and row["close"] < row["open"]:
            if i - last_short < cooldown:
                continue
            stop = float(row["high"])
            if open_next >= stop:
                continue
            risk = stop - open_next
            if risk < atr * 0.3:
                continue
            if CONFIG.get("SKIP_WEEKENDS") and i + 1 < len(df) and df["is_weekend"].iloc[i + 1]:
                continue
            tp = open_next - risk * RR_RATIO
            signals.append(_make_signal(i, "SHORT", open_next, stop, tp, signal_time))
            last_short = i
    return signals


def get_signals_breakout_10(df):
    """More entries: 10-bar breakout. New close above 10-bar high = long, below 10-bar low = short. Enter next open. R:R 1:3."""
    lookback = 10
    if df is None or len(df) < lookback + 30:
        return []
    signals = []
    last_long, last_short = -99, -99
    cooldown = CONFIG.get("COOLDOWN_BARS", 4)
    for i in range(lookback, len(df) - 1):
        high_10 = df["high"].iloc[i - lookback : i].max()
        low_10 = df["low"].iloc[i - lookback : i].min()
        prev_high = df["high"].iloc[i - lookback : i - 1].max() if i > lookback else high_10
        prev_low = df["low"].iloc[i - lookback : i - 1].min() if i > lookback else low_10
        close_i = float(df["close"].iloc[i])
        open_next = float(df["open"].iloc[i + 1])
        low_i = float(df["low"].iloc[i])
        high_i = float(df["high"].iloc[i])
        dt = df["datetime"].iloc[i]
        signal_time = dt.strftime("%Y-%m-%d %H:%M") if hasattr(dt, "strftime") else str(dt)
        if close_i > high_10 and (i == lookback or df["close"].iloc[i - 1] <= prev_high):
            if i - last_long < cooldown:
                continue
            stop = low_i
            if open_next <= stop:
                continue
            risk = open_next - stop
            if CONFIG.get("SKIP_WEEKENDS") and i + 1 < len(df) and df["is_weekend"].iloc[i + 1]:
                continue
            tp = open_next + risk * RR_RATIO
            signals.append(_make_signal(i, "LONG", open_next, stop, tp, signal_time))
            last_long = i
        elif close_i < low_10 and (i == lookback or df["close"].iloc[i - 1] >= prev_low):
            if i - last_short < cooldown:
                continue
            stop = high_i
            if open_next >= stop:
                continue
            risk = stop - open_next
            if CONFIG.get("SKIP_WEEKENDS") and i + 1 < len(df) and df["is_weekend"].iloc[i + 1]:
                continue
            tp = open_next - risk * RR_RATIO
            signals.append(_make_signal(i, "SHORT", open_next, stop, tp, signal_time))
            last_short = i
    return signals


def get_signals_hybrid(df):
    """Combine best: (1) Pullback to EMA20 IN TREND + RSI + volume. (2) EMA9/21 cross + strong bar + volume. Params from CONFIG for fewer SL."""
    if df is None or len(df) < 120:
        return []
    signals = []
    last_long, last_short = -99, -99
    cooldown = CONFIG.get("COOLDOWN_BARS", 10)
    cd_pb = cooldown + 2
    cd_cross = cooldown
    vol_min = CONFIG.get("VOL_MIN", 1.3)
    rsi_high = CONFIG.get("RSI_LONG_MAX", 57)
    rsi_low = CONFIG.get("RSI_SHORT_MIN", 43)
    strong_pct = CONFIG.get("STRONG_BAR_PCT", 0.5)
    atr_mul = 1.5
    for i in range(55, len(df) - 1):
        row = df.iloc[i]
        if np.isnan(row["ema_20"]) or np.isnan(row["ema_slow"]) or row["atr"] <= 0:
            continue
        atr = float(row["atr"])
        ema20, ema50 = row["ema_20"], row["ema_slow"]
        open_next = float(df["open"].iloc[i + 1])
        dt = row["datetime"]
        signal_time = dt.strftime("%Y-%m-%d %H:%M") if hasattr(dt, "strftime") else str(dt)
        rsi = row["rsi"]
        vol_ok = row["vol_ratio"] > vol_min
        rsi_mid = np.isnan(rsi) or (rsi_low <= rsi <= rsi_high)

        # --- (1) Pullback in trend: long = close>EMA50, low touched EMA20, bullish bar ---
        in_uptrend = row["close"] > ema50
        in_downtrend = row["close"] < ema50
        touch_ema20_long = row["low"] <= ema20 * 1.002 and row["close"] > ema20
        touch_ema20_short = row["high"] >= ema20 * 0.998 and row["close"] < ema20
        bullish_bar = row["close"] > row["open"]
        bearish_bar = row["close"] < row["open"]
        prev_close = df["close"].iloc[i - 1] if i > 0 else row["close"]
        prev_above_ema50 = prev_close > ema50
        prev_below_ema50 = prev_close < ema50
        bar_range = row["high"] - row["low"]
        not_climax = bar_range <= atr * 2.0 if atr > 0 else True

        if in_uptrend and prev_above_ema50 and touch_ema20_long and bullish_bar and rsi_mid and vol_ok and not_climax and (i - last_long >= cd_pb):
            stop = min(float(row["low"]), open_next - atr * atr_mul)
            if stop >= open_next or open_next - stop < atr * 0.25:
                pass
            elif CONFIG.get("SKIP_WEEKENDS") and i + 1 < len(df) and df["is_weekend"].iloc[i + 1]:
                pass
            else:
                risk = open_next - stop
                tp = open_next + risk * RR_RATIO
                sig = _make_signal(i, "LONG", open_next, stop, tp, signal_time)
                sig["entry_type"] = "pullback_ema20"
                signals.append(sig)
                last_long = i
                continue
        if in_downtrend and prev_below_ema50 and touch_ema20_short and bearish_bar and rsi_mid and vol_ok and not_climax and (i - last_short >= cd_pb):
            stop = max(float(row["high"]), open_next + atr * atr_mul)
            if stop <= open_next or stop - open_next < atr * 0.25:
                pass
            elif CONFIG.get("SKIP_WEEKENDS") and i + 1 < len(df) and df["is_weekend"].iloc[i + 1]:
                pass
            else:
                risk = stop - open_next
                tp = open_next - risk * RR_RATIO
                sig = _make_signal(i, "SHORT", open_next, stop, tp, signal_time)
                sig["entry_type"] = "pullback_ema20"
                signals.append(sig)
                last_short = i
                continue

        # --- (2) EMA 9/21 cross + strong bar (only if no pullback this bar) ---
        if i < 25:
            continue
        curr = df.iloc[i - 1]
        confirm = df.iloc[i]
        if np.isnan(curr["ema_9"]) or np.isnan(curr["ema_21"]):
            continue
        ema9_prev = df["ema_9"].iloc[i - 3]
        ema21_prev = df["ema_21"].iloc[i - 3]
        cross_long = ema9_prev <= ema21_prev and curr["ema_9"] > curr["ema_21"]
        cross_short = ema9_prev >= ema21_prev and curr["ema_9"] < curr["ema_21"]
        confirm_long = confirm["close"] > curr["close"]
        confirm_short = confirm["close"] < curr["close"]
        sh, slo = curr["high"], curr["low"]
        rng = (sh - slo) if (sh > slo) else 1e-9
        strong_long = (float(curr["close"]) - slo) >= strong_pct * rng
        strong_short = (sh - float(curr["close"])) >= strong_pct * rng
        rsi_c = curr["rsi"]
        rsi_ok_l = np.isnan(rsi_c) or (rsi_low <= rsi_c <= rsi_high)
        rsi_ok_s = np.isnan(rsi_c) or (rsi_low <= rsi_c <= rsi_high)
        vol_c = curr["vol_ratio"] > vol_min

        entry = float(confirm["close"])
        if cross_long and confirm_long and vol_c and rsi_ok_l and strong_long and (i - 1 - last_long >= cd_cross):
            stop = entry - atr * 1.2
            if stop < entry and (entry - stop) >= atr * 0.3 and not (CONFIG.get("SKIP_WEEKENDS") and df["is_weekend"].iloc[i]):
                risk = entry - stop
                tp = entry + risk * RR_RATIO
                sig = _make_signal(i, "LONG", entry, stop, tp, signal_time)
                sig["entry_type"] = "ema9_21_cross"
                signals.append(sig)
                last_long = i - 1
        elif cross_short and confirm_short and vol_c and rsi_ok_s and strong_short and (i - 1 - last_short >= cd_cross):
            stop = entry + atr * 1.2
            if stop > entry and (stop - entry) >= atr * 0.3 and not (CONFIG.get("SKIP_WEEKENDS") and df["is_weekend"].iloc[i]):
                risk = stop - entry
                tp = entry - risk * RR_RATIO
                sig = _make_signal(i, "SHORT", entry, stop, tp, signal_time)
                sig["entry_type"] = "ema9_21_cross"
                signals.append(sig)
                last_short = i - 1
    return signals


def get_signals_ema9_21(df):
    """More entries: EMA 9/21 cross only (no BOS). Confirm candle + RSI filter. More crosses on 5m than 20/50."""
    if df is None or len(df) < 100:
        return []
    signals = []
    last_long, last_short = -99, -99
    vol_min = CONFIG.get("VOL_MIN", 1.0)
    cooldown = CONFIG.get("COOLDOWN_BARS", 6)
    rsi_max_l = CONFIG.get("RSI_LONG_MAX", 65)
    rsi_min_s = CONFIG.get("RSI_SHORT_MIN", 35)
    for i in range(22, len(df)):
        prev = df.iloc[i - 2]
        curr = df.iloc[i - 1]
        confirm = df.iloc[i]
        if np.isnan(curr["ema_9"]) or np.isnan(curr["ema_21"]) or curr["atr"] <= 0:
            continue
        ema9_prev = df["ema_9"].iloc[i - 3]
        ema21_prev = df["ema_21"].iloc[i - 3]
        cross_long = ema9_prev <= ema21_prev and curr["ema_9"] > curr["ema_21"]
        cross_short = ema9_prev >= ema21_prev and curr["ema_9"] < curr["ema_21"]
        confirm_long = confirm["close"] > curr["close"]
        confirm_short = confirm["close"] < curr["close"]
        vol_ok = curr["vol_ratio"] > vol_min
        rsi = curr["rsi"]
        rsi_ok_long = np.isnan(rsi) or (30 < rsi < rsi_max_l)
        rsi_ok_short = np.isnan(rsi) or (rsi_min_s < rsi < 70)
        entry = float(confirm["close"])
        atr = float(curr["atr"])
        dt = confirm["datetime"]
        signal_time = dt.strftime("%Y-%m-%d %H:%M") if hasattr(dt, "strftime") else str(dt)
        if cross_long and confirm_long and vol_ok and rsi_ok_long and (i - 1 - last_long >= cooldown):
            stop = entry - atr * CONFIG.get("STOP_ATR_MUL", 1.2)
            if stop >= entry:
                continue
            if CONFIG.get("SKIP_WEEKENDS") and df["is_weekend"].iloc[i]:
                continue
            risk = entry - stop
            tp = entry + risk * RR_RATIO
            signals.append(_make_signal(i, "LONG", entry, stop, tp, signal_time))
            last_long = i - 1
        elif cross_short and confirm_short and vol_ok and rsi_ok_short and (i - 1 - last_short >= cooldown):
            stop = entry + atr * CONFIG.get("STOP_ATR_MUL", 1.2)
            if stop <= entry:
                continue
            if CONFIG.get("SKIP_WEEKENDS") and df["is_weekend"].iloc[i]:
                continue
            risk = stop - entry
            tp = entry - risk * RR_RATIO
            signals.append(_make_signal(i, "SHORT", entry, stop, tp, signal_time))
            last_short = i - 1
    return signals


def get_signals(df):
    strategy = CONFIG.get("STRATEGY", "trend_change")
    if strategy == "pullback_ema":
        return get_signals_pullback_ema(df)
    if strategy == "breakout_10":
        return get_signals_breakout_10(df)
    if strategy == "ema9_21":
        return get_signals_ema9_21(df)
    if strategy == "hybrid":
        return get_signals_hybrid(df)
    # trend_change: EMA 20/50 cross + BOS/retest + volume + confirm + RSI
    if df is None or len(df) < MIN_BARS:
        return []
    signals = []
    last_long, last_short = -999, -999
    for i in range(220, len(df)):
        signal = df.iloc[i - 1]
        confirm = df.iloc[i]
        if np.isnan(signal["ema_fast"]) or np.isnan(signal["ema_slow"]) or np.isnan(signal["atr"]) or signal["atr"] <= 0:
            continue
        if CONFIG.get("USE_ADX", False) and (np.isnan(signal.get("adx")) or signal["adx"] < CONFIG.get("ADX_MIN", 20)):
            continue
        swing_high = signal["swing_high"]
        swing_low = signal["swing_low"]
        vol_ok = signal["vol_ratio"] > CONFIG.get("VOL_MIN", 1.1)
        ema_fast_prev = df["ema_fast"].iloc[i - 3]
        ema_slow_prev = df["ema_slow"].iloc[i - 3]
        recent_cross_long = ema_fast_prev <= ema_slow_prev and signal["ema_fast"] > signal["ema_slow"]
        recent_cross_short = ema_fast_prev >= ema_slow_prev and signal["ema_fast"] < signal["ema_slow"]
        bos_long = signal["close"] > swing_high if pd.notna(swing_high) else False
        bos_short = signal["close"] < swing_low if pd.notna(swing_low) else False
        atr_band = signal["atr"] * 0.5
        retest_long = abs(signal["close"] - swing_high) <= atr_band if pd.notna(swing_high) else False
        retest_short = abs(signal["close"] - swing_low) <= atr_band if pd.notna(swing_low) else False
        too_late_long = abs(confirm["close"] - swing_high) > signal["atr"] * 1.2 if pd.notna(swing_high) else False
        too_late_short = abs(confirm["close"] - swing_low) > signal["atr"] * 1.2 if pd.notna(swing_low) else False
        bos_only = CONFIG.get("BOS_ONLY", False)
        long_trigger = recent_cross_long and (bos_long if bos_only else (bos_long or retest_long)) and not too_late_long
        short_trigger = recent_cross_short and (bos_short if bos_only else (bos_short or retest_short)) and not too_late_short
        confirm_long = confirm["close"] > signal["close"]
        confirm_short = confirm["close"] < signal["close"]
        rsi = signal["rsi"]
        rsi_ok_long = np.isnan(rsi) or (rsi < CONFIG.get("RSI_LONG_MAX", 68) and rsi > 30)
        rsi_ok_short = np.isnan(rsi) or (rsi > CONFIG.get("RSI_SHORT_MIN", 32) and rsi < 70)
        strong_pct = CONFIG.get("STRONG_BAR_PCT", 0.45)
        sh, slo = signal["high"], signal["low"]
        rng = (sh - slo) if (sh > slo) else 1e-9
        strong_long_bar = (float(signal["close"]) - slo) >= strong_pct * rng
        strong_short_bar = (sh - float(signal["close"])) >= strong_pct * rng

        entry = float(confirm["close"])
        atr = float(signal["atr"])
        dt = confirm["datetime"]
        signal_time = dt.strftime("%Y-%m-%d %H:%M") if hasattr(dt, "strftime") else str(dt)

        cooldown = CONFIG.get("COOLDOWN_BARS", 6)
        if long_trigger and vol_ok and confirm_long and rsi_ok_long and strong_long_bar and (i - last_long >= cooldown):
            stop_cand = float(swing_low) if pd.notna(swing_low) else entry - atr * 1.5
            stop = min(stop_cand, entry - atr * CONFIG.get("STOP_ATR_MUL", 1.2))
            if stop >= entry:
                continue
            if CONFIG.get("SKIP_WEEKENDS") and df["is_weekend"].iloc[i]:
                continue
            risk = entry - stop
            tp = entry + risk * RR_RATIO
            signals.append({"bar_index": i, "direction": "LONG", "entry": entry, "stop": stop, "take_profit": tp, "signal_time": signal_time})
            last_long = i
        elif short_trigger and vol_ok and confirm_short and rsi_ok_short and strong_short_bar and (i - last_short >= cooldown):
            stop_cand = float(swing_high) if pd.notna(swing_high) else entry + atr * 1.5
            stop = max(stop_cand, entry + atr * CONFIG.get("STOP_ATR_MUL", 1.2))
            if stop <= entry:
                continue
            if CONFIG.get("SKIP_WEEKENDS") and df["is_weekend"].iloc[i]:
                continue
            risk = stop - entry
            tp = entry - risk * RR_RATIO
            signals.append({"bar_index": i, "direction": "SHORT", "entry": entry, "stop": stop, "take_profit": tp, "signal_time": signal_time})
            last_short = i
    return signals


def backtest(df, signals):
    results = []
    total_r = 0.0
    loss_per_sl = CONFIG.get("SL_R", 1.0)
    for s in signals:
        idx = s["bar_index"]
        future = df.iloc[idx + 1:]
        outcome = "OPEN"
        for _, row in future.iterrows():
            high, low = float(row["high"]), float(row["low"])
            if s["direction"] == "LONG":
                if low <= s["stop"]:
                    outcome = "SL"
                    total_r -= loss_per_sl
                    break
                if high >= s["take_profit"]:
                    outcome = "TP"
                    total_r += RR_RATIO
                    break
            else:
                if high >= s["stop"]:
                    outcome = "SL"
                    total_r -= loss_per_sl
                    break
                if low <= s["take_profit"]:
                    outcome = "TP"
                    total_r += RR_RATIO
                    break
        s["outcome"] = outcome
        results.append(s)
    return results, total_r


def run_all_coins_silent(days, symbols_list=None):
    """Run backtest on all high-volume symbols, return (total_net_r, results_list). No per-coin print."""
    if symbols_list is None:
        symbols_list = get_high_volume_symbols(min_volume=10_000_000)
    if not symbols_list:
        return 0.0, []
    results = []
    for s in symbols_list:
        api_symbol = s["symbol"]
        row = run_backtest_one(api_symbol, days, verbose=False)
        if row is None:
            continue
        results.append(row)
        time.sleep(0.2)
    total_net = sum(r["net_r"] for r in results)
    return total_net, results


def run_backtest_one(api_symbol, days, verbose=True, interval=None):
    """Run backtest for one symbol. interval: Min1, Min5, Min15, Min30, Min60. Returns dict with stats or None."""
    df = fetch_history(api_symbol, days, interval=interval or INTERVAL)
    if df is None or len(df) < MIN_BARS:
        return None
    cutoff = df["datetime"].max() - timedelta(days=days)
    df = df[df["datetime"] >= cutoff].reset_index(drop=True)
    if len(df) < MIN_BARS:
        return None
    df = add_indicators(df)
    avg_price = float(df["close"].mean()) if len(df) else 0.0
    signals = get_signals(df)
    if not signals:
        return {"symbol": api_symbol.replace("_", ""), "trades": 0, "tp": 0, "sl": 0, "open": 0, "net_r": 0.0, "win_rate": 0, "interval": interval or INTERVAL, "avg_price": avg_price}
    trades, total_r = backtest(df, signals)
    tp_count = sum(1 for t in trades if t["outcome"] == "TP")
    sl_count = sum(1 for t in trades if t["outcome"] == "SL")
    open_count = sum(1 for t in trades if t["outcome"] == "OPEN")
    closed = tp_count + sl_count
    win_rate = round(100.0 * tp_count / closed, 1) if closed else 0
    trades_detail = [
        {"entry_type": t.get("entry_type", "other"), "outcome": t["outcome"], "signal_time": t.get("signal_time", ""), "direction": t.get("direction", "")}
        for t in trades
    ]
    return {
        "symbol": api_symbol.replace("_", ""),
        "trades": len(trades),
        "tp": tp_count,
        "sl": sl_count,
        "open": open_count,
        "net_r": total_r,
        "win_rate": win_rate,
        "interval": interval or INTERVAL,
        "trades_detail": trades_detail,
        "avg_price": avg_price,
    }


def run_compare_strategies(days=60, max_coins=None):
    """Run 60 days for each strategy; print and write results. max_coins=None = all."""
    strategies = ["hybrid", "pullback_ema", "breakout_10", "ema9_21", "trend_change"]
    symbols_list = get_high_volume_symbols(min_volume=10_000_000)
    if not symbols_list:
        print("  Failed to get symbol list.")
        return
    if max_coins:
        symbols_list = symbols_list[:max_coins]
        print(f"  (Using first {max_coins} coins for speed)")
    print()
    print("=" * 72)
    print(f"  COMPARE STRATEGIES (5m, {days} days, {len(symbols_list)} coins). R:R 1:3")
    print("=" * 72)
    results_by_strategy = []
    for name in strategies:
        CONFIG["STRATEGY"] = name
        print(f"\n  Running: {name} ...")
        total_net, res = run_all_coins_silent(days, symbols_list)
        total_trades = sum(r["trades"] for r in res)
        total_tp = sum(r["tp"] for r in res)
        total_sl = sum(r["sl"] for r in res)
        total_open = sum(r["open"] for r in res)
        results_by_strategy.append({
            "name": name,
            "trades": total_trades,
            "tp": total_tp,
            "sl": total_sl,
            "open": total_open,
            "net_r": total_net,
        })
        print(f"    -> Trades: {total_trades}  TP: {total_tp}  SL: {total_sl}  Net: {total_net:+.1f} R")
    print()
    print("-" * 72)
    print(f"  {'STRATEGY':<18} {'TRADES':>8} {'TP':>6} {'SL':>6} {'OPEN':>5} {'NET R':>10}")
    print("-" * 72)
    for r in results_by_strategy:
        print(f"  {r['name']:<18} {r['trades']:>8} {r['tp']:>6} {r['sl']:>6} {r['open']:>5} {r['net_r']:>+10.1f}")
    print("-" * 72)
    best = max(results_by_strategy, key=lambda x: x["net_r"])
    print(f"  Best by Net R: {best['name']} ({best['net_r']:+.1f} R, {best['trades']} trades)")
    out_path = os.path.join(os.path.dirname(__file__), "backtest_5m_strategy_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# 5m backtest, {days} days, all coins. R:R 1:3\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"{'STRATEGY':<18} {'TRADES':>8} {'TP':>6} {'SL':>6} {'OPEN':>5} {'NET R':>10}\n")
        f.write("-" * 60 + "\n")
        for r in results_by_strategy:
            f.write(f"{r['name']:<18} {r['trades']:>8} {r['tp']:>6} {r['sl']:>6} {r['open']:>5} {r['net_r']:>+10.1f}\n")
        f.write(f"\nBest: {best['name']}  Net={best['net_r']:+.1f} R  Trades={best['trades']}\n")
    print(f"\n  Results written to: {out_path}")
    print("  Done.")


def run_tune(days=60, quick=False):
    """Try configs until total_net_r >= TARGET_R on all coins. If quick=True, tune on first 15 coins then run full once with best."""
    symbols_list = get_high_volume_symbols(min_volume=10_000_000)
    if not symbols_list:
        print("  Failed to get symbol list.")
        return
    tune_symbols = symbols_list[:15] if quick else symbols_list
    configs_to_try = [
        {"VOL_MIN": 1.3, "COOLDOWN_BARS": 12, "RSI_LONG_MAX": 62, "RSI_SHORT_MIN": 38, "STRONG_BAR_PCT": 0.55, "USE_ADX": True, "ADX_MIN": 22, "BOS_ONLY": False},
        {"VOL_MIN": 1.4, "COOLDOWN_BARS": 15, "RSI_LONG_MAX": 60, "RSI_SHORT_MIN": 40, "STRONG_BAR_PCT": 0.55, "USE_ADX": True, "ADX_MIN": 25, "BOS_ONLY": False},
        {"VOL_MIN": 1.35, "COOLDOWN_BARS": 10, "RSI_LONG_MAX": 63, "RSI_SHORT_MIN": 37, "STRONG_BAR_PCT": 0.52, "USE_ADX": True, "ADX_MIN": 20, "BOS_ONLY": True},
        {"VOL_MIN": 1.25, "COOLDOWN_BARS": 10, "RSI_LONG_MAX": 65, "RSI_SHORT_MIN": 35, "STRONG_BAR_PCT": 0.5, "USE_ADX": False, "BOS_ONLY": False},
        {"VOL_MIN": 1.5, "COOLDOWN_BARS": 18, "RSI_LONG_MAX": 58, "RSI_SHORT_MIN": 42, "STRONG_BAR_PCT": 0.6, "USE_ADX": True, "ADX_MIN": 24, "BOS_ONLY": True},
        {"VOL_MIN": 1.2, "COOLDOWN_BARS": 8, "RSI_LONG_MAX": 64, "RSI_SHORT_MIN": 36, "STRONG_BAR_PCT": 0.5, "USE_ADX": True, "ADX_MIN": 18, "BOS_ONLY": False},
        {"VOL_MIN": 1.45, "COOLDOWN_BARS": 14, "RSI_LONG_MAX": 60, "RSI_SHORT_MIN": 40, "STRONG_BAR_PCT": 0.58, "USE_ADX": True, "ADX_MIN": 22, "BOS_ONLY": True},
        {"VOL_MIN": 1.3, "COOLDOWN_BARS": 12, "RSI_LONG_MAX": 61, "RSI_SHORT_MIN": 39, "STRONG_BAR_PCT": 0.55, "USE_ADX": True, "ADX_MIN": 21, "BOS_ONLY": True},
        {"VOL_MIN": 1.2, "COOLDOWN_BARS": 10, "RSI_LONG_MAX": 63, "RSI_SHORT_MIN": 37, "STRONG_BAR_PCT": 0.52, "USE_ADX": True, "ADX_MIN": 20, "BOS_ONLY": False},
        {"VOL_MIN": 1.35, "COOLDOWN_BARS": 14, "RSI_LONG_MAX": 59, "RSI_SHORT_MIN": 41, "STRONG_BAR_PCT": 0.58, "USE_ADX": True, "ADX_MIN": 23, "BOS_ONLY": True},
    ]
    best_net = -1e9
    best_config = None
    best_results = []
    for run, cfg in enumerate(configs_to_try):
        CONFIG.update(cfg)
        print(f"\n  TUNE run {run+1}/{len(configs_to_try)}: VOL_MIN={cfg.get('VOL_MIN')} COOLDOWN={cfg.get('COOLDOWN_BARS')} RSI_L={cfg.get('RSI_LONG_MAX')} RSI_S={cfg.get('RSI_SHORT_MIN')} STRONG={cfg.get('STRONG_BAR_PCT')} ADX={cfg.get('USE_ADX')} BOS_ONLY={cfg.get('BOS_ONLY')}")
        total_net, results = run_all_coins_silent(days, tune_symbols)
        total_tp = sum(r["tp"] for r in results)
        total_sl = sum(r["sl"] for r in results)
        total_trades = sum(r["trades"] for r in results)
        print(f"  -> Total: {total_trades} trades  TP={total_tp} SL={total_sl}  Net={total_net:+.1f} R")
        if total_net >= TARGET_R:
            print(f"\n  *** TARGET REACHED: {total_net:+.1f} R >= {TARGET_R} R ***")
            best_net = total_net
            best_config = dict(cfg)
            best_results = results
            break
        if total_net > best_net:
            best_net = total_net
            best_config = dict(cfg)
            best_results = results
    if best_config is not None:
        CONFIG.update(best_config)
        if quick:
            print(f"\n  Quick tune best on {len(tune_symbols)} coins: {best_net:+.1f} R. Running FULL run on all {len(symbols_list)} coins...")
            full_net, best_results = run_all_coins_silent(days, symbols_list)
            best_net = full_net
        print("\n" + "=" * 72)
        print(f"  BEST CONFIG (Net = {best_net:+.1f} R): {best_config}")
        print("=" * 72)
        if best_results:
            print(f"  {'SYMBOL':<16} {'TRADES':>6} {'TP':>4} {'SL':>4} {'NET R':>8}")
            print("-" * 72)
            for r in sorted(best_results, key=lambda x: -x["net_r"])[:40]:
                print(f"  {r['symbol']:<16} {r['trades']:>6} {r['tp']:>4} {r['sl']:>4} {r['net_r']:>+8.1f}")
            print("-" * 72)
            print(f"  TOTAL: {sum(r['trades'] for r in best_results)} trades  TP={sum(r['tp'] for r in best_results)}  SL={sum(r['sl'] for r in best_results)}  Net={best_net:+.1f} R")
        try:
            with open(os.path.join(os.path.dirname(__file__), "backtest_5m_best_config.txt"), "w", encoding="utf-8") as f:
                f.write(f"# Best config: Net {best_net:+.1f} R\n")
                for k, v in best_config.items():
                    f.write(f"{k}={v}\n")
        except Exception:
            pass
    print("  Done.")


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    symbol_arg = (sys.argv[2] if len(sys.argv) > 2 else "LINKUSDT").strip().upper()
    run_all = symbol_arg == "ALL"
    run_tune_mode = symbol_arg == "TUNE"
    quick_tune = len(sys.argv) > 3 and sys.argv[3].strip().upper() == "QUICK"
    run_timeframes = symbol_arg == "TIMEFRAMES" or (len(sys.argv) > 3 and sys.argv[3].strip().upper() == "TIMEFRAMES")

    if run_tune_mode:
        run_tune(days, quick=quick_tune)
        return

    if symbol_arg == "STRATEGIES":
        max_coins = int(sys.argv[3]) if len(sys.argv) > 3 else None
        run_compare_strategies(days, max_coins=max_coins)
        return

    if run_timeframes:
        if symbol_arg == "TIMEFRAMES":
            tf_symbol = (sys.argv[3] if len(sys.argv) > 3 else "LINKUSDT").strip().upper()
        else:
            tf_symbol = symbol_arg
        if "USDT" not in tf_symbol:
            tf_symbol = tf_symbol + "USDT"
        api_symbol = tf_symbol.replace("USDT", "_USDT")
        timeframes = ["Min1", "Min5", "Min15", "Min30", "Min60"]
        print()
        print("=" * 72)
        print("  BACKTEST: SAME STRATEGY (hybrid strict) ON DIFFERENT TIMEFRAMES  R:R 1:3")
        print(f"  Symbol: {tf_symbol}  |  Last {days} days")
        print("=" * 72)
        results_tf = []
        for tf in timeframes:
            label = {"Min1": "1m", "Min5": "5m", "Min15": "15m", "Min30": "30m", "Min60": "1h"}.get(tf, tf)
            print(f"  [{tf}] {label}...", end=" ", flush=True)
            row = run_backtest_one(api_symbol, days, verbose=False, interval=tf)
            if row is None:
                print("no data/skip")
                continue
            row["tf"] = label
            results_tf.append(row)
            print(f"trades={row['trades']} TP={row['tp']} SL={row['sl']} Net={row['net_r']:+.1f}R  Win%={row['win_rate']}")
            time.sleep(0.3)
        if not results_tf:
            print("  No results.")
            return
        print()
        print("-" * 72)
        print(f"  {'TF':>4} {'TRADES':>7} {'TP':>4} {'SL':>4} {'OPEN':>4} {'NET R':>8} {'WIN%':>6}")
        print("-" * 72)
        for r in results_tf:
            print(f"  {r['tf']:>4} {r['trades']:>7} {r['tp']:>4} {r['sl']:>4} {r['open']:>4} {r['net_r']:>+8.1f} {r['win_rate']:>5.1f}%")
        print("-" * 72)
        best = max(results_tf, key=lambda x: x["net_r"])
        print(f"  Best by Net R: {best['tf']}  ({best['net_r']:+.1f} R)")
        print("  Done.")
        return

    if run_all:
        symbols = get_high_volume_symbols(min_volume=10_000_000)
        if not symbols:
            print("  Failed to get symbol list.")
            return
        print()
        print("=" * 72)
        print("  BACKTEST 5m: ALL COINS  Strategy:", CONFIG.get("STRATEGY", "trend_change"), "  R:R 1:3  SL=0.5R", "  (no weekends)" if CONFIG.get("SKIP_WEEKENDS") else "")
        print(f"  Last {days} days  |  {len(symbols)} symbols (volume > 10M, excl. {len(EXCLUDED_SYMBOLS)} min order > $5)")
        print("=" * 72)
        results = []
        for i, s in enumerate(symbols):
            api_symbol = s["symbol"]
            name = api_symbol.replace("_", "")
            print(f"  [{i+1}/{len(symbols)}] {name}...", end=" ", flush=True)
            row = run_backtest_one(api_symbol, days, verbose=False)
            if row is None:
                print("no data/skip")
                continue
            results.append(row)
            print(f"trades={row['trades']} TP={row['tp']} SL={row['sl']} Net={row['net_r']:+.1f}R")
            time.sleep(0.25)
        if not results:
            print("  No results.")
            return
        print()
        print("-" * 72)
        print(f"  {'SYMBOL':<14} {'TRADES':>6} {'TP':>4} {'SL':>4} {'OPEN':>4} {'NET R':>8} {'WIN%':>6}")
        print("-" * 72)
        for r in sorted(results, key=lambda x: -x["net_r"]):
            print(f"  {r['symbol']:<14} {r['trades']:>6} {r['tp']:>4} {r['sl']:>4} {r['open']:>4} {r['net_r']:>+8.1f} {r['win_rate']:>5.1f}%")
        print("-" * 72)
        total_trades = sum(r["trades"] for r in results)
        total_tp = sum(r["tp"] for r in results)
        total_sl = sum(r["sl"] for r in results)
        total_open = sum(r["open"] for r in results)
        total_net = sum(r["net_r"] for r in results)
        print(f"  {'TOTAL':<14} {total_trades:>6} {total_tp:>4} {total_sl:>4} {total_open:>4} {total_net:>+8.1f}")
        risk_per_trade_usd = 1.0
        pnl_usd = total_net * risk_per_trade_usd
        print(f"  Profit (risk ${risk_per_trade_usd:.0f}/trade, 1R=${risk_per_trade_usd:.0f}, 5x leverage): Total P&L = ${pnl_usd:+.2f}")

        # --- Statistics by entry strategy (hybrid sub-strategies) ---
        from collections import defaultdict
        by_type = defaultdict(lambda: {"trades": 0, "tp": 0, "sl": 0, "open": 0})
        log_rows = []
        for r in results:
            for td in r.get("trades_detail", []):
                et = td["entry_type"]
                by_type[et]["trades"] += 1
                if td["outcome"] == "TP":
                    by_type[et]["tp"] += 1
                elif td["outcome"] == "SL":
                    by_type[et]["sl"] += 1
                else:
                    by_type[et]["open"] += 1
                log_rows.append((r["symbol"], td.get("signal_time", ""), td.get("direction", ""), et, td["outcome"]))
        loss_per_sl = CONFIG.get("SL_R", 0.5)
        if by_type:
            print()
            print("  --- STATISTICS BY ENTRY STRATEGY (hybrid sub-types) ---")
            print("-" * 72)
            print(f"  {'ENTRY_TYPE':<20} {'TRADES':>6} {'TP':>4} {'SL':>4} {'OPEN':>4} {'NET R':>8} {'WIN%':>6}")
            print("-" * 72)
            for et in sorted(by_type.keys()):
                d = by_type[et]
                net_r = d["tp"] * RR_RATIO - d["sl"] * loss_per_sl
                closed = d["tp"] + d["sl"]
                win_pct = round(100.0 * d["tp"] / closed, 1) if closed else 0
                print(f"  {et:<20} {d['trades']:>6} {d['tp']:>4} {d['sl']:>4} {d['open']:>4} {net_r:>+8.1f} {win_pct:>5.1f}%")
            print("-" * 72)

        # --- Statistics by price range (where do we get most SL, few TP) ---
        PRICE_BUCKETS = [
            (0, 0.01, "<0.01"),
            (0.01, 0.1, "0.01-0.1"),
            (0.1, 1, "0.1-1"),
            (1, 5, "1-5"),
            (5, 10, "5-10"),
            (10, 50, "10-50"),
            (50, 100, "50-100"),
            (100, float("inf"), "100+"),
        ]
        by_bucket = {}
        for lo, hi, label in PRICE_BUCKETS:
            by_bucket[label] = {"tp": 0, "sl": 0, "net_r": 0.0, "symbols": [], "count": 0}
        for r in results:
            p = r.get("avg_price") or 0
            for lo, hi, label in PRICE_BUCKETS:
                if lo <= p < hi:
                    by_bucket[label]["tp"] += r["tp"]
                    by_bucket[label]["sl"] += r["sl"]
                    by_bucket[label]["net_r"] += r["net_r"]
                    by_bucket[label]["count"] += 1
                    by_bucket[label]["symbols"].append((r["symbol"], r["tp"], r["sl"], r["net_r"], r["win_rate"], p))
                    break
        loss_per_sl = CONFIG.get("SL_R", 0.5)
        print()
        print("  --- BY PRICE RANGE (coin value) — where are most stop-losses? ---")
        print("-" * 72)
        print(f"  {'PRICE RANGE':<12} {'COINS':>5} {'TP':>5} {'SL':>5} {'NET R':>8} {'WIN%':>6}")
        print("-" * 72)
        for label in [b[2] for b in PRICE_BUCKETS]:
            d = by_bucket[label]
            if d["count"] == 0:
                continue
            closed = d["tp"] + d["sl"]
            win_pct = round(100.0 * d["tp"] / closed, 1) if closed else 0
            print(f"  {label:<12} {d['count']:>5} {d['tp']:>5} {d['sl']:>5} {d['net_r']:>+8.1f} {win_pct:>5.1f}%")
        print("-" * 72)

        # Worst coins: most SL, few TP (candidates to exclude)
        worst = [r for r in results if r["trades"] >= 10 and (r["sl"] - r["tp"]) >= 5 and r["net_r"] < 10]
        worst.sort(key=lambda x: (x["net_r"], -x["tp"], x["sl"]))
        if worst:
            print()
            print("  --- WORST COINS (many SL, few TP) — consider excluding ---")
            print("-" * 72)
            print(f"  {'SYMBOL':<14} {'PRICE':>10} {'TRADES':>6} {'TP':>4} {'SL':>4} {'NET R':>8} {'WIN%':>6}")
            print("-" * 72)
            for r in worst[:25]:
                p = r.get("avg_price") or 0
                print(f"  {r['symbol']:<14} {p:>10.4f} {r['trades']:>6} {r['tp']:>4} {r['sl']:>4} {r['net_r']:>+8.1f} {r['win_rate']:>5.1f}%")
            print("-" * 72)

        # Log every transaction to CSV
        log_path = os.path.join(os.path.dirname(__file__) or ".", "backtest_30d_strategy_log.csv")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("symbol,signal_time,direction,entry_type,outcome\n")
                for sym, st, dr, typ, out in log_rows:
                    f.write(f'"{sym}","{st}","{dr}","{typ}","{out}"\n')
            print(f"  Transaction log: {log_path}")
        except Exception as e:
            print(f"  (Could not write log: {e})")
        print("  Done.")
        return

    symbol = symbol_arg
    if "USDT" not in symbol:
        symbol = symbol + "USDT"
    api_symbol = symbol.replace("USDT", "_USDT")

    print()
    print("=" * 60)
    print("  BACKTEST: Strategy:", CONFIG.get("STRATEGY", "trend_change"), "  TF:", INTERVAL, "  R:R 1:3")
    print(f"  Symbol: {symbol}  |  Last {days} days")
    print("=" * 60)
    print(f"  Fetching {INTERVAL} history (chunks)...")
    df = fetch_history(api_symbol, days, interval=INTERVAL)
    if df is None or len(df) < MIN_BARS:
        print("  Failed to load data or not enough bars.")
        return
    cutoff = df["datetime"].max() - timedelta(days=days)
    df = df[df["datetime"] >= cutoff].reset_index(drop=True)
    if len(df) < MIN_BARS:
        print("  Not enough bars after cutoff.")
        return
    print(f"  Bars: {len(df)}  |  {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    df = add_indicators(df)
    signals = get_signals(df)
    if not signals:
        print("  No signals.")
        return
    trades, total_r = backtest(df, signals)
    tp_count = sum(1 for t in trades if t["outcome"] == "TP")
    sl_count = sum(1 for t in trades if t["outcome"] == "SL")
    open_count = sum(1 for t in trades if t["outcome"] == "OPEN")
    closed = tp_count + sl_count
    win_rate = round(100.0 * tp_count / closed, 1) if closed else 0
    pf = (tp_count * RR_RATIO) / sl_count if sl_count else (tp_count * RR_RATIO) if tp_count else 0

    print()
    print("  Trades:")
    for i, t in enumerate(trades[:30], 1):
        print(f"    {i}. {t['signal_time']}  {t['direction']}  Entry: {format_price(t['entry'])}  Stop: {format_price(t['stop'])}  TP: {format_price(t['take_profit'])}  ->  {t['outcome']}")
    if len(trades) > 30:
        print(f"    ... and {len(trades) - 30} more")
    print()
    print("  --- STATISTICS ---")
    loss_per_sl = CONFIG.get("SL_R", 1.0)
    print(f"  Total: {len(trades)}  |  TP: {tp_count} (+{tp_count * RR_RATIO:.1f} R)  |  SL: {sl_count} (-{sl_count * loss_per_sl:.1f} R)  |  Open: {open_count}")
    if closed:
        print(f"  Win rate: {win_rate}%  |  Net: {total_r:.2f} R  |  Profit factor: {pf:.2f}")
    risk_per_trade_usd = 1.0
    pnl_usd = total_r * risk_per_trade_usd
    print(f"  Profit (risk ${risk_per_trade_usd:.0f}/trade, 1R=${risk_per_trade_usd:.0f}, 5x leverage): P&L = ${pnl_usd:+.2f}")
    print("  Done.")


if __name__ == "__main__":
    main()
