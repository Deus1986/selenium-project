# -*- coding: utf-8 -*-
"""
Торговый скрипт: Confluence + Fair Value Gaps + Order Blocks (MEXC, 5m).

Стратегия объединяет несколько подтверждений для повышения вероятности входа:
1. Confluence Score (RSI + MACD + Volume + Trend) — минимум 3 из 4 в одну сторону.
2. Fair Value Gaps (FVG) — зоны дисбаланса, куда цена часто возвращается.
3. Order Blocks (OB) — последняя противоположная свеча перед сильным движением.

Вход только при совпадении направления confluence и касания зоны FVG или OB.
Параметры сделки: Вход, Стоп-лосс, Тейк-профит (R/R ≥ 1:3, прибыль ≥ 5%).
Цикл: каждые 5 минут, звуковой сигнал при новом сигнале.

Внимание: ни одна стратегия не даёт 100% результата. Этот скрипт использует
проверенные концепции (confluence, FVG, OB) для максимизации вероятности.
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


STATE_FILE = "confluence_signals_state.json"
MIN_CONFLUENCE_SCORE = 50   # минимум из 100 для направления
MIN_RR = 3.0                # минимальное соотношение риск/прибыль 1:3
MIN_PROFIT_PCT = 0.05       # минимальная прибыль 5% от входа (3R >= 5%)
FVG_LOOKBACK = 80           # сколько свечей искать FVG
OB_LOOKBACK = 60            # сколько свечей искать Order Blocks
ZONE_TOUCH_BARS = 5         # считаем "касание зоны" за последние N свечей


def get_high_volume_symbols(min_volume=10_000_000):
    """Фьючерсные пары MEXC с высоким объёмом."""
    url = "https://contract.mexc.com/api/v1/contract/ticker"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data.get("success", False):
            print("   ⚠️ API вернул success=false")
            return []
        raw = data.get("data")
        if raw is None:
            print("   ⚠️ API: data пусто")
            return []
        symbols = []
        for item in (raw if isinstance(raw, list) else [raw]):
            if not isinstance(item, dict):
                continue
            vol = item.get("amount24") or item.get("volume24") or 0
            try:
                vol = float(vol) if vol is not None else 0
            except (TypeError, ValueError):
                vol = 0
            sym = item.get("symbol", "")
            if sym and str(sym).endswith("_USDT") and vol > min_volume:
                symbols.append({"symbol": sym, "volume_24h": vol})
        if not symbols and raw:
            min_vol = 1_000_000
            for item in (raw if isinstance(raw, list) else [raw]):
                if not isinstance(item, dict):
                    continue
                vol = item.get("amount24") or item.get("volume24") or 0
                try:
                    vol = float(vol) if vol is not None else 0
                except (TypeError, ValueError):
                    vol = 0
                sym = item.get("symbol", "")
                if sym and str(sym).endswith("_USDT") and vol > min_vol:
                    symbols.append({"symbol": sym, "volume_24h": vol})
            if symbols:
                print(f"   ⚠️ Порог 10M не прошёл ни одна монета. Используем порог 1M ({len(symbols)} монет).")
        return sorted(symbols, key=lambda x: x["volume_24h"], reverse=True)
    except requests.RequestException as e:
        print(f"   ⚠️ Ошибка сети при запросе ticker: {e}")
        return []
    except Exception as e:
        print(f"   ⚠️ Ошибка списка монет: {e}")
        return []


def get_candles(symbol, interval="Min5", limit=500):
    url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
    try:
        r = requests.get(url, params={"interval": interval, "limit": limit}, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"   Ошибка данных {symbol}: {e}")
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
            df = pd.DataFrame({k: raw[k][:n] for k in req})
            df = df.rename(columns={"time": "timestamp", "vol": "volume"})
        elif isinstance(raw, list):
            rows = [row[:6] for row in raw if isinstance(row, (list, tuple)) and len(row) >= 6]
            if not rows:
                return None
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        else:
            return None
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()
        if len(df) < 150:
            return None
        ts = df["timestamp"].iloc[0]
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms" if ts > 1e12 else "s")
        df = df.sort_values("datetime").reset_index(drop=True)
        return df
    except Exception:
        return None


def add_indicators(df):
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    v = df["volume"].values
    df["atr"] = talib.ATR(h, l, c, timeperiod=14)
    df["vol_sma"] = talib.SMA(v, timeperiod=20)
    df["vol_ratio"] = v / df["vol_sma"]
    df["ema_5"] = talib.EMA(c, timeperiod=5)
    df["ema_20"] = talib.EMA(c, timeperiod=20)
    df["ema_50"] = talib.EMA(c, timeperiod=50)
    df["rsi"] = talib.RSI(c, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    df["body_size"] = np.abs(df["close"] - df["open"])
    return df


def format_price(x):
    if x is None or x == "":
        return ""
    if abs(x) < 0.0001:
        return f"{x:.8f}"
    if abs(x) < 0.01:
        return f"{x:.6f}"
    if abs(x) < 1:
        return f"{x:.5f}"
    if abs(x) < 100:
        return f"{x:.4f}"
    return f"{x:.2f}"


def confluence_score(df):
    """
    Confluence: RSI + MACD + Volume + Trend.
    Возвращает (bull_score, bear_score), каждый 0..100.
    """
    row = df.iloc[-1]
    bull, bear = 0, 0

    # RSI: перепроданность -> бычий, перекупленность -> медвежий
    rsi = row["rsi"]
    if not np.isnan(rsi):
        if rsi < 35:
            bull += 30
        elif rsi < 45:
            bull += 15
        elif rsi > 65:
            bear += 30
        elif rsi > 55:
            bear += 15

    # MACD гистограмма
    hist = row["macd_hist"]
    if not np.isnan(hist) and hist != 0:
        if hist > 0:
            bull += 25
        else:
            bear += 25

    # Объём выше среднего — подтверждение
    vr = row["vol_ratio"]
    if not np.isnan(vr) and vr > 1.2:
        # Не даём перевес по объёму одному направлению
        bull += 15
        bear += 15
    elif vr > 1.0:
        bull += 8
        bear += 8

    # Тренд по EMA
    e5, e20, e50 = row["ema_5"], row["ema_20"], row["ema_50"]
    if not (np.isnan(e5) or np.isnan(e20) or np.isnan(e50)):
        if e5 > e20 > e50:
            bull += 30
        elif e5 < e20 < e50:
            bear += 30

    return (min(100, bull), min(100, bear))


def find_fvg_zones(df, lookback=FVG_LOOKBACK):
    """
    Fair Value Gaps (ICT).
    Bullish FVG: low[i] > high[i-2] -> зона [high[i-2], low[i]] = (zone_low, zone_high).
    Bearish FVG: high[i] < low[i-2] -> зона [high[i], low[i-2]] = (zone_low, zone_high).
    """
    zones = []
    end = len(df) - 1
    start = max(2, end - lookback)
    for i in range(start, end - 1):
        if i < 2:
            continue
        h2, l2 = df["high"].iloc[i - 2], df["low"].iloc[i - 2]
        h1, l1 = df["high"].iloc[i - 1], df["low"].iloc[i - 1]
        h0, l0 = df["high"].iloc[i], df["low"].iloc[i]
        if l0 > h2 and l1 > h2:
            zones.append(("LONG", h2, l0, i))
        if h0 < l2 and h1 < l2:
            zones.append(("SHORT", h0, l2, i))
    return zones


def find_order_blocks(df, lookback=OB_LOOKBACK, atr_mult=1.2):
    """
    Упрощённые Order Blocks: последняя противоположная свеча перед сильным движением.
    Bullish OB: последняя красная свеча перед зелёной с телом > atr_mult*ATR.
    Bearish OB: последняя зелёная перед сильной красной.
    """
    blocks = []
    end = len(df) - 1
    start = max(2, end - lookback)
    atr = df["atr"].iloc[end - 1]
    if pd.isna(atr) or atr <= 0:
        atr = df["close"].iloc[end - 1] * 0.01

    for i in range(start, end - 1):
        if i < 1:
            continue
        body_next = df["body_size"].iloc[i + 1]
        if body_next < atr_mult * atr:
            continue
        # Сильная бычья свеча после — ищем медвежью свечу i
        if df["close"].iloc[i + 1] > df["open"].iloc[i + 1]:
            if df["close"].iloc[i] < df["open"].iloc[i]:
                blocks.append(("LONG", df["low"].iloc[i], df["high"].iloc[i], i))
        # Сильная медвежья после — ищем бычью i
        if df["close"].iloc[i + 1] < df["open"].iloc[i + 1]:
            if df["close"].iloc[i] > df["open"].iloc[i]:
                blocks.append(("SHORT", df["low"].iloc[i], df["high"].iloc[i], i))
    return blocks


def price_touches_zone(df, zone_low, zone_high, bars=ZONE_TOUCH_BARS):
    """Проверяет, касалась ли цена зоны [zone_low, zone_high] за последние bars свечей."""
    tail = df.tail(bars)
    for _, row in tail.iterrows():
        high, low, close = row["high"], row["low"], row["close"]
        if low <= zone_high and high >= zone_low:
            return True
        if zone_low <= close <= zone_high:
            return True
    return False


def evaluate_signal(df):
    """
    Сигнал: Confluence в одну сторону + касание зоны FVG или OB в ту же сторону.
    Возвращает dict с entry, stop, take_profit, direction, reasons или None.
    """
    if len(df) < 100:
        return None
    row = df.iloc[-1]
    atr = row["atr"]
    if pd.isna(atr) or atr <= 0:
        atr = row["close"] * 0.01
    bull_score, bear_score = confluence_score(df)
    fvg_zones = find_fvg_zones(df, lookback=FVG_LOOKBACK)
    ob_zones = find_order_blocks(df, lookback=OB_LOOKBACK)

    # LONG: зона (low, high) = (нижняя граница, верхняя граница)
    if bull_score >= MIN_CONFLUENCE_SCORE and bull_score >= bear_score:
        zone_used = None
        for ztype, zlo, zhi, _ in fvg_zones:
            if ztype == "LONG" and price_touches_zone(df, zlo, zhi, ZONE_TOUCH_BARS):
                zone_used = ("FVG", zlo, zhi)
                break
        if not zone_used:
            for ztype, zlo, zhi, _ in ob_zones:
                if ztype == "LONG" and price_touches_zone(df, zlo, zhi, ZONE_TOUCH_BARS):
                    zone_used = ("OB", zlo, zhi)
                    break
        if zone_used:
            source, zlo, zhi = zone_used
            entry = row["close"]
            stop = zlo - atr * 0.5
            if stop >= entry:
                stop = zlo - atr * 0.3
            risk = entry - stop
            if risk <= 0:
                return None
            take = entry + risk * MIN_RR
            rr = (take - entry) / risk
            # Фильтр: прибыль (3R) должна быть не менее 10% от входа
            profit_pct = (take - entry) / entry
            if profit_pct < MIN_PROFIT_PCT:
                return None
            return {
                "direction": "LONG",
                "entry": entry,
                "stop": stop,
                "take_profit": take,
                "rr": round(rr, 2),
                "reasons": f"Confluence LONG (score {bull_score}) + зона {source} [{format_price(zlo)}–{format_price(zhi)}]",
                "signal_time": row["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
                "score": bull_score,
            }

    # SHORT
    if bear_score >= MIN_CONFLUENCE_SCORE and bear_score >= bull_score:
        zone_used = None
        for ztype, zlo, zhi, _ in fvg_zones:
            if ztype == "SHORT" and price_touches_zone(df, zlo, zhi, ZONE_TOUCH_BARS):
                zone_used = ("FVG", zlo, zhi)
                break
        if not zone_used:
            for ztype, zlo, zhi, _ in ob_zones:
                if ztype == "SHORT" and price_touches_zone(df, zlo, zhi, ZONE_TOUCH_BARS):
                    zone_used = ("OB", zlo, zhi)
                    break
        if zone_used:
            source, zlo, zhi = zone_used
            entry = row["close"]
            stop = zhi + atr * 0.5
            if stop <= entry:
                stop = zhi + atr * 0.3
            risk = stop - entry
            if risk <= 0:
                return None
            take = entry - risk * MIN_RR
            rr = (entry - take) / risk
            # Фильтр: прибыль (3R) должна быть не менее 10% от входа
            profit_pct = (entry - take) / entry
            if profit_pct < MIN_PROFIT_PCT:
                return None
            return {
                "direction": "SHORT",
                "entry": entry,
                "stop": stop,
                "take_profit": take,
                "rr": round(rr, 2),
                "reasons": f"Confluence SHORT (score {bear_score}) + зона {source} [{format_price(zlo)}–{format_price(zhi)}]",
                "signal_time": row["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
                "score": bear_score,
            }

    return None


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()


def save_state(state_set):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(list(state_set)), f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def wait_until_next_5min():
    now = datetime.now()
    next_min = ((now.minute // 5) + 1) * 5
    if next_min >= 60:
        next_time = now.replace(minute=0, second=1, microsecond=0) + pd.Timedelta(hours=1)
    else:
        next_time = now.replace(minute=next_min, second=1, microsecond=0)
    sec = (next_time - now).total_seconds()
    if sec > 0:
        time.sleep(sec)


def run_once(state_set):
    print("\n" + "=" * 60)
    print("CONFLUENCE + FVG + ORDER BLOCKS (MEXC 5m)")
    print("Вход | Стоп | Тейк-профит | R/R ≥ 1:3 | Прибыль ≥ 5%")
    print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    symbols = get_high_volume_symbols(min_volume=10_000_000)
    if not symbols:
        print("Нет монет по объёму.")
        return state_set

    new_signals = []
    checked = 0
    for s in symbols:
        symbol = s["symbol"]
        checked += 1
        if checked % 15 == 0:
            print(f"   Проверено: {checked}/{len(symbols)}")
        data = get_candles(symbol, "Min5", 500)
        df = create_dataframe(data)
        if df is None:
            continue
        df = add_indicators(df)
        signal = evaluate_signal(df)
        if not signal:
            continue
        # Ключ: символ + направление + час (без минут) — не дублируем сигнал в пределах часа
        ts = signal["signal_time"][:13]  # "YYYY-MM-DD HH"
        key = f"{symbol}|{signal['direction']}|{ts}"
        if key not in state_set:
            new_signals.append((symbol, signal))
            state_set.add(key)
        time.sleep(0.2)

    if new_signals:
        print(f"\n🔔 НОВЫЕ СИГНАЛЫ: {len(new_signals)}")
        for symbol, s in new_signals:
            print("-" * 50)
            print(f"  Символ:     {symbol}")
            print(f"  Направление: {s['direction']}")
            print(f"  Вход:       {format_price(s['entry'])}")
            print(f"  Стоп-лосс:  {format_price(s['stop'])}")
            print(f"  Тейк-профит: {format_price(s['take_profit'])}  (R/R = {s['rr']})")
            print(f"  Время:      {s['signal_time']}")
            print(f"  Условия:    {s['reasons']}")
        print("-" * 50)
        try:
            winsound.MessageBeep(winsound.MB_ICONHAND)
        except Exception:
            pass
        for _ in range(5):
            winsound.Beep(1200, 500)
            winsound.Beep(800, 400)
            time.sleep(0.08)
    else:
        print(f"\nПроверено монет: {checked}. Новых сигналов нет.")

    return state_set


def run_loop():
    state_set = load_state()
    print("Запуск: каждые 5 минут (на 01-й секунде). Остановка: Ctrl+C")
    while True:
        wait_until_next_5min()
        try:
            state_set = run_once(state_set)
            save_state(state_set)
        except Exception as e:
            print(f"Ошибка цикла: {e}")


if __name__ == "__main__":
    run_loop()
