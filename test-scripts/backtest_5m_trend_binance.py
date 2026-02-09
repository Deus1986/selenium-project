"""
Binance USDT-M Futures 5m backtest. Same strategy as MEXC (hybrid, 3R TP, 0.3R SL).
Uses Binance historical klines; results comparable to MEXC run.

Run:  python test-scripts/backtest_5m_trend_binance.py [DAYS]
      Default 30 days, all high-volume USDT perpetuals (excluding same list as MEXC).
"""
import sys
import os
import time
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

# Reuse strategy logic from MEXC backtest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_5m_trend_mexc import (
    add_indicators,
    get_signals,
    backtest,
    CONFIG,
    RR_RATIO,
    MIN_BARS,
    format_price,
)

BINANCE_BASE = "https://fapi.binance.com"
API_TIMEOUT = 60
API_RETRIES = 3
MIN_VOLUME_24H_USDT = 10_000_000
# Min order (notional) max $5 so we can trade with $5 position (like MEXC)
MAX_MIN_NOTIONAL_USDT = 5.0
# Worst coins: trades >= this, (SL - TP) >= this, and net_r < this => exclude
BAD_MIN_TRADES = 10
BAD_SL_MINUS_TP = 5
BAD_MAX_NET_R = 10.0

# Same coins excluded as on MEXC (Binance symbol format: no underscore)
BINANCE_EXCLUDED = {
    "PEPEUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "BTCUSDT", "BNBUSDT",
    "CHZUSDT", "ZECUSDT", "XMRUSDT", "BCHUSDT", "NIGHTUSDT",
}


def _safe_console(s):
    """Avoid UnicodeEncodeError on Windows console (cp1251)."""
    if s is None:
        return ""
    return s.encode("ascii", "replace").decode("ascii")


def get_symbols_min_notional_at_most(max_notional=MAX_MIN_NOTIONAL_USDT):
    """Fetch exchangeInfo; return set of symbols where MIN_NOTIONAL <= max_notional (so we can trade with $5)."""
    url = f"{BINANCE_BASE}/fapi/v1/exchangeInfo"
    for attempt in range(API_RETRIES):
        try:
            r = requests.get(url, timeout=API_TIMEOUT)
            if r.status_code != 200:
                continue
            data = r.json()
            allowed = set()
            for sym_info in data.get("symbols", []):
                sym = sym_info.get("symbol", "")
                if not sym.endswith("USDT"):
                    continue
                for f in sym_info.get("filters", []):
                    if f.get("filterType") == "MIN_NOTIONAL":
                        try:
                            n = float(f.get("notional", 999))
                            if n <= max_notional:
                                allowed.add(sym)
                        except (TypeError, ValueError):
                            pass
                        break
            return allowed
        except Exception as e:
            if attempt < API_RETRIES - 1:
                time.sleep(3)
                continue
            print(f"  Binance exchangeInfo error: {e}")
            return set()
    return set()


def get_symbols_min_notional_map():
    """Fetch exchangeInfo; return dict symbol -> min_notional (float) for all USDT symbols."""
    url = f"{BINANCE_BASE}/fapi/v1/exchangeInfo"
    for attempt in range(API_RETRIES):
        try:
            r = requests.get(url, timeout=API_TIMEOUT)
            if r.status_code != 200:
                continue
            data = r.json()
            out = {}
            for sym_info in data.get("symbols", []):
                sym = sym_info.get("symbol", "")
                if not sym.endswith("USDT"):
                    continue
                for f in sym_info.get("filters", []):
                    if f.get("filterType") == "MIN_NOTIONAL":
                        try:
                            out[sym] = float(f.get("notional", 999))
                        except (TypeError, ValueError):
                            out[sym] = 999.0
                        break
            return out
        except Exception as e:
            if attempt < API_RETRIES - 1:
                time.sleep(3)
                continue
            print(f"  Binance exchangeInfo error: {e}")
            return {}
    return {}


def get_binance_symbols(min_volume_usdt=MIN_VOLUME_24H_USDT, max_min_notional_usdt=None):
    """Get USDT perpetual symbols: volume > min_volume, optional min notional <= max_min_notional_usdt, sorted by volume."""
    if max_min_notional_usdt is None:
        max_min_notional_usdt = MAX_MIN_NOTIONAL_USDT
    allowed_notional = get_symbols_min_notional_at_most(max_notional=max_min_notional_usdt)
    url = f"{BINANCE_BASE}/fapi/v1/ticker/24hr"
    for attempt in range(API_RETRIES):
        try:
            r = requests.get(url, timeout=API_TIMEOUT)
            if r.status_code != 200:
                continue
            data = r.json()
            symbols = []
            for item in data:
                sym = item.get("symbol", "")
                if not sym.endswith("USDT") or sym in BINANCE_EXCLUDED:
                    continue
                if allowed_notional and sym not in allowed_notional:
                    continue
                try:
                    qv = float(item.get("quoteVolume", 0))
                except (TypeError, ValueError):
                    continue
                if qv >= min_volume_usdt:
                    symbols.append({"symbol": sym, "volume_24h": qv})
            return sorted(symbols, key=lambda x: x["volume_24h"], reverse=True)
        except Exception as e:
            if attempt < API_RETRIES - 1:
                time.sleep(3)
                continue
            print(f"  Binance ticker error: {e}")
            return []
    return []


def get_binance_klines(symbol, interval="5m", limit=1500, end_time_ms=None):
    """Fetch one chunk of klines. Returns list of [openTime, open, high, low, close, volume, ...] or None."""
    url = f"{BINANCE_BASE}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)
    for attempt in range(API_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=API_TIMEOUT)
            if r.status_code != 200:
                continue
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                return data
            return None
        except Exception as e:
            if attempt < API_RETRIES - 1:
                time.sleep(2)
                continue
            return None
    return None


def fetch_binance_history(symbol, days, interval="5m"):
    """Fetch 5m (or interval) klines for last `days` days. Returns DataFrame like MEXC (time, open, high, low, close, volume, datetime)."""
    need_bars = days * 24 * (60 // 5)
    all_rows = []
    end_time_ms = None
    while len(all_rows) < need_bars:
        chunk = get_binance_klines(symbol, interval=interval, limit=1500, end_time_ms=end_time_ms)
        if not chunk:
            break
        for bar in chunk:
            ot = int(bar[0])
            o, h, l, c = float(bar[1]), float(bar[2]), float(bar[3]), float(bar[4])
            v = float(bar[5])
            all_rows.append({"time": ot, "open": o, "high": h, "low": l, "close": c, "volume": v})
        if len(chunk) < 1500:
            break
        end_time_ms = chunk[0][0] - 1
        time.sleep(0.15)
    if not all_rows:
        return None
    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["time"], unit="ms")
    return df


def run_backtest_one_binance(symbol, days, verbose=False):
    """Run backtest for one Binance symbol. Returns dict with stats or None."""
    df = fetch_binance_history(symbol, days, interval="5m")
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
        return {"symbol": symbol, "trades": 0, "tp": 0, "sl": 0, "open": 0, "net_r": 0.0, "win_rate": 0, "avg_price": avg_price}
    trades, total_r = backtest(df, signals)
    tp_count = sum(1 for t in trades if t["outcome"] == "TP")
    sl_count = sum(1 for t in trades if t["outcome"] == "SL")
    open_count = sum(1 for t in trades if t["outcome"] == "OPEN")
    closed = tp_count + sl_count
    win_rate = round(100.0 * tp_count / closed, 1) if closed else 0
    return {
        "symbol": symbol,
        "trades": len(trades),
        "tp": tp_count,
        "sl": sl_count,
        "open": open_count,
        "net_r": total_r,
        "win_rate": win_rate,
        "avg_price": avg_price,
    }


def _is_bad_result(r):
    """Worst coins: many trades, more SL than TP, low net R."""
    return (
        r["trades"] >= BAD_MIN_TRADES
        and (r["sl"] - r["tp"]) >= BAD_SL_MINUS_TP
        and r["net_r"] < BAD_MAX_NET_R
    )


def main():
    # По умолчанию бэктест считаем по последним 5 дням.
    # При необходимости можно явно передать количество дней аргументом, например: python backtest_5m_trend_binance.py 30
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    symbols_list = get_binance_symbols(min_volume_usdt=MIN_VOLUME_24H_USDT, max_min_notional_usdt=MAX_MIN_NOTIONAL_USDT)
    if not symbols_list:
        print("  Failed to get Binance symbol list.")
        return
    # Фильтр: только монеты с MIN_NOTIONAL <= 5.3 (как в лайв-скрипте)
    notional_map = get_symbols_min_notional_map()
    symbols_list = [s for s in symbols_list if float(notional_map.get(s["symbol"], 999)) <= 5.3]
    print()
    print("=" * 72)
    print("  BACKTEST BINANCE 5m (USDT-M Futures)  Strategy: hybrid   R:R 1:3  SL=0.3R")
    print(f"  Last {days} days  |  {len(symbols_list)} symbols (vol > {MIN_VOLUME_24H_USDT/1e6:.0f}M, MIN_NOTIONAL <= 5.3 USDT)")
    print("=" * 72)
    results = []
    for i, s in enumerate(symbols_list):
        symbol = s["symbol"]
        print(f"  [{i+1}/{len(symbols_list)}] {_safe_console(symbol)}...", end=" ", flush=True)
        row = run_backtest_one_binance(symbol, days, verbose=False)
        if row is None:
            print("no data/skip")
            continue
        results.append(row)
        print(f"trades={row['trades']} TP={row['tp']} SL={row['sl']} Net={row['net_r']:+.1f}R")
        time.sleep(0.2)
    if not results:
        print("  No results.")
        return
    print()
    print("-" * 72)
    print(f"  {'SYMBOL':<14} {'TRADES':>6} {'TP':>4} {'SL':>4} {'OPEN':>4} {'NET R':>8} {'WIN%':>6}")
    print("-" * 72)
    for r in sorted(results, key=lambda x: -x["net_r"]):
        print(f"  {_safe_console(r['symbol']):<14} {r['trades']:>6} {r['tp']:>4} {r['sl']:>4} {r['open']:>4} {r['net_r']:>+8.1f} {r['win_rate']:>5.1f}%")
    print("-" * 72)
    total_trades = sum(r["trades"] for r in results)
    total_tp = sum(r["tp"] for r in results)
    total_sl = sum(r["sl"] for r in results)
    total_open = sum(r["open"] for r in results)
    total_net = sum(r["net_r"] for r in results)
    print(f"  {'TOTAL':<14} {total_trades:>6} {total_tp:>4} {total_sl:>4} {total_open:>4} {total_net:>+8.1f}")
    print(f"  Total net R = {total_net:+.1f}  (SL = {CONFIG.get('SL_R', 0.3)}R per loss)")
    # Exclusions: no trades + bad deals (like MEXC)
    no_trades = [r for r in results if r["trades"] == 0]
    bad_deals = [r for r in results if r["trades"] > 0 and _is_bad_result(r)]
    coins_left = [r for r in results if r["trades"] > 0 and not _is_bad_result(r)]
    print()
    print("  Exclusions (for live / next run):")
    print(f"    - No transactions (0 trades): {len(no_trades)} coins")
    print(f"    - Bad deals (trades>={BAD_MIN_TRADES}, SL-TP>={BAD_SL_MINUS_TP}, net_r<{BAD_MAX_NET_R}): {len(bad_deals)} coins")
    print(f"  Coins left (tradeable, good): {len(coins_left)}")
    if coins_left:
        left_net = sum(r["net_r"] for r in coins_left)
        print(f"  Net R (coins left only): {left_net:+.1f}")
    print("  Done.")


if __name__ == "__main__":
    main()
