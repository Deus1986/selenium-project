"""
Live trading: 5m hybrid strategy on MEXC. Signal -> MARKET order -> auto set TP/SL.

ВАЖНО: На MEXC в настройках API для фьючерсов указано «Размещение ордера (не поддерживается)».
Пока биржа не включит размещение ордеров по API — используйте DRY_RUN=1: скрипт только
находит сигналы и выводит их; ордера выставляете вручную или на другой бирже.

Setup:
  - Secrets: MEXC_API_KEY, MEXC_SECRET in .env (для запросов к данным; ордера пока недоступны).
  - DRY_RUN=1 — только логировать сигналы, ордера не отправлять (рекомендуется, пока MEXC не поддержит).

Run:  python test-scripts/trade_5m_hybrid_live.py
"""
import os
import sys

# Load .env from project root so MEXC_API_KEY, MEXC_SECRET are available (not in repo; CI/CD uses env)
try:
    from dotenv import load_dotenv
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _env_path = os.path.join(_script_dir, "..", ".env")
    load_dotenv(_env_path)
    load_dotenv()  # also cwd
except ImportError:
    pass  # optional: without python-dotenv, use system env only
import json
import time
import hmac
import hashlib
import requests
from datetime import datetime, timedelta

# Import strategy and data from backtest script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_5m_trend_mexc import (
    get_high_volume_symbols,
    fetch_history,
    add_indicators,
    get_signals,
    CONFIG,
    RR_RATIO,
    INTERVAL,
    MIN_BARS,
    EXCLUDED_SYMBOLS,
    format_price,
)

# --- Live config ---
# Position: $1 margin × 5x leverage = $5 notional per trade
POSITION_MARGIN_USD = 1.0
LEVERAGE = 5
POSITION_NOTIONAL_USD = POSITION_MARGIN_USD * LEVERAGE  # $5
OPEN_TYPE = 1   # 1=isolated, 2=cross
API_BASE = "https://contract.mexc.com"
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_5m_hybrid_state.json")
DRY_RUN = os.environ.get("DRY_RUN", "").strip().lower() in ("1", "true", "yes")
TP_SL_MAX_ATTEMPTS = 15
TP_SL_RETRY_DELAY_SEC = 1.0


def _mexc_sign(secret_key: str, access_key: str, timestamp: str, body: str) -> str:
    """Signature for MEXC Contract API: accessKey + timestamp + body_json, HMAC-SHA256 hex."""
    s = access_key + timestamp + body
    return hmac.new(secret_key.encode("utf-8"), s.encode("utf-8"), hashlib.sha256).hexdigest()


def _signed_post(path: str, body: dict):
    """POST to MEXC Contract private API. Returns (ok: bool, data_or_msg)."""
    api_key = os.environ.get("MEXC_API_KEY", "").strip()
    secret = os.environ.get("MEXC_SECRET", "").strip()
    if not api_key or not secret:
        return False, "MEXC_API_KEY / MEXC_SECRET not set"
    body_json = json.dumps(body, separators=(",", ":"))
    timestamp = str(int(time.time() * 1000))
    sig = _mexc_sign(secret, api_key, timestamp, body_json)
    headers = {"ApiKey": api_key, "Request-Time": timestamp, "Signature": sig, "Content-Type": "application/json"}
    try:
        r = requests.post(f"{API_BASE}{path}", data=body_json, headers=headers, timeout=15)
        out = r.json() if r.text else {}
        if r.status_code == 200 and out.get("success") and out.get("code") == 0:
            return True, out.get("data", out)
        return False, out.get("message", out.get("msg", r.text or str(r.status_code)))
    except Exception as e:
        return False, str(e)


def _signed_get(path: str, params: dict = None):
    """GET to MEXC Contract private API. Returns (ok: bool, data_or_msg)."""
    api_key = os.environ.get("MEXC_API_KEY", "").strip()
    secret = os.environ.get("MEXC_SECRET", "").strip()
    if not api_key or not secret:
        return False, "MEXC_API_KEY / MEXC_SECRET not set"
    params = params or {}
    # GET: param string sorted, with & (doc says dictionary order)
    param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()) if v is not None and v != "")
    timestamp = str(int(time.time() * 1000))
    s = api_key + timestamp + param_str
    sig = hmac.new(secret.encode("utf-8"), s.encode("utf-8"), hashlib.sha256).hexdigest()
    headers = {"ApiKey": api_key, "Request-Time": timestamp, "Signature": sig, "Content-Type": "application/json"}
    try:
        url = f"{API_BASE}{path}" + ("?" + param_str if param_str else "")
        r = requests.get(url, headers=headers, timeout=15)
        out = r.json() if r.text else {}
        if r.status_code == 200 and out.get("success") and out.get("code") == 0:
            return True, out.get("data", out)
        return False, out.get("message", out.get("msg", r.text or str(r.status_code)))
    except Exception as e:
        return False, str(e)


def get_contract_detail(symbol: str):
    """Get contract info (contractSize, minVol, volUnit) for symbol."""
    try:
        r = requests.get(f"{API_BASE}/api/v1/contract/detail", params={"symbol": symbol}, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data.get("success") or not data.get("data"):
            return None
        for c in data["data"]:
            if c.get("symbol") == symbol:
                return c
    except Exception:
        pass
    return None


def place_market_order(symbol: str, side: str, vol: int, stop: float, take_profit: float):
    """
    Place MARKET order on MEXC Contract. side = "LONG" | "SHORT".
    Returns (ok: bool, order_id_or_data, message). On success order_id can be used for set_tp_sl.
    """
    # side: 1 = buy (open long), 2 = sell (open short)
    side_id = 1 if side.upper() == "LONG" else 2
    body = {
        "symbol": symbol,
        "vol": vol,
        "leverage": LEVERAGE,
        "side": side_id,
        "type": 2,
        "openType": OPEN_TYPE,
    }
    ok, data = _signed_post("/api/v1/private/order/submit", body)
    if not ok:
        return False, None, data
    # Response may be dict with orderId / id, or direct id
    order_id = None
    if isinstance(data, dict):
        order_id = data.get("orderId") or data.get("id") or data.get("order_id")
    if order_id is None and isinstance(data, (int, str)):
        order_id = data
    return True, order_id, data


def get_open_position_id(symbol: str, is_long: bool):
    """Get current open position id for symbol. is_long: True = long position. Returns positionId or None."""
    ok, data = _signed_get("/api/v1/private/position/list/history_positions", {"symbol": symbol, "page_num": 1, "page_size": 20})
    if not ok or not data:
        return None
    # API may return { resultList: [...] } or list; position has positionId, holdSide (1=long 2=short)
    result_list = data.get("resultList", data) if isinstance(data, dict) else data
    if not isinstance(result_list, list):
        return None
    side_match = 1 if is_long else 2
    for pos in result_list:
        if pos.get("holdSide") == side_match and int(pos.get("holdVol", 0) or 0) > 0:
            return pos.get("positionId") or pos.get("id")
    return None


def set_tp_sl_after_order(symbol: str, order_or_position_id, stop: float, take_profit: float, is_long: bool):
    """
    Set Take Profit and Stop Loss immediately after opening. Uses stoporder/change_price.
    order_or_position_id: from place_market_order response or get_open_position_id.
    Returns (ok: bool, message: str).
    """
    # Format prices as strings (MEXC often expects string for precision)
    def fmt(p):
        if p is None or p == "":
            return None
        return f"{float(p):.8f}".rstrip("0").rstrip(".")
    stop_s = fmt(stop)
    tp_s = fmt(take_profit)
    if not stop_s and not tp_s:
        return False, "Need at least one of stop or take_profit"
    body = {}
    if order_or_position_id is not None:
        body["orderId"] = str(order_or_position_id)
    if stop_s:
        body["stopLossPrice"] = stop_s
    if tp_s:
        body["takeProfitPrice"] = tp_s
    if not body:
        return False, "No orderId or prices"
    ok, msg = _signed_post("/api/v1/private/stoporder/change_price", body)
    if ok:
        return True, "TP/SL set"
    # If change_price expects positionId, try with position after short delay
    if "5002" in str(msg) or "not exist" in str(msg).lower() or "position" in str(msg).lower():
        time.sleep(1.0)
        pos_id = get_open_position_id(symbol, is_long)
        if pos_id is not None:
            body2 = {}
            if stop_s:
                body2["stopLossPrice"] = stop_s
            if tp_s:
                body2["takeProfitPrice"] = tp_s
            body2["orderId"] = str(pos_id)
            ok2, msg2 = _signed_post("/api/v1/private/stoporder/change_price", body2)
            if ok2:
                return True, "TP/SL set (by position)"
            return False, msg2
    return False, msg


def close_position_market(symbol: str, is_long: bool, vol: int):
    """
    Close position by opposite market order: long -> sell (side=2), short -> buy (side=1).
    Returns (ok: bool, message: str).
    """
    # side 1 = buy, 2 = sell. To close long we sell; to close short we buy.
    side_id = 2 if is_long else 1
    body = {
        "symbol": symbol,
        "vol": vol,
        "leverage": LEVERAGE,
        "side": side_id,
        "type": 2,
        "openType": OPEN_TYPE,
    }
    ok, msg = _signed_post("/api/v1/private/order/submit", body)
    if ok:
        return True, "Position closed"
    return False, msg


def set_tp_sl_with_retries(symbol: str, order_id, stop: float, take_profit: float, is_long: bool, vol: int):
    """
    Try up to TP_SL_MAX_ATTEMPTS times to set TP/SL. If all fail, close the position immediately.
    Returns (tp_sl_ok: bool, closed_anyway: bool, message: str).
    """
    oid = order_id
    for attempt in range(1, TP_SL_MAX_ATTEMPTS + 1):
        if oid is None or attempt > 1:
            if attempt > 1:
                time.sleep(TP_SL_RETRY_DELAY_SEC)
            oid = get_open_position_id(symbol, is_long) or oid or order_id
        if oid is not None:
            ok, msg = set_tp_sl_after_order(symbol, oid, stop, take_profit, is_long)
            if ok:
                return True, False, "TP/SL set"
        if attempt < TP_SL_MAX_ATTEMPTS:
            time.sleep(TP_SL_RETRY_DELAY_SEC)
    # All 15 attempts failed — close position immediately to avoid unprotected loss
    close_ok, close_msg = close_position_market(symbol, is_long, vol)
    if close_ok:
        return False, True, "TP/SL failed after 15 attempts; position closed to limit risk."
    return False, True, f"TP/SL failed; close position failed: {close_msg}"


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
        print("  Save state error:", e)


def wait_until_next_5m():
    """Wait until 1–2 seconds after next 5m bar close (e.g. 12:05:02)."""
    now = datetime.now()
    # current 5m slot end
    minute = (now.minute // 5) * 5
    bar_end = now.replace(minute=minute, second=0, microsecond=0) + timedelta(minutes=5)
    # add 2 sec for exchange to close bar
    target = bar_end + timedelta(seconds=2)
    if now >= target:
        target = target + timedelta(minutes=5)
    sec = (target - now).total_seconds()
    if sec > 0:
        print(f"  Next check at {target.strftime('%H:%M:%S')} (in {int(sec)} s)")
        time.sleep(sec)


def run_cycle(state_set):
    """Fetch last ~30 days 5m, get signals; return only signals on the very last bar (new)."""
    symbols = get_high_volume_symbols(10_000_000)
    new_signals = []
    for s in symbols:
        api_symbol = s["symbol"]
        try:
            df = fetch_history(api_symbol, days=30, interval=INTERVAL)
            if df is None or len(df) < MIN_BARS:
                continue
            df = add_indicators(df)
            signals = get_signals(df)
            if not signals:
                continue
            # only the last signal (most recent bar)
            last = signals[-1]
            bar_idx = last["bar_index"]
            if bar_idx < len(df) - 2:
                continue
            key = f"{api_symbol}|{last['direction']}|{last.get('signal_time', '')}"
            if key in state_set:
                continue
            state_set.add(key)
            new_signals.append({
                "symbol": api_symbol,
                "direction": last["direction"],
                "entry": last["entry"],
                "stop": last["stop"],
                "take_profit": last["take_profit"],
                "signal_time": last.get("signal_time", ""),
            })
        except Exception as e:
            print(f"  {api_symbol}: {e}")
        time.sleep(0.2)
    return state_set, new_signals


def main():
    print()
    print("=" * 64)
    print("  5m HYBRID LIVE — MEXC")
    print("  Strategy: hybrid  |  Entry: MARKET  |  R:R 1:3  SL=0.3R")
    print(f"  Position: ${POSITION_MARGIN_USD:.0f} margin x {LEVERAGE} = ${POSITION_NOTIONAL_USD:.0f} notional")
    print("  Start:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if DRY_RUN:
        print("  *** DRY RUN: only signals, no orders (recommended while MEXC disables order API) ***")
    else:
        print("  WARNING: MEXC Futures API may have 'Place order (not supported)'. If orders fail, use DRY_RUN=1.")
        if not os.environ.get("MEXC_API_KEY") or not os.environ.get("MEXC_SECRET"):
            print("  Set MEXC_API_KEY and MEXC_SECRET in .env to send orders (when supported).")
    print("=" * 64)
    print()

    state_set = load_state()
    while True:
        wait_until_next_5m()
        print(f"\n  Check at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        state_set, new_signals = run_cycle(state_set)
        if not new_signals:
            print("  No new signals.")
            continue
        save_state(state_set)
        for sig in new_signals:
            sym = sig["symbol"]
            direction = sig["direction"]
            entry = sig["entry"]
            stop = sig["stop"]
            tp = sig["take_profit"]
            print(f"  [SIGNAL] {sym}  {direction}  Entry:{format_price(entry)}  SL:{format_price(stop)}  TP:{format_price(tp)}  | {sig['signal_time']}")
            if DRY_RUN:
                print("    (DRY_RUN — order skipped)")
                continue
            # Position size: notional = $5 (margin $1 × 5x). vol = notional / (contractSize * entry)
            detail = get_contract_detail(sym)
            if detail:
                cs = float(detail.get("contractSize", 1) or 1)
                min_vol = int(detail.get("minVol", 1) or 1)
                if cs > 0 and entry > 0:
                    vol = max(min_vol, int(round(POSITION_NOTIONAL_USD / (cs * entry))))
                else:
                    vol = min_vol
            else:
                vol = 1
            ok, order_id, msg = place_market_order(sym, direction, vol, stop, tp)
            if ok:
                print(f"    Order sent. Vol={vol}. Setting TP/SL (up to {TP_SL_MAX_ATTEMPTS} attempts)...")
                time.sleep(0.5)
                is_long = direction.upper() == "LONG"
                tp_sl_ok, closed_anyway, tp_sl_msg = set_tp_sl_with_retries(sym, order_id, stop, tp, is_long, vol)
                if tp_sl_ok:
                    print(f"    TP/SL set automatically.")
                elif closed_anyway:
                    print(f"    {tp_sl_msg}")
                else:
                    print(f"    TP/SL failed: {tp_sl_msg}")
            else:
                print(f"    Order failed: {msg}")


if __name__ == "__main__":
    main()
