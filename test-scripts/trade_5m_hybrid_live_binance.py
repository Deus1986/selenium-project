"""
Live trading: 5m hybrid strategy на Binance USDT-M Futures.
Сигнал -> рыночный ордер -> авто-установка TP и SL (STOP_MARKET / TAKE_PROFIT_MARKET).

Setup:
  - В .env: BINANCE_API_KEY, BINANCE_SECRET (от Binance Futures API).
  - MIN_POSITION_NOTIONAL_USD=5 — номинал позиции $5 ($1 маржа при 5x). Все 167 монет (не трогаем).
  - DRY_RUN=1 — только сигналы, ордера не отправлять.
  - ONLY_TOP_COINS=1 — только топ-монеты по TP (по умолчанию выключено, торгуем всеми 167).

Запуск:  python test-scripts/trade_5m_hybrid_live_binance.py
"""
import os
import sys
import time
import json
import hmac
import hashlib
import urllib.parse
import requests
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _env_path = os.path.join(_script_dir, "..", ".env")
    load_dotenv(_env_path)
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_5m_trend_mexc import (
    add_indicators,
    get_signals,
    CONFIG,
    RR_RATIO,
    MIN_BARS,
    format_price,
)
from backtest_5m_trend_binance import (
    get_binance_symbols,
    fetch_binance_history,
    BINANCE_BASE,
    MIN_VOLUME_24H_USDT,
)

# --- Конфиг ---
# Минимальная позиция для теста: в .env задайте MIN_POSITION_NOTIONAL_USD=2 ($0.4 маржа при 5x).
# По умолчанию $5 номинал ($1 маржа). Binance часто min notional = 5; если 2 — только пары с min <= 2.
_def_notional = os.environ.get("MIN_POSITION_NOTIONAL_USD", "").strip()
try:
    POSITION_NOTIONAL_USD = float(_def_notional) if _def_notional else 5.0
except ValueError:
    POSITION_NOTIONAL_USD = 5.0
if POSITION_NOTIONAL_USD < 2:
    POSITION_NOTIONAL_USD = 2.0
LEVERAGE = 5
POSITION_MARGIN_USD = POSITION_NOTIONAL_USD / LEVERAGE
# Только монеты с большим числом TP (топ бэктеста): задайте ONLY_TOP_COINS=1 в .env для теста
ONLY_TOP_COINS = os.environ.get("ONLY_TOP_COINS", "").strip().lower() in ("1", "true", "yes")
# Топ по TP/Net R из бэктеста 30d (можно торговать с минимальной позицией для проверки скрипта)
TOP_COINS_BY_TP = [
    "SIGNUSDT", "1000SHIBUSDT", "ALCHUSDT", "SIRENUSDT", "PARTIUSDT", "MAGICUSDT", "TRUMPUSDT", "RVVUSDT",
    "PTBUSDT", "DYDXUSDT", "LAUSDT", "XPLUSDT", "XAGUSDT", "MERLUSDT", "WCTUSDT", "SOMIUSDT", "TIAUSDT",
    "ZILUSDT", "BERAUSDT", "STXUSDT", "SUIUSDT", "FETUSDT", "PAXGUSDT", "XRPUSDT", "ETCUSDT", "CYSUSDT",
    "GIGGLEUSDT", "WIFUSDT", "C98USDT", "COMMONUSDT", "BULLAUSDT", "HUSDT", "ZKUSDT", "DOTUSDT", "TAOUSDT",
    "TONUSDT", "PENGUUSDT", "PENDLEUSDT", "ALGOUSDT", "BREVUSDT",
]
DRY_RUN = os.environ.get("DRY_RUN", "").strip().lower() in ("1", "true", "yes")
# Фактический номинал для ордера (если для $2 нет пар — используем $5)
EFFECTIVE_NOTIONAL_USD = None  # задаётся в main()
TP_SL_MAX_ATTEMPTS = 15
TP_SL_RETRY_DELAY_SEC = 1.0
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_5m_hybrid_binance_state.json")
SIGNALS_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_5m_hybrid_binance_signals.json")
API_TIMEOUT = 30


def _binance_sign(secret: str, query: str) -> str:
    return hmac.new(secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()


def _signed_request(method: str, path: str, params: dict = None):
    """GET или POST с подписью. Параметры в query string, signature добавляется в конец."""
    api_key = os.environ.get("BINANCE_API_KEY", "").strip()
    secret = os.environ.get("BINANCE_SECRET", "").strip()
    if not api_key or not secret:
        return False, "BINANCE_API_KEY / BINANCE_SECRET not set"
    params = dict(params or {})
    params["timestamp"] = int(time.time() * 1000)
    query = urllib.parse.urlencode(sorted(params.items()))
    sig = _binance_sign(secret, query)
    url = f"{BINANCE_BASE}{path}?{query}&signature={sig}"
    headers = {"X-MBX-APIKEY": api_key}
    try:
        if method == "GET":
            r = requests.get(url, headers=headers, timeout=API_TIMEOUT)
        else:
            r = requests.post(url, headers=headers, timeout=API_TIMEOUT)
        data = r.json() if r.text else {}
        if r.status_code != 200:
            return False, data.get("msg", data.get("code", r.text))
        if isinstance(data, dict) and "code" in data and data.get("code") != 0:
            return False, data.get("msg", data)
        return True, data
    except Exception as e:
        return False, str(e)


def get_symbol_filters():
    """Кэш: символ -> {stepSize, minQty} из exchangeInfo LOT_SIZE."""
    url = f"{BINANCE_BASE}/fapi/v1/exchangeInfo"
    try:
        r = requests.get(url, timeout=API_TIMEOUT)
        if r.status_code != 200:
            return {}
        data = r.json()
        out = {}
        for s in data.get("symbols", []):
            sym = s.get("symbol", "")
            for f in s.get("filters", []):
                if f.get("filterType") == "LOT_SIZE":
                    out[sym] = {
                        "stepSize": float(f.get("stepSize", 0.001)),
                        "minQty": float(f.get("minQty", 0)),
                    }
                    break
        return out
    except Exception:
        return {}


def round_quantity(qty: float, step: float, min_qty: float) -> float:
    if step <= 0:
        return max(min_qty, qty)
    rounded = round(qty / step) * step
    rounded = round(rounded, 8)
    return max(min_qty, rounded)


def set_leverage(symbol: str, leverage: int):
    ok, msg = _signed_request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})
    return ok


def place_market_order(symbol: str, side: str, quantity: float):
    """Рыночный ордер. side: BUY | SELL. quantity в базе. Возвращает (ok, orderId или msg)."""
    # newOrderRespType=RESULT чтобы получить исполненную цену при необходимости
    ok, data = _signed_request("POST", "/fapi/v1/order", {
        "symbol": symbol,
        "side": side.upper(),
        "type": "MARKET",
        "quantity": quantity,
    })
    if not ok:
        return False, data
    order_id = data.get("orderId")
    return True, order_id


def place_stop_order(symbol: str, side: str, order_type: str, stop_price: float):
    """Условный ордер: STOP_MARKET или TAKE_PROFIT_MARKET с closePosition=true."""
    ok, data = _signed_request("POST", "/fapi/v1/order", {
        "symbol": symbol,
        "side": side.upper(),
        "type": order_type,
        "stopPrice": stop_price,
        "closePosition": "true",
        "workingType": "CONTRACT_PRICE",
    })
    return ok, data


def get_position_amt(symbol: str):
    """Текущая позиция по символу (one-way). Возвращает (positionAmt, side): + long, - short, 0 нет."""
    ok, data = _signed_request("GET", "/fapi/v2/positionRisk", {"symbol": symbol})
    if not ok or not isinstance(data, list):
        return 0.0, None
    for p in data:
        if p.get("symbol") == symbol:
            amt = float(p.get("positionAmt", 0))
            if amt == 0:
                return 0.0, None
            return amt, "LONG" if amt > 0 else "SHORT"
    return 0.0, None


def close_position_market(symbol: str, is_long: bool, quantity: float):
    """Закрыть позицию рыночным ордером reduceOnly."""
    side = "SELL" if is_long else "BUY"
    ok, msg = _signed_request("POST", "/fapi/v1/order", {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": abs(quantity),
        "reduceOnly": "true",
    })
    return ok, msg


def set_tp_sl_with_retries(symbol: str, stop: float, take_profit: float, is_long: bool, quantity: float):
    """
    Выставить TP и SL двумя ордерами (TAKE_PROFIT_MARKET, STOP_MARKET).
    Если после TP_SL_MAX_ATTEMPTS не получилось — закрыть позицию.
    """
    close_side = "SELL" if is_long else "BUY"
    for attempt in range(1, TP_SL_MAX_ATTEMPTS + 1):
        ok_tp, _ = place_stop_order(symbol, close_side, "TAKE_PROFIT_MARKET", take_profit)
        ok_sl, _ = place_stop_order(symbol, close_side, "STOP_MARKET", stop)
        if ok_tp and ok_sl:
            return True, False, "TP и SL выставлены"
        if attempt < TP_SL_MAX_ATTEMPTS:
            time.sleep(TP_SL_RETRY_DELAY_SEC)
    close_ok, close_msg = close_position_market(symbol, is_long, quantity)
    if close_ok:
        return False, True, "TP/SL не удалось выставить после попыток; позиция закрыта."
    return False, True, f"TP/SL не удалось; закрыть не получилось: {close_msg}"


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
        print("  Ошибка сохранения state:", e)


def append_signal_to_log(signal_dict: dict, sent_at: str = None):
    """Добавить отправленный сигнал в JSON-лог для статистики (монета, вход, SL, TP; outcome — позже TP/SL)."""
    if sent_at is None:
        sent_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = float(signal_dict.get("entry", 0))
    stop = float(signal_dict.get("stop", 0))
    tp = float(signal_dict.get("take_profit", 0))
    record = {
        "symbol": signal_dict.get("symbol", ""),
        "direction": signal_dict.get("direction", ""),
        "entry": round(entry, 8),
        "stop": round(stop, 8),
        "take_profit": round(tp, 8),
        "signal_time": signal_dict.get("signal_time", ""),
        "sent_at": sent_at,
        "outcome": None,
    }
    try:
        log = []
        if os.path.isfile(SIGNALS_LOG_FILE):
            with open(SIGNALS_LOG_FILE, "r", encoding="utf-8") as f:
                log = json.load(f)
        if not isinstance(log, list):
            log = []
        log.append(record)
        with open(SIGNALS_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("  Ошибка записи лога сигналов:", e)


def wait_until_next_5m():
    now = datetime.now()
    minute = (now.minute // 5) * 5
    bar_end = now.replace(minute=minute, second=0, microsecond=0) + timedelta(minutes=5)
    target = bar_end + timedelta(seconds=2)
    if now >= target:
        target = target + timedelta(minutes=5)
    sec = (target - now).total_seconds()
    if sec > 0:
        print(f"  Следующая проверка в {target.strftime('%H:%M:%S')} (через {int(sec)} с)")
        time.sleep(sec)


def run_cycle(state_set):
    """Сбор сигналов: символы с min notional <= эффективный номинал; при ONLY_TOP_COINS — только топ по TP."""
    notional = EFFECTIVE_NOTIONAL_USD if EFFECTIVE_NOTIONAL_USD is not None else POSITION_NOTIONAL_USD
    symbols = get_binance_symbols(min_volume_usdt=MIN_VOLUME_24H_USDT, max_min_notional_usdt=notional)
    if ONLY_TOP_COINS and symbols:
        top_set = set(TOP_COINS_BY_TP)
        symbols = [s for s in symbols if s["symbol"] in top_set]
    new_signals = []
    for s in symbols:
        symbol = s["symbol"]
        try:
            df = fetch_binance_history(symbol, days=30, interval="5m")
            if df is None or len(df) < MIN_BARS:
                continue
            df = add_indicators(df)
            signals = get_signals(df)
            if not signals:
                continue
            last = signals[-1]
            bar_idx = last["bar_index"]
            if bar_idx < len(df) - 2:
                continue
            key = f"{symbol}|{last['direction']}|{last.get('signal_time', '')}"
            if key in state_set:
                continue
            state_set.add(key)
            new_signals.append({
                "symbol": symbol,
                "direction": last["direction"],
                "entry": last["entry"],
                "stop": last["stop"],
                "take_profit": last["take_profit"],
                "signal_time": last.get("signal_time", ""),
            })
        except Exception as e:
            print(f"  {symbol}: {e}")
        time.sleep(0.15)
    return state_set, new_signals


def _safe_console(s):
    if s is None:
        return ""
    return s.encode("ascii", "replace").decode("ascii")


def main():
    print()
    print("=" * 64)
    print("  5m HYBRID LIVE — Binance USDT-M Futures")
    print("  Стратегия: hybrid  |  Вход: MARKET  |  R:R 1:3  SL=0.3R")
    print(f"  Позиция: ${POSITION_MARGIN_USD:.2f} маржа x {LEVERAGE} = ${POSITION_NOTIONAL_USD:.2f} номинал (мин. позиция для теста)")
    if ONLY_TOP_COINS:
        print(f"  Режим: только топ-монеты по TP ({len(TOP_COINS_BY_TP)} шт.) — ONLY_TOP_COINS=1")
    # Если для минимального номинала ($2) нет пар — используем $5
    global EFFECTIVE_NOTIONAL_USD
    _syms = get_binance_symbols(min_volume_usdt=MIN_VOLUME_24H_USDT, max_min_notional_usdt=POSITION_NOTIONAL_USD)
    if ONLY_TOP_COINS and _syms:
        _syms = [s for s in _syms if s["symbol"] in set(TOP_COINS_BY_TP)]
    if not _syms and POSITION_NOTIONAL_USD < 5:
        EFFECTIVE_NOTIONAL_USD = 5.0
        print(f"  Для номинала ${POSITION_NOTIONAL_USD:.0f} нет подходящих пар; используем ${EFFECTIVE_NOTIONAL_USD:.0f} номинал.")
    else:
        EFFECTIVE_NOTIONAL_USD = POSITION_NOTIONAL_USD
    print("  Старт:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if DRY_RUN:
        print("  *** DRY_RUN: только сигналы, ордера НЕ отправляются ***")
    else:
        print("  Режим: РЕАЛЬНЫЕ ОРДЕРА (вход + авто TP/SL)")
        if not os.environ.get("BINANCE_API_KEY") or not os.environ.get("BINANCE_SECRET"):
            print("  Задайте BINANCE_API_KEY и BINANCE_SECRET в .env для отправки ордеров.")
    print("=" * 64)
    print()

    filters_cache = get_symbol_filters()
    state_set = load_state()

    while True:
        wait_until_next_5m()
        print(f"\n  Проверка {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        state_set, new_signals = run_cycle(state_set)
        if not new_signals:
            print("  Новых сигналов нет.")
            continue
        save_state(state_set)

        for sig in new_signals:
            symbol = sig["symbol"]
            direction = sig["direction"]
            entry = float(sig["entry"])
            stop = float(sig["stop"])
            tp = float(sig["take_profit"])
            print(f"  [СИГНАЛ] {_safe_console(symbol)}  {direction}  Entry:{format_price(entry)}  SL:{format_price(stop)}  TP:{format_price(tp)}  | {sig['signal_time']}")
            append_signal_to_log(sig)

            if DRY_RUN:
                print("    (DRY_RUN — ордер не отправлен)")
                continue

            # Размер позиции: номинал (effective) -> quantity = notional / entry (в базе)
            notional = EFFECTIVE_NOTIONAL_USD if EFFECTIVE_NOTIONAL_USD is not None else POSITION_NOTIONAL_USD
            if entry <= 0:
                print("    Пропуск: некорректная цена входа")
                continue
            raw_qty = notional / entry
            filt = filters_cache.get(symbol, {})
            step = filt.get("stepSize", 0.001)
            min_qty = filt.get("minQty", 0)
            quantity = round_quantity(raw_qty, step, min_qty)
            if quantity <= 0:
                quantity = min_qty if min_qty > 0 else raw_qty

            if not set_leverage(symbol, LEVERAGE):
                print("    Не удалось выставить плечо, пропуск")
                continue

            side = "BUY" if direction.upper() == "LONG" else "SELL"
            ok, order_id_or_msg = place_market_order(symbol, side, quantity)
            if not ok:
                print(f"    Ордер не отправлен: {order_id_or_msg}")
                continue

            print(f"    Ордер отправлен. Vol={quantity}. Выставляю TP/SL (до {TP_SL_MAX_ATTEMPTS} попыток)...")
            time.sleep(0.5)
            is_long = direction.upper() == "LONG"
            tp_sl_ok, closed_anyway, tp_sl_msg = set_tp_sl_with_retries(symbol, stop, tp, is_long, quantity)
            if tp_sl_ok:
                print("    TP и SL выставлены.")
                print(f"    >>> Вход: {format_price(entry)}  |  SL: {format_price(stop)}  |  TP: {format_price(tp)}  <<<")
            elif closed_anyway:
                print(f"    {tp_sl_msg}")
            else:
                print(f"    Ошибка TP/SL: {tp_sl_msg}")


if __name__ == "__main__":
    main()
