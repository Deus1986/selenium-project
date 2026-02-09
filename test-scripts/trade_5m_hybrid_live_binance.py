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
import math
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    get_symbols_min_notional_map,
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
MIN_NOTIONAL_MAP = {}


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
    """Кэш: символ -> {stepSize, minQty, tickSize} из exchangeInfo LOT_SIZE/PRICE_FILTER."""
    url = f"{BINANCE_BASE}/fapi/v1/exchangeInfo"
    try:
        r = requests.get(url, timeout=API_TIMEOUT)
        if r.status_code != 200:
            return {}
        data = r.json()
        out = {}
        for s in data.get("symbols", []):
            sym = s.get("symbol", "")
            f_obj = {"stepSize": 0.001, "minQty": 0.0, "tickSize": 0.0001}
            for f in s.get("filters", []):
                ftype = f.get("filterType")
                if ftype == "LOT_SIZE":
                    f_obj["stepSize"] = float(f.get("stepSize", 0.001))
                    f_obj["minQty"] = float(f.get("minQty", 0))
                elif ftype == "PRICE_FILTER":
                    try:
                        f_obj["tickSize"] = float(f.get("tickSize", 0.0001))
                    except (TypeError, ValueError):
                        pass
            out[sym] = f_obj
        return out
    except Exception:
        return {}


def round_quantity(qty: float, step: float, min_qty: float) -> float:
    """Простое округление по шагу Binance (stepSize)."""
    if step <= 0:
        return max(min_qty, qty)
    steps = round(qty / step)
    return round(steps * step, 8)


def round_price_to_tick(price: float, tick: float) -> float:
    """Округление цены к разрешённому шагу tickSize."""
    if tick <= 0:
        return price
    steps = round(price / tick)
    return round(steps * tick, 8)


def set_leverage(symbol: str, leverage: int):
    ok, msg = _signed_request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})
    return ok


def set_margin_isolated(symbol: str):
    """
    Включить изолированную маржу для символа.
    Если уже ISOLATED / нет необходимости менять тип маржи — считаем это успехом.
    """
    ok, data = _signed_request("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": "ISOLATED"})
    if ok:
        return True
    # Binance может вернуть ошибку вида 'No need to change margin type.' — это ок.
    text = str(data)
    if "No need to change margin type" in text or "no need to change margin type" in text:
        return True
    return False


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


def place_take_profit_limit(symbol: str, side: str, price: float, quantity: float):
    """TP как обычный лимитный ордер reduceOnly.
    
    ВАЖНО: Проверяем, что позиция открыта перед выставлением reduceOnly ордера.
    """
    # Проверяем, что позиция действительно открыта
    position_amt, position_side = get_position_amt(symbol)
    if position_amt == 0:
        return False, "Позиция не открыта, нельзя выставить reduceOnly ордер"
    
    # Убеждаемся, что side правильный для закрытия позиции
    if position_side == "LONG" and side.upper() != "SELL":
        return False, f"Неправильный side для LONG позиции: нужен SELL, получен {side}"
    if position_side == "SHORT" and side.upper() != "BUY":
        return False, f"Неправильный side для SHORT позиции: нужен BUY, получен {side}"
    
    ok, data = _signed_request("POST", "/fapi/v1/order", {
        "symbol": symbol,
        "side": side.upper(),
        "type": "LIMIT",
        "price": price,
        "timeInForce": "GTC",
        "quantity": abs(quantity),
        "reduceOnly": "true",
    })
    return ok, data


def place_stop_order(symbol: str, side: str, order_type: str, stop_price: float, quantity: float = None):
    """Условный ордер STOP_MARKET через обычный order endpoint.
    
    ВАЖНО: closePosition и reduceOnly взаимоисключающие - используем только closePosition.
    Пробуем сначала с MARK_PRICE, потом с CONTRACT_PRICE.
    """
    params = {
        "symbol": symbol,
        "side": side.upper(),
        "type": "STOP_MARKET",
        "stopPrice": stop_price,
        "closePosition": "true",  # Закрыть всю позицию (НЕ используем reduceOnly вместе с closePosition!)
    }
    
    # Пробуем сначала MARK_PRICE
    params["workingType"] = "MARK_PRICE"
    ok, data = _signed_request("POST", "/fapi/v1/order", params)
    if ok:
        return ok, data
    
    # Если не получилось, пробуем CONTRACT_PRICE
    params["workingType"] = "CONTRACT_PRICE"
    ok, data = _signed_request("POST", "/fapi/v1/order", params)
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


def get_all_open_positions():
    """Получить все открытые позиции. Возвращает список dict с symbol, positionAmt, entryPrice, markPrice."""
    ok, data = _signed_request("GET", "/fapi/v2/positionRisk", {})
    if not ok or not isinstance(data, list):
        return []
    positions = []
    for p in data:
        amt = float(p.get("positionAmt", 0))
        if amt != 0:
            positions.append({
                "symbol": p.get("symbol"),
                "positionAmt": amt,
                "entryPrice": float(p.get("entryPrice", 0)),
                "markPrice": float(p.get("markPrice", 0)),
                "isLong": amt > 0,
            })
    return positions


def get_open_orders(symbol: str):
    """Получить открытые ордера по символу. Возвращает список ордеров."""
    ok, data = _signed_request("GET", "/fapi/v1/openOrders", {"symbol": symbol})
    if not ok:
        return []
    if not isinstance(data, list):
        return []
    return data


def has_tp_sl_orders(symbol: str):
    """Проверить, есть ли у позиции TP/SL ордера. Возвращает (has_tp, has_sl)."""
    orders = get_open_orders(symbol)
    has_tp = False
    has_sl = False
    position_amt, position_side = get_position_amt(symbol)
    if position_amt == 0:
        return False, False
    
    is_long = position_side == "LONG"
    
    for order in orders:
        order_type = order.get("type", "").upper()
        reduce_only = order.get("reduceOnly", False)
        
        # TAKE_PROFIT_MARKET или TAKE_PROFIT - это TP
        if order_type in ["TAKE_PROFIT_MARKET", "TAKE_PROFIT"]:
            has_tp = True
        # STOP_MARKET с reduceOnly - это SL
        elif order_type == "STOP_MARKET" and reduce_only:
            has_sl = True
        # LIMIT ордер с reduceOnly=true - это наш TP (как мы выставляем в place_take_profit_limit)
        elif order_type == "LIMIT" and reduce_only:
            has_tp = True
    
    return has_tp, has_sl


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
    # Проверяем, что позиция действительно открылась
    position_amt, position_side = get_position_amt(symbol)
    if position_amt == 0:
        # Позиция еще не открылась, ждем еще немного
        time.sleep(1.0)
        position_amt, position_side = get_position_amt(symbol)
        if position_amt == 0:
            return False, False, f"Позиция по {symbol} не открылась, TP/SL не выставлены"
    
    close_side = "SELL" if is_long else "BUY"
    last_tp_error = None
    last_sl_error = None
    for attempt in range(1, TP_SL_MAX_ATTEMPTS + 1):
        # TP: лимитный ордер reduceOnly по цене take_profit.
        ok_tp, tp_data = place_take_profit_limit(symbol, close_side, take_profit, quantity)
        # SL: STOP_MARKET с триггером по стоп-цене.
        ok_sl, sl_data = place_stop_order(symbol, close_side, "STOP_MARKET", stop, quantity)
        if ok_tp and ok_sl:
            return True, False, "TP и SL выставлены"
        if not ok_tp:
            last_tp_error = tp_data if isinstance(tp_data, str) else str(tp_data)
        if not ok_sl:
            last_sl_error = sl_data if isinstance(sl_data, str) else str(sl_data)
        if attempt < TP_SL_MAX_ATTEMPTS:
            time.sleep(TP_SL_RETRY_DELAY_SEC)
    # Детальный вывод ошибок перед закрытием позиции
    error_msg = f"TP/SL не удалось выставить после {TP_SL_MAX_ATTEMPTS} попыток."
    if last_tp_error:
        error_msg += f" TP ошибка: {last_tp_error}"
    if last_sl_error:
        error_msg += f" SL ошибка: {last_sl_error}"
    close_ok, close_msg = close_position_market(symbol, is_long, quantity)
    if close_ok:
        return False, True, error_msg + " Позиция закрыта."
    return False, True, error_msg + f" Закрыть не получилось: {close_msg}"


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


def get_signal_from_log(symbol: str, is_long: bool):
    """Найти последний сигнал для символа в логе. Возвращает (stop, tp) или (None, None)."""
    try:
        if not os.path.isfile(SIGNALS_LOG_FILE):
            return None, None
        with open(SIGNALS_LOG_FILE, "r", encoding="utf-8") as f:
            log = json.load(f)
        if not isinstance(log, list):
            return None, None
        # Ищем последний сигнал для этого символа с правильным направлением
        direction = "LONG" if is_long else "SHORT"
        for record in reversed(log):
            if record.get("symbol") == symbol and record.get("direction", "").upper() == direction:
                stop = record.get("stop")
                tp = record.get("take_profit")
                if stop and tp:
                    return float(stop), float(tp)
    except Exception as e:
        print(f"  Ошибка чтения лога сигналов для {symbol}:", e)
    return None, None


def calculate_tp_sl_from_entry(symbol: str, entry_price: float, is_long: bool):
    """Пересчитать TP/SL на основе entry price и текущего ATR. Возвращает (stop, tp)."""
    try:
        # Получаем историю для расчета ATR
        df = fetch_binance_history(symbol, "5m", days=5)
        if df is None or len(df) < MIN_BARS:
            return None, None
        df = add_indicators(df)
        if "atr" not in df.columns or len(df) == 0:
            return None, None
        last_atr = df["atr"].iloc[-1]
        if pd.isna(last_atr) or last_atr <= 0:
            last_atr = entry_price * 0.01  # Fallback
        
        stop_atr_mul = CONFIG.get("STOP_ATR_MUL", 0.3)
        if is_long:
            stop = entry_price - last_atr * stop_atr_mul
            risk = entry_price - stop
            tp = entry_price + risk * RR_RATIO
        else:
            stop = entry_price + last_atr * stop_atr_mul
            risk = stop - entry_price
            tp = entry_price - risk * RR_RATIO
        return stop, tp
    except Exception as e:
        print(f"  Ошибка расчета TP/SL для {symbol}:", e)
    return None, None


def restore_tp_sl_for_positions():
    """Проверить все открытые позиции и выставить TP/SL для тех, у которых их нет."""
    positions = get_all_open_positions()
    if not positions:
        return
    
    filters_cache = get_symbol_filters()
    restored_count = 0
    
    for pos in positions:
        # Пропускаем, если позиция слишком маленькая (может быть остаток от закрытой позиции)
        if abs(pos["positionAmt"]) < 0.001:
            continue
        symbol = pos["symbol"]
        is_long = pos["isLong"]
        entry_price = pos["entryPrice"]
        position_amt = abs(pos["positionAmt"])
        
        # Проверяем, есть ли уже TP/SL ордера
        has_tp, has_sl = has_tp_sl_orders(symbol)
        if has_tp and has_sl:
            continue  # Уже есть TP/SL, пропускаем
        
        print(f"  [ВОССТАНОВЛЕНИЕ] {_safe_console(symbol)}: позиция без TP/SL, пытаюсь восстановить...")
        
        # Пытаемся найти stop/tp в логе сигналов
        stop, tp = get_signal_from_log(symbol, is_long)
        
        # Если не нашли в логе - пересчитываем на основе entry и ATR
        if stop is None or tp is None:
            stop, tp = calculate_tp_sl_from_entry(symbol, entry_price, is_long)
        
        if stop is None or tp is None:
            print(f"    Не удалось определить stop/tp для {_safe_console(symbol)}")
            continue
        
        # Округляем цены по tickSize
        filt_prices = filters_cache.get(symbol, {})
        tick = filt_prices.get("tickSize", 0.0001)
        stop = round_price_to_tick(stop, tick)
        tp = round_price_to_tick(tp, tick)
        
        # Выставляем TP/SL
        ok, closed, msg = set_tp_sl_with_retries(symbol, stop, tp, is_long, position_amt)
        if ok:
            print(f"    ✓ TP/SL восстановлены для {_safe_console(symbol)}: SL={format_price(stop)}, TP={format_price(tp)}")
            restored_count += 1
        else:
            print(f"    ✗ Не удалось восстановить TP/SL для {_safe_console(symbol)}: {msg}")
    
    if restored_count > 0:
        print(f"  Восстановлено TP/SL для {restored_count} позиций.")


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


def _compute_signal_for_symbol(symbol: str):
    """Вынесено отдельно, чтобы можно было считать сигналы в нескольких потоках."""
    try:
        # В лайве берём последние 5 дней истории (5m) для ускорения цикла.
        df = fetch_binance_history(symbol, days=5, interval="5m")
        if df is None or len(df) < MIN_BARS:
            return None
        df = add_indicators(df)
        signals = get_signals(df)
        if not signals:
            return None
        last = signals[-1]
        bar_idx = last["bar_index"]
        if bar_idx < len(df) - 2:
            return None
        return {
            "symbol": symbol,
            "direction": last["direction"],
            "entry": last["entry"],
            "stop": last["stop"],
            "take_profit": last["take_profit"],
            "signal_time": last.get("signal_time", ""),
            "bar_index": last["bar_index"],
        }
    except Exception as e:
        print(f"  {symbol}: {e}")
        return None


def run_cycle(state_set):
    """Сбор сигналов: символы с min notional <= эффективный номинал; при ONLY_TOP_COINS — только топ по TP.

    Для ускорения считаем историю/индикаторы по монетам параллельно (несколько потоков).
    """
    notional = EFFECTIVE_NOTIONAL_USD if EFFECTIVE_NOTIONAL_USD is not None else POSITION_NOTIONAL_USD
    symbols = get_binance_symbols(min_volume_usdt=MIN_VOLUME_24H_USDT, max_min_notional_usdt=notional)
    if ONLY_TOP_COINS and symbols:
        top_set = set(TOP_COINS_BY_TP)
        symbols = [s for s in symbols if s["symbol"] in top_set]
    new_signals = []
    if not symbols:
        return state_set, new_signals

    # Небольшой пул потоков, чтобы не долбить Binance слишком агрессивно.
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_symbol = {
            executor.submit(_compute_signal_for_symbol, s["symbol"]): s["symbol"]
            for s in symbols
        }
        for fut in as_completed(future_to_symbol):
            sig = fut.result()
            if not sig:
                continue
            key = f"{sig['symbol']}|{sig['direction']}|{sig.get('signal_time', '')}"
            if key in state_set:
                continue
            state_set.add(key)
            new_signals.append({
                "symbol": sig["symbol"],
                "direction": sig["direction"],
                "entry": sig["entry"],
                "stop": sig["stop"],
                "take_profit": sig["take_profit"],
                "signal_time": sig.get("signal_time", ""),
            })
    return state_set, new_signals


def _safe_console(s):
    if s is None:
        return ""
    return s.encode("ascii", "replace").decode("ascii")


def main():
    print()
    print("=" * 64)
    print("  5m HYBRID LIVE — Binance USDT-M Futures")
    print("  Стратегия: hybrid  |  Вход: MARKET  |  R:R 1:3  SL=0.3*ATR")
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
    # Кэш min notional по символам, чтобы не получать ошибки вида "Order's notional must be no smaller than 5".
    global MIN_NOTIONAL_MAP
    try:
        MIN_NOTIONAL_MAP = get_symbols_min_notional_map() or {}
    except Exception as e:
        print("  Не удалось загрузить MIN_NOTIONAL карту:", e)
        MIN_NOTIONAL_MAP = {}
    state_set = load_state()

    # При старте проверяем и восстанавливаем TP/SL для всех открытых позиций
    if not DRY_RUN:
        print("\n  Проверка открытых позиций и восстановление TP/SL...")
        restore_tp_sl_for_positions()

    while True:
        wait_until_next_5m()
        print(f"\n  Проверка {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Перед каждой проверкой новых сигналов проверяем открытые позиции
        if not DRY_RUN:
            restore_tp_sl_for_positions()
        
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
            # Округляем SL/TP к разрешённому шагу цены (tickSize), чтобы не было ошибок precision.
            filt_prices = filters_cache.get(symbol, {})
            tick = filt_prices.get("tickSize", 0.0001)
            stop = round_price_to_tick(stop, tick)
            tp = round_price_to_tick(tp, tick)
            print(f"  [СИГНАЛ] {_safe_console(symbol)}  {direction}  Entry:{format_price(entry)}  SL:{format_price(stop)}  TP:{format_price(tp)}  | {sig['signal_time']}")
            append_signal_to_log(sig)

            if DRY_RUN:
                print("    (DRY_RUN — ордер не отправлен)")
                continue

            # Проверка: пропускаем монеты, где MIN_NOTIONAL > 5.3
            try:
                sym_min_notional = float(MIN_NOTIONAL_MAP.get(symbol, 999))
            except (TypeError, ValueError):
                sym_min_notional = 999.0
            if sym_min_notional > 5.3:
                print(f"    Пропуск: MIN_NOTIONAL {sym_min_notional:.2f} > 5.3 USDT")
                continue

            # Вход: фиксированный номинал 5.3 USDT
            notional = 5.3
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

            # Пытаемся включить изолированную маржу (неблокирующе, только попытка).
            set_margin_isolated(symbol)

            if not set_leverage(symbol, LEVERAGE):
                print("    Не удалось выставить плечо, пропуск")
                continue

            side = "BUY" if direction.upper() == "LONG" else "SELL"
            ok, order_id_or_msg = place_market_order(symbol, side, quantity)
            if not ok:
                print(f"    Ордер не отправлен: {order_id_or_msg}")
                continue

            print(f"    Ордер отправлен. Vol={quantity}. Выставляю TP/SL (до {TP_SL_MAX_ATTEMPTS} попыток)...")
            time.sleep(3.0)  # Увеличена задержка, чтобы позиция точно успела открыться и синхронизироваться
            is_long = direction.upper() == "LONG"
            
            # Получаем актуальный размер позиции перед выставлением TP/SL
            position_amt, position_side = get_position_amt(symbol)
            if position_amt == 0:
                print(f"    Позиция по {_safe_console(symbol)} не открылась, пропуск TP/SL")
                continue
            
            # Используем реальный размер позиции для TP/SL
            actual_quantity = abs(position_amt)
            tp_sl_ok, closed_anyway, tp_sl_msg = set_tp_sl_with_retries(symbol, stop, tp, is_long, actual_quantity)
            if tp_sl_ok:
                print("    TP и SL выставлены.")
                print(f"    >>> Вход: {format_price(entry)}  |  SL: {format_price(stop)}  |  TP: {format_price(tp)}  <<<")
            elif closed_anyway:
                print(f"    {tp_sl_msg}")
            else:
                print(f"    Ошибка TP/SL: {tp_sl_msg}")


if __name__ == "__main__":
    main()
