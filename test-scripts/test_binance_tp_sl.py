"""
Тестовый скрипт для проверки выставления TP/SL на Binance Futures.
Открывает небольшую позицию и сразу выставляет TP/SL, затем закрывает позицию.
"""
import os
import sys
import time
import json
import hmac
import hashlib
import urllib.parse
import requests
from datetime import datetime

try:
    from dotenv import load_dotenv
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _env_path = os.path.join(_script_dir, "..", ".env")
    load_dotenv(_env_path)
    load_dotenv()
except ImportError:
    pass

BINANCE_BASE = "https://fapi.binance.com"
API_KEY = os.environ.get("BINANCE_API_KEY")
SECRET = os.environ.get("BINANCE_SECRET")

if not API_KEY or not SECRET:
    print("Ошибка: задайте BINANCE_API_KEY и BINANCE_SECRET в .env")
    sys.exit(1)


def _binance_sign(params: dict) -> str:
    query = urllib.parse.urlencode(params)
    return hmac.new(SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()


def _signed_request(method: str, endpoint: str, params: dict = None):
    if params is None:
        params = {}
    params["timestamp"] = int(time.time() * 1000)
    params["signature"] = _binance_sign(params)
    url = f"{BINANCE_BASE}{endpoint}"
    headers = {"X-MBX-APIKEY": API_KEY}
    
    if method == "GET":
        resp = requests.get(url, params=params, headers=headers, timeout=10)
    else:
        resp = requests.post(url, data=params, headers=headers, timeout=10)
    
    if resp.status_code == 200:
        return True, resp.json()
    return False, resp.text


def get_position_amt(symbol: str):
    """Текущая позиция по символу."""
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


def get_symbol_filters(symbol: str):
    """Получить фильтры для символа (tickSize, stepSize)."""
    ok, data = _signed_request("GET", "/fapi/v1/exchangeInfo", {})
    if not ok:
        return {}
    for s in data.get("symbols", []):
        if s["symbol"] == symbol:
            filters = {}
            for f in s.get("filters", []):
                if f["filterType"] == "PRICE_FILTER":
                    filters["tickSize"] = float(f["tickSize"])
                elif f["filterType"] == "LOT_SIZE":
                    filters["stepSize"] = float(f["stepSize"])
                    filters["minQty"] = float(f["minQty"])
            return filters
    return {}


def round_price_to_tick(price: float, tick: float) -> float:
    """Округление цены к разрешённому шагу tickSize."""
    if tick <= 0:
        return price
    steps = round(price / tick)
    return round(steps * tick, 8)


def round_quantity(qty: float, step: float, min_qty: float) -> float:
    """Округление количества к stepSize."""
    if step <= 0:
        return qty
    steps = round(qty / step)
    result = steps * step
    return max(result, min_qty) if min_qty > 0 else result


def place_market_order(symbol: str, side: str, quantity: float):
    """Рыночный ордер."""
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
    """TP как лимитный ордер reduceOnly."""
    position_amt, position_side = get_position_amt(symbol)
    if position_amt == 0:
        return False, "Позиция не открыта"
    
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


def place_stop_order_algo(symbol: str, side: str, stop_price: float):
    """SL через обычный order endpoint (не Algo Order API)."""
    params = {
        "symbol": symbol,
        "side": side.upper(),
        "type": "STOP_MARKET",
        "stopPrice": stop_price,
        "closePosition": "true",  # ВАЖНО: не используем reduceOnly вместе с closePosition!
        "workingType": "MARK_PRICE",
    }
    ok, data = _signed_request("POST", "/fapi/v1/order", params)
    if ok:
        return ok, data
    # Пробуем CONTRACT_PRICE
    params["workingType"] = "CONTRACT_PRICE"
    ok, data = _signed_request("POST", "/fapi/v1/order", params)
    return ok, data


def close_position_market(symbol: str, is_long: bool, quantity: float):
    """Закрыть позицию рыночным ордером."""
    side = "SELL" if is_long else "BUY"
    ok, msg = _signed_request("POST", "/fapi/v1/order", {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": abs(quantity),
        "reduceOnly": "true",
    })
    return ok, msg


def get_open_orders(symbol: str):
    """Получить открытые ордера."""
    ok, data = _signed_request("GET", "/fapi/v1/openOrders", {"symbol": symbol})
    if not ok:
        return []
    return data if isinstance(data, list) else []


def main():
    print("=" * 64)
    print("  ТЕСТ: Открытие позиции и выставление TP/SL на Binance Futures")
    print("=" * 64)
    print()
    
    # Используем SIGNUSDT для теста (низкий минимальный номинал)
    symbol = "SIGNUSDT"
    print(f"Символ для теста: {symbol}")
    
    # Получаем текущую цену
    ok, ticker = _signed_request("GET", "/fapi/v1/ticker/price", {"symbol": symbol})
    if not ok:
        print(f"Ошибка получения цены: {ticker}")
        return
    current_price = float(ticker["price"])
    print(f"Текущая цена: {current_price}")
    
    # Получаем фильтры
    filters = get_symbol_filters(symbol)
    tick = filters.get("tickSize", 0.0001)
    step = filters.get("stepSize", 0.001)
    min_qty = filters.get("minQty", 0.001)
    
    # Рассчитываем небольшую позицию (примерно $5.3, как в основном скрипте)
    notional = 5.3
    raw_qty = notional / current_price
    quantity = round_quantity(raw_qty, step, min_qty)
    print(f"Количество: {quantity} SIGN")
    print()
    
    # Шаг 1: Открываем LONG позицию
    print("Шаг 1: Открываем LONG позицию...")
    ok, order_id = place_market_order(symbol, "BUY", quantity)
    if not ok:
        print(f"Ошибка открытия позиции: {order_id}")
        return
    print(f"[OK] Ордер отправлен, ID: {order_id}")
    
    # Ждем, пока позиция откроется
    print("Ожидание открытия позиции...")
    time.sleep(3)
    
    position_amt, position_side = get_position_amt(symbol)
    if position_amt == 0:
        print("Ошибка: позиция не открылась")
        return
    print(f"[OK] Позиция открыта: {abs(position_amt)} SIGN ({position_side})")
    print()
    
    # Шаг 2: Рассчитываем TP/SL
    entry_price = current_price
    # SL на 0.5% ниже входа
    stop_price = entry_price * 0.995
    # TP на 1.5% выше входа
    take_profit_price = entry_price * 1.015
    
    stop_price = round_price_to_tick(stop_price, tick)
    take_profit_price = round_price_to_tick(take_profit_price, tick)
    
    print(f"Entry: {entry_price:.5f}")
    print(f"SL: {stop_price:.5f}")
    print(f"TP: {take_profit_price:.5f}")
    print()
    
    # Шаг 3: Выставляем TP
    print("Шаг 2: Выставляем Take Profit...")
    ok_tp, tp_data = place_take_profit_limit(symbol, "SELL", take_profit_price, abs(position_amt))
    if ok_tp:
        print(f"[OK] TP выставлен: {tp_data}")
    else:
        print(f"[ERROR] Ошибка выставления TP: {tp_data}")
    print()
    
    # Шаг 4: Выставляем SL
    print("Шаг 3: Выставляем Stop Loss через Algo Order API...")
    ok_sl, sl_data = place_stop_order_algo(symbol, "SELL", stop_price)
    if ok_sl:
        print(f"[OK] SL выставлен: {sl_data}")
    else:
        error_msg = str(sl_data).encode('ascii', 'replace').decode('ascii')
        print(f"[ERROR] Ошибка выставления SL: {error_msg}")
    print()
    
    # Проверяем открытые ордера
    print("Проверка открытых ордеров...")
    orders = get_open_orders(symbol)
    print(f"Открытых ордеров: {len(orders)}")
    for order in orders:
        print(f"  - {order.get('type')} {order.get('side')} @ {order.get('price', order.get('stopPrice', 'N/A'))}")
    print()
    
    # Шаг 5: Закрываем позицию
    print("Шаг 4: Закрываем позицию...")
    ok_close, close_data = close_position_market(symbol, True, abs(position_amt))
    if ok_close:
        print(f"[OK] Позиция закрыта: {close_data}")
    else:
        print(f"[ERROR] Ошибка закрытия позиции: {close_data}")
    print()
    
    # Отменяем оставшиеся ордера
    print("Отмена оставшихся TP/SL ордеров...")
    for order in orders:
        order_id = order.get("orderId")
        if order_id:
            ok, result = _signed_request("DELETE", "/fapi/v1/order", {
                "symbol": symbol,
                "orderId": order_id
            })
            if ok:
                print(f"[OK] Ордер {order_id} отменен")
            else:
                print(f"[ERROR] Не удалось отменить ордер {order_id}: {result}")
    
    print()
    print("=" * 64)
    print("  ТЕСТ ЗАВЕРШЕН")
    print("=" * 64)


if __name__ == "__main__":
    main()
