"""
One-off: fetch MEXC contract specs + ticker, compute min order value (USDT) per symbol.
List symbols where min order > TARGET_USD (default 5) so user knows which to skip for $5 position.
"""
import os
import requests

TARGET_USD = 5.0
MIN_VOLUME_24H = 10_000_000

def main():
    # Contract specs
    r = requests.get("https://contract.mexc.com/api/v1/contract/detail", timeout=15)
    if not r.ok:
        print("Failed to get contract detail")
        return
    detail = r.json()
    if not detail.get("success") or "data" not in detail:
        print("No data in contract detail")
        return

    # Ticker (prices)
    r2 = requests.get("https://contract.mexc.com/api/v1/contract/ticker", timeout=15)
    if not r2.ok:
        print("Failed to get ticker")
        return
    ticker = r2.json()
    if not ticker.get("success") or "data" not in ticker:
        print("No data in ticker")
        return

    prices = {d["symbol"]: float(d.get("lastPrice", 0)) for d in ticker["data"]}
    volumes = {d["symbol"]: float(d.get("amount24", 0)) for d in ticker["data"]}

    # Build list: symbol -> min_notional_usd
    contracts = detail["data"]
    results = []
    for c in contracts:
        sym = c.get("symbol", "")
        if "_USDT" not in sym and "_USDC" not in sym and "_USD" not in sym:
            continue
        vol_24 = volumes.get(sym, 0)
        if vol_24 < MIN_VOLUME_24H:
            continue
        min_vol = int(c.get("minVol", 1))
        contract_size = float(c.get("contractSize", 0))
        price = prices.get(sym, 0)
        if price <= 0:
            continue
        # USDT/USDC: notional in USD. USD-margined (e.g. BTC_USD) notional in USD too (contract in USD).
        if "_USDT" in sym or "_USDC" in sym:
            min_notional = min_vol * contract_size * price
        else:
            # e.g. BTC_USD: contractSize in base (BTC), settle in BTC - need to convert to USD
            min_notional = min_vol * contract_size * price
        results.append({
            "symbol": sym,
            "min_vol": min_vol,
            "contract_size": contract_size,
            "price": price,
            "min_notional_usd": round(min_notional, 2),
        })

    above = [x for x in results if x["min_notional_usd"] > TARGET_USD]
    above.sort(key=lambda x: -x["min_notional_usd"])

    print(f"Symbols with volume > {MIN_VOLUME_24H/1e6:.0f}M. Target position: ${TARGET_USD} (e.g. $1 x 5x).")
    print(f"Min order = minVol * contractSize * price (USDT).")
    print()
    out_path = os.path.join(os.path.dirname(__file__), "mexc_min_order_above_5.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Coins with min order > $5 (volume > 10M). Minimal bet size in USD.\n\n")
        for x in above:
            f.write(f"{x['symbol']}\t{x['min_notional_usd']:.2f}\n")
    print("--- COINS WHERE MIN ORDER > $5 (need to increase bet or exclude) ---")
    for x in above:
        print(f"  {x['symbol']:<22}  min order ${x['min_notional_usd']:>8.2f}  (minVol={x['min_vol']}  size={x['contract_size']}  price={x['price']})")
    print()
    print("--- COINS WHERE MIN ORDER <= $5 (OK for $5 position) ---")
    below = [x for x in results if x["min_notional_usd"] <= TARGET_USD]
    below.sort(key=lambda x: x["symbol"])
    for x in below[:40]:
        print(f"  {x['symbol']:<22}  min order ${x['min_notional_usd']:>6.2f}")
    if len(below) > 40:
        print(f"  ... and {len(below) - 40} more")
    print()
    print("--- DEPOSIT (rough) ---")
    print(f"  Risk $1/trade, 5x leverage => margin ~$1 per trade. 30d had ~3300 trades across 64 coins.")
    print(f"  Suggested min deposit: $100-200 (buffer for drawdown, multiple open positions).")
    print(f"  If trading only coins with min<=$5: fewer coins, similar logic.")

if __name__ == "__main__":
    main()
