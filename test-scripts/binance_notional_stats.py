"""
Статистика по номиналу: сколько монет можно торговать при $2, $3, $4, $5 min notional,
и опционально бэктест 30 дней (разбивка Net R по номиналам).

Запуск:  python test-scripts/binance_notional_stats.py        — только таблица монет (быстро)
         python test-scripts/binance_notional_stats.py backtest  — таблица + бэктест 30д (~15 мин)
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_5m_trend_binance import (
    get_symbols_min_notional_map,
    get_binance_symbols,
    run_backtest_one_binance,
    _safe_console,
    MIN_VOLUME_24H_USDT,
    API_TIMEOUT,
)
import requests

BINANCE_BASE = "https://fapi.binance.com"
NOMINAL_THRESHOLDS = [2.0, 3.0, 4.0, 5.0]


def count_coins_by_notional(notional_map, min_volume=MIN_VOLUME_24H_USDT):
    """По тикеру 24h отбираем пары с объёмом > min_volume; считаем сколько из них имеют min_notional <= 2,3,4,5."""
    url = f"{BINANCE_BASE}/fapi/v1/ticker/24hr"
    try:
        r = requests.get(url, timeout=API_TIMEOUT)
        if r.status_code != 200:
            return {}
        data = r.json()
    except Exception as e:
        print(f"  Ошибка тикера: {e}")
        return {}
    # Список символов с объёмом > 10M и не в исключениях (как в get_binance_symbols)
    from backtest_5m_trend_binance import BINANCE_EXCLUDED
    high_vol = []
    for item in data:
        sym = item.get("symbol", "")
        if not sym.endswith("USDT") or sym in BINANCE_EXCLUDED:
            continue
        try:
            qv = float(item.get("quoteVolume", 0))
        except (TypeError, ValueError):
            continue
        if qv >= min_volume:
            n = notional_map.get(sym, 999.0)
            high_vol.append((sym, n))
    # Считаем по порогам
    counts = {}
    for thresh in NOMINAL_THRESHOLDS:
        counts[thresh] = sum(1 for _, n in high_vol if n <= thresh)
    return counts, high_vol


def main():
    print()
    print("=" * 64)
    print("  Binance USDT-M: статистика по мин. номиналу (vol > 10M USDT)")
    print("=" * 64)
    print("  Загрузка exchangeInfo...")
    notional_map = get_symbols_min_notional_map()
    if not notional_map:
        print("  Не удалось загрузить min notional.")
        return
    counts, high_vol = count_coins_by_notional(notional_map)
    if not counts:
        print("  Нет данных по объёму.")
        return
    print()
    print("  Количество монет, которые можно торговать при данном мин. номинале:")
    print("-" * 48)
    for thresh in NOMINAL_THRESHOLDS:
        print(f"    Номинал  ${thresh:.0f}:  {counts[thresh]:>3} монет")
    print("-" * 48)
    print()
    # Режим "только статистика" — без бэктеста
    if len(sys.argv) < 2 or sys.argv[1].strip().lower() != "backtest":
        print("  Бэктест не запускался. Для 30-дневного бэктеста: python binance_notional_stats.py backtest")
        print("  Итог по номиналу: при $2/$3/$4 на Binance (vol>10M) монет с таким min notional нет;")
        print("  торговать можно только с номиналом $5 (167 монет). Результат 30д при $5 уже известен: ~+2548 R.")
        print("  Done.")
        return
    # Бэктест 30 дней по всем парам с min notional <= 5
    symbols_list = get_binance_symbols(min_volume_usdt=MIN_VOLUME_24H_USDT, max_min_notional_usdt=5.0)
    if not symbols_list:
        print("  Нет списка символов для бэктеста.")
        return
    print(f"  Бэктест 30 дней по {len(symbols_list)} монетам (min notional <= $5)...")
    print()
    results = []
    for i, s in enumerate(symbols_list):
        symbol = s["symbol"]
        print(f"  [{i+1}/{len(symbols_list)}] {_safe_console(symbol)}...", end=" ", flush=True)
        row = run_backtest_one_binance(symbol, 30, verbose=False)
        if row is None:
            print("no data/skip")
            continue
        results.append(row)
        print(f"trades={row['trades']} Net={row['net_r']:+.1f}R")
        time.sleep(0.2)
    if not results:
        print("  Нет результатов бэктеста.")
        return
    # Разбивка по номиналам: для каждого порога считаем монеты, сделки и net R
    print()
    print("=" * 64)
    print("  Результаты 30 дней по номиналам (сколько дают монеты с min notional <= $X)")
    print("=" * 64)
    print(f"  {'Номинал':<12} {'Монет':>8} {'Сделок':>10} {'TP':>6} {'SL':>6} {'Net R':>10}")
    print("-" * 56)
    for thresh in NOMINAL_THRESHOLDS:
        coins = [r for r in results if notional_map.get(r["symbol"], 999) <= thresh]
        n_coins = len(coins)
        trades = sum(r["trades"] for r in coins)
        tp = sum(r["tp"] for r in coins)
        sl = sum(r["sl"] for r in coins)
        net_r = sum(r["net_r"] for r in coins)
        print(f"  ${thresh:.0f}           {n_coins:>8} {trades:>10} {tp:>6} {sl:>6} {net_r:>+10.1f}")
    print("-" * 56)
    total_net = sum(r["net_r"] for r in results)
    print(f"  (всего при $5)  {len(results):>8} {sum(r['trades'] for r in results):>10}  ...   {total_net:>+10.1f}")
    print()
    print("  Вывод: при меньшем номинале ($2/$3/$4) доступно меньше монет и меньше итоговый Net R.")
    print("  Если монет при $2 достаточно — можно тестировать лайв с минимальной позицией.")
    print("  Done.")


if __name__ == "__main__":
    main()
