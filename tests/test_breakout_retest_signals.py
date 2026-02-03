import time
import os
import json
import winsound
import numpy as np
import pandas as pd
import requests
import talib
from datetime import datetime


STATE_FILE = "breakout_retest_state.json"
MIN_PROFIT_PCT = 0.05  # минимальная прибыль 5% от входа (3R >= 5%)


def get_high_volume_symbols(min_volume=10_000_000):
    """Фьючерсные монеты MEXC с высоким объемом."""
    url = "https://contract.mexc.com/api/v1/contract/ticker"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        symbols = []
        if "data" in data:
            for item in data["data"]:
                if item.get("amount24", 0) > min_volume:
                    symbols.append({
                        "symbol": item["symbol"],
                        "volume_24h": item["amount24"],
                    })
        return sorted(symbols, key=lambda x: x["volume_24h"], reverse=True)
    except Exception as e:
        print(f"Ошибка получения списка монет: {e}")
        return []


def get_candles(symbol, interval="Min5", limit=800):
    url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
    params = {"interval": interval, "limit": limit}
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"   ⚠️ Ошибка получения данных для {symbol}: {e}")
    return None


def create_dataframe(data):
    if not data or not data.get("success") or not data.get("data"):
        return None
    raw = data["data"]
    try:
        if isinstance(raw, dict):
            required = ["time", "open", "high", "low", "close", "vol"]
            if not all(k in raw for k in required):
                return None
            length = min(len(raw["time"]), len(raw["open"]), len(raw["high"]),
                         len(raw["low"]), len(raw["close"]), len(raw["vol"]))
            if length == 0:
                return None
            df = pd.DataFrame({
                "timestamp": raw["time"][:length],
                "open": raw["open"][:length],
                "high": raw["high"][:length],
                "low": raw["low"][:length],
                "close": raw["close"][:length],
                "volume": raw["vol"][:length],
            })
        elif isinstance(raw, list):
            cleaned = []
            for row in raw:
                if isinstance(row, (list, tuple)) and len(row) >= 6:
                    cleaned.append(row[:6])
            if not cleaned:
                return None
            df = pd.DataFrame(cleaned, columns=["timestamp", "open", "high", "low", "close", "volume"])
        else:
            return None

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()
        if len(df) < 200:
            return None

        ts = df["timestamp"].iloc[0]
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms" if ts > 1e12 else "s")
        df = df.sort_values("datetime").reset_index(drop=True)
        return df
    except Exception:
        return None


def add_indicators(df):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values
    df["atr"] = talib.ATR(high, low, close, timeperiod=14)
    df["vol_sma"] = talib.SMA(volume, timeperiod=20)
    df["vol_ratio"] = volume / df["vol_sma"]
    # EMA для тренда и динамических уровней
    df["ema_5"] = talib.EMA(close, timeperiod=5)
    df["ema_10"] = talib.EMA(close, timeperiod=10)
    df["ema_30"] = talib.EMA(close, timeperiod=30)
    df["ema_60"] = talib.EMA(close, timeperiod=60)
    # RSI для перепроданности/перекупленности
    df["rsi"] = talib.RSI(close, timeperiod=14)
    # Размеры свечей и теней
    df["body_size"] = abs(df["close"] - df["open"])
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["candle_range"] = df["high"] - df["low"]
    return df


def format_price(value):
    if value is None or value == "":
        return ""
    if value < 0.0001:
        return f"{value:.8f}"
    if value < 0.001:
        return f"{value:.7f}"
    if value < 0.01:
        return f"{value:.6f}"
    if value < 1:
        return f"{value:.5f}"
    if value < 100:
        return f"{value:.4f}"
    return f"{value:.2f}"


def find_strong_levels(df, lookback=200, vol_threshold=1.8, min_touches=3):
    """
    Улучшенный поиск сильных уровней:
    1. VPVR-подобная логика - кластеризация объема по ценам
    2. Требуем больше касаний и выше порог объема
    3. Учитываем EMA как динамические уровни
    """
    window = df.tail(lookback).copy()
    levels = []
    
    # 1. VPVR-подобная логика: находим ценовые зоны с высоким объемом
    price_bins = {}
    atr = df["atr"].iloc[-2] if not pd.isna(df["atr"].iloc[-2]) else df["close"].iloc[-2] * 0.003
    bin_size = max(atr * 0.5, df["close"].iloc[-2] * 0.003)
    
    for i in range(len(window)):
        row = window.iloc[i]
        if row["vol_ratio"] < 1.2:  # Минимальный фильтр для VPVR
            continue
        
        # Кластеризуем объем по ценовым зонам
        for price_level in [row["high"], row["low"], row["close"]]:
            bin_key = round(price_level / bin_size) * bin_size
            if bin_key not in price_bins:
                price_bins[bin_key] = {"volume": 0, "touches": 0, "vol_ratios": []}
            price_bins[bin_key]["volume"] += row["volume"]
            price_bins[bin_key]["touches"] += 1
            price_bins[bin_key]["vol_ratios"].append(row["vol_ratio"])
    
    # Находим сильные ценовые зоны (VPVR nodes)
    avg_volume = window["volume"].mean()
    strong_zones = []
    for price, data in price_bins.items():
        avg_vol_ratio = np.mean(data["vol_ratios"]) if data["vol_ratios"] else 0
        if data["volume"] > avg_volume * 1.5 and avg_vol_ratio > 1.3 and data["touches"] >= 3:
            strong_zones.append((price, avg_vol_ratio, data["touches"], data["volume"]))
    
    # 2. Традиционный метод: экстремумы с высоким объемом
    for i in range(2, len(window) - 2):
        row = window.iloc[i]
        if row["vol_ratio"] < vol_threshold:
            continue
        
        # Swing high
        if row["high"] > window.iloc[i - 1]["high"] and row["high"] > window.iloc[i + 1]["high"]:
            levels.append(("RES", row["high"], row["vol_ratio"], 1))
        
        # Swing low
        if row["low"] < window.iloc[i - 1]["low"] and row["low"] < window.iloc[i + 1]["low"]:
            levels.append(("SUP", row["low"], row["vol_ratio"], 1))
    
    # 3. Добавляем EMA как динамические уровни (если они близко к текущей цене)
    current_price = df["close"].iloc[-1]
    tolerance_ema = max(atr * 0.6, current_price * 0.003)
    
    for ema_name, ema_value in [("EMA30", df["ema_30"].iloc[-1]), ("EMA60", df["ema_60"].iloc[-1])]:
        if not pd.isna(ema_value) and abs(current_price - ema_value) < tolerance_ema:
            # Определяем тип уровня по тренду
            if df["ema_5"].iloc[-1] > df["ema_30"].iloc[-1]:
                levels.append(("SUP", ema_value, 1.5, 0))  # Поддержка в восходящем тренде
            else:
                levels.append(("RES", ema_value, 1.5, 0))  # Сопротивление в нисходящем тренде
    
    # 4. Объединяем VPVR зоны с традиционными уровнями
    tolerance = max(atr * 0.4, df["close"].iloc[-2] * 0.002)
    confirmed = []
    
    # Обрабатываем VPVR зоны
    for price, vol_ratio, touches, vol in strong_zones:
        # Определяем тип уровня
        if price < current_price:
            lvl_type = "SUP"
            check_col = "low"
        else:
            lvl_type = "RES"
            check_col = "high"
        
        # Подсчитываем точные касания
        exact_touches = (abs(window[check_col] - price) <= tolerance).sum()
        if exact_touches >= min_touches:
            confirmed.append((lvl_type, price, vol_ratio, exact_touches, vol))
    
    # Обрабатываем традиционные уровни
    for lvl_type, level, lvl_vol, _ in levels:
        if lvl_type == "RES":
            touches = (abs(window["high"] - level) <= tolerance).sum()
        else:
            touches = (abs(window["low"] - level) <= tolerance).sum()
        
        if touches >= min_touches:
            # Проверяем, не дублируем ли мы VPVR уровень
            is_duplicate = False
            for _, vpvr_price, _, _, _ in confirmed:
                if abs(level - vpvr_price) <= tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                confirmed.append((lvl_type, level, lvl_vol, touches, 0))
    
    # Сортируем по силе: объем, количество касаний, vol_ratio
    confirmed.sort(key=lambda x: (x[4] if x[4] > 0 else x[2] * 1000, x[3], x[2]), reverse=True)
    return confirmed[:8]  # Возвращаем больше уровней для лучшего выбора


def evaluate_breakout_retest(df):
    """
    Улучшенная логика Breakout + Retest:
    1. Более строгие требования к пробою (сильное тело свечи)
    2. Ретест должен иметь длинную тень (wick) - показывает отскок
    3. Объем при ретесте должен быть НИЖЕ среднего (отсутствие давления)
    4. Объем при подтверждении должен быть ВЫШЕ среднего
    5. Проверка на тренд (EMA alignment)
    6. Проверка на RSI (перепроданность/перекупленность)
    7. Минимальный R/R = 1:3
    """
    if len(df) < 200:
        return None

    breakout = df.iloc[-3]
    retest = df.iloc[-2]
    confirm = df.iloc[-1]

    if np.isnan(retest["atr"]) or retest["atr"] <= 0:
        return None

    levels = find_strong_levels(df)
    if not levels:
        return None

    atr = retest["atr"]
    tolerance = max(atr * 0.4, retest["close"] * 0.002)
    
    # Более строгие требования к объему
    breakout_vol_ok = breakout["vol_ratio"] > 1.5  # Выше порог для пробоя
    retest_vol_ok = retest["vol_ratio"] < 1.0  # Ретест на низком объеме (отсутствие давления)
    confirm_vol_ok = confirm["vol_ratio"] > 1.2  # Подтверждение на повышенном объеме
    
    # Проверка тренда
    uptrend = df["ema_5"].iloc[-1] > df["ema_10"].iloc[-1] > df["ema_30"].iloc[-1]
    downtrend = df["ema_5"].iloc[-1] < df["ema_10"].iloc[-1] < df["ema_30"].iloc[-1]
    
    # RSI фильтры
    rsi = df["rsi"].iloc[-1]
    oversold = rsi < 35  # Перепроданность (хорошо для LONG)
    overbought = rsi > 65  # Перекупленность (хорошо для SHORT)

    best_signal = None
    best_score = 0

    for level_data in levels:
        if len(level_data) == 5:
            lvl_type, level, lvl_vol, touches, vpvr_vol = level_data
        else:
            lvl_type, level, lvl_vol, touches = level_data
            vpvr_vol = 0

        score = 0
        reasons_parts = []

        if lvl_type == "RES":
            # LONG breakout + retest
            # 1. Пробой должен быть сильным (большое тело, закрытие выше уровня)
            breakout_body_ok = breakout["close"] > level + atr * 0.2  # Пробой с запасом
            breakout_green = breakout["close"] > breakout["open"]  # Зеленая свеча
            breakout_body_size = breakout["body_size"] / breakout["candle_range"] > 0.6  # Большое тело
            
            if not (breakout_body_ok and breakout_green and breakout_body_size):
                continue
            
            score += 20
            reasons_parts.append("сильный пробой вверх")

            # 2. Ретест должен иметь длинную нижнюю тень (отскок от уровня)
            retest_touches_level = abs(retest["low"] - level) <= tolerance
            retest_holds_above = retest["close"] > level
            retest_lower_wick_ok = retest["lower_wick"] > retest["body_size"] * 0.8  # Длинная нижняя тень
            
            if not (retest_touches_level and retest_holds_above):
                continue
            
            if retest_lower_wick_ok:
                score += 25
                reasons_parts.append("длинная нижняя тень при ретесте")
            else:
                score += 10

            # 3. Подтверждение
            confirm_ok = confirm["close"] > retest["close"] and confirm["close"] > level
            confirm_green = confirm["close"] > confirm["open"]
            
            if not confirm_ok:
                continue
            
            if confirm_green:
                score += 15
                reasons_parts.append("зеленая свеча подтверждения")
            else:
                score += 5

            # 4. Объем
            if breakout_vol_ok:
                score += 15
                reasons_parts.append(f"объем пробоя {breakout['vol_ratio']:.2f}x")
            else:
                continue
            
            if retest_vol_ok:
                score += 10
                reasons_parts.append("низкий объем ретеста")
            else:
                score -= 5  # Штраф за высокий объем при ретесте
            
            if confirm_vol_ok:
                score += 10
                reasons_parts.append(f"объем подтверждения {confirm['vol_ratio']:.2f}x")

            # 5. Тренд
            if uptrend:
                score += 15
                reasons_parts.append("восходящий тренд (EMA)")
            elif downtrend:
                score -= 10  # Штраф за вход против тренда

            # 6. RSI
            if oversold:
                score += 10
                reasons_parts.append(f"RSI перепродан ({rsi:.1f})")
            elif rsi > 50:
                score += 5

            # 7. Сила уровня
            if vpvr_vol > 0:
                score += 20
                reasons_parts.append(f"VPVR уровень (касаний {touches})")
            else:
                score += 10
                reasons_parts.append(f"уровень (объем {lvl_vol:.2f}x, касаний {touches})")

            # Расчет уровней входа
            entry = confirm["close"]
            stop = level - atr * 0.8  # Стоп ниже уровня с запасом
            take = entry + (entry - stop) * 3.5  # R/R >= 1:3.5
            
            rr = abs((take - entry) / (entry - stop))
            if rr < 3.0:
                continue  # Минимальный R/R = 1:3
            profit_pct = (take - entry) / entry
            if profit_pct < MIN_PROFIT_PCT:
                continue  # Прибыль должна быть не менее 10%

            if score > best_score:
                best_score = score
                best_signal = {
                    "enter_now": "Да",
                    "direction": "LONG",
                    "entry": entry,
                    "stop": stop,
                    "take_profit": take,
                    "rr": round(rr, 2),
                    "reasons": f"Breakout+Retest LONG от {format_price(level)} | " + " | ".join(reasons_parts) + f" | Score: {score}",
                    "signal_time": retest["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
                    "score": score,
                }

        elif lvl_type == "SUP":
            # SHORT breakout + retest
            # 1. Пробой должен быть сильным
            breakout_body_ok = breakout["close"] < level - atr * 0.2  # Пробой с запасом
            breakout_red = breakout["close"] < breakout["open"]  # Красная свеча
            breakout_body_size = breakout["body_size"] / breakout["candle_range"] > 0.6  # Большое тело
            
            if not (breakout_body_ok and breakout_red and breakout_body_size):
                continue
            
            score += 20
            reasons_parts.append("сильный пробой вниз")

            # 2. Ретест должен иметь длинную верхнюю тень (отскок от уровня)
            retest_touches_level = abs(retest["high"] - level) <= tolerance
            retest_holds_below = retest["close"] < level
            retest_upper_wick_ok = retest["upper_wick"] > retest["body_size"] * 0.8  # Длинная верхняя тень
            
            if not (retest_touches_level and retest_holds_below):
                continue
            
            if retest_upper_wick_ok:
                score += 25
                reasons_parts.append("длинная верхняя тень при ретесте")
            else:
                score += 10

            # 3. Подтверждение
            confirm_ok = confirm["close"] < retest["close"] and confirm["close"] < level
            confirm_red = confirm["close"] < confirm["open"]
            
            if not confirm_ok:
                continue
            
            if confirm_red:
                score += 15
                reasons_parts.append("красная свеча подтверждения")
            else:
                score += 5

            # 4. Объем
            if breakout_vol_ok:
                score += 15
                reasons_parts.append(f"объем пробоя {breakout['vol_ratio']:.2f}x")
            else:
                continue
            
            if retest_vol_ok:
                score += 10
                reasons_parts.append("низкий объем ретеста")
            else:
                score -= 5  # Штраф за высокий объем при ретесте
            
            if confirm_vol_ok:
                score += 10
                reasons_parts.append(f"объем подтверждения {confirm['vol_ratio']:.2f}x")

            # 5. Тренд
            if downtrend:
                score += 15
                reasons_parts.append("нисходящий тренд (EMA)")
            elif uptrend:
                score -= 10  # Штраф за вход против тренда

            # 6. RSI
            if overbought:
                score += 10
                reasons_parts.append(f"RSI перекуплен ({rsi:.1f})")
            elif rsi < 50:
                score += 5

            # 7. Сила уровня
            if vpvr_vol > 0:
                score += 20
                reasons_parts.append(f"VPVR уровень (касаний {touches})")
            else:
                score += 10
                reasons_parts.append(f"уровень (объем {lvl_vol:.2f}x, касаний {touches})")

            # Расчет уровней входа
            entry = confirm["close"]
            stop = level + atr * 0.8  # Стоп выше уровня с запасом
            take = entry - (stop - entry) * 3.5  # R/R >= 1:3.5
            
            rr = abs((entry - take) / (stop - entry))
            if rr < 3.0:
                continue  # Минимальный R/R = 1:3
            profit_pct = (entry - take) / entry
            if profit_pct < MIN_PROFIT_PCT:
                continue  # Прибыль должна быть не менее 10%

            if score > best_score:
                best_score = score
                best_signal = {
                    "enter_now": "Да",
                    "direction": "SHORT",
                    "entry": entry,
                    "stop": stop,
                    "take_profit": take,
                    "rr": round(rr, 2),
                    "reasons": f"Breakout+Retest SHORT от {format_price(level)} | " + " | ".join(reasons_parts) + f" | Score: {score}",
                    "signal_time": retest["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
                    "score": score,
                }

    # Возвращаем только сигналы с достаточно высоким score
    if best_signal and best_score >= 60:  # Минимальный порог уверенности
        return best_signal
    
    return None


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def save_state(state_set):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(list(state_set)), f, ensure_ascii=False, indent=2)


def wait_until_next_5min():
    now = datetime.now()
    next_min = ((now.minute // 5) + 1) * 5
    if next_min == 60:
        next_time = now.replace(minute=0, second=1, microsecond=0) + pd.Timedelta(hours=1)
    else:
        next_time = now.replace(minute=next_min, second=1, microsecond=0)
    sleep_seconds = (next_time - now).total_seconds()
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)


def run_once(state_set):
    print("🚀 BREAKOUT + RETEST (MEXC, 5m) - УЛУЧШЕННАЯ ВЕРСИЯ")
    print("✅ VPVR-уровни | Строгие ретесты | R/R ≥ 1:3 | Прибыль ≥ 5%")
    print(f"⏰ Проверка: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    symbols = get_high_volume_symbols(min_volume=10_000_000)
    if not symbols:
        print("❌ Нет монет по объёму")
        return state_set

    new_entries = []
    checked_count = 0

    for s in symbols:
        symbol = s["symbol"]
        checked_count += 1
        if checked_count % 10 == 0:
            print(f"   Проверено монет: {checked_count}/{len(symbols)}")
        
        data = get_candles(symbol, "Min5", 800)
        df = create_dataframe(data)
        if df is None:
            continue
        df = add_indicators(df)
        signal = evaluate_breakout_retest(df)
        if not signal:
            continue

        key = f"{symbol}|{signal['direction']}|{signal['signal_time']}"
        if key not in state_set:
            new_entries.append({
                "symbol": symbol,
                "signal": signal
            })
            state_set.add(key)

        time.sleep(0.25)

    if new_entries:
        print(f"\n🔔 НАЙДЕНО НОВЫХ СИГНАЛОВ: {len(new_entries)}")
        print("=" * 80)
        for entry in new_entries:
            s = entry["signal"]
            print(f"\n📊 {entry['symbol']} {s['direction']}")
            print(f"   ⏰ Время сигнала: {s['signal_time']}")
            print(f"   💰 Вход: {format_price(s['entry'])}")
            print(f"   🛑 Стоп-лосс: {format_price(s['stop'])}")
            print(f"   🎯 Тейк-профит: {format_price(s['take_profit'])} (R/R = {s['rr']})")
            print(f"   📈 Score: {s.get('score', 0)}")
            print(f"   📝 {s['reasons']}")
            print("-" * 80)
        
        # Громкий звуковой сигнал
        print("\n🔊 ЗВУКОВОЙ СИГНАЛ!")
        try:
            winsound.MessageBeep(winsound.MB_ICONHAND)
        except Exception:
            pass
        for _ in range(5):
            winsound.Beep(1200, 600)
            winsound.Beep(800, 400)
            time.sleep(0.1)
    else:
        print(f"\n✅ Проверено {checked_count} монет. Новых сигналов не найдено.")

    return state_set


def run_loop():
    state_set = load_state()
    print("⏱️ Запуск: каждые 5 минут на 01-й секунде")
    while True:
        wait_until_next_5min()
        print(f"\n⏰ Запуск проверки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            state_set = run_once(state_set)
            save_state(state_set)
        except Exception as e:
            print(f"⚠️ Ошибка цикла: {e}")


if __name__ == "__main__":
    run_loop()
