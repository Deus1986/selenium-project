"""
Алгоритм 1H только по уровням поддержки и сопротивления.
Единые параметры под 0 SL на нескольких монетах (HYPE, ZEC, ETH). Допустимо SL <= 10% от TP (10 TP → макс. 1 SL).

Правила:
1. Уровни: сопротивление = max(high) за 24 бара до сигнальной; поддержка = min(low) за 24 бара (без сигнальной свечи).
2. LONG: отбой от поддержки, бычья подтверждающая свеча; поддержка не ниже EMA30*0.99 (не лонг в даунтренде); нет падения >20% за 10 бар.
3. SHORT: ретест сопротивления, медвежья подтверждающая (тело >= 0.35*ATR), закрытие подтверждения ниже EMA30; ADX >= 22; RSI <= 72.
4. Коудон 6 бар по направлению; после SHORT не лонг 26 бар. R:R LONG 1.8, SHORT 1.7.

Тренд: LONG только при EMA10 > EMA30 (бычий), SHORT только при EMA10 < EMA30 (медвежий). Без стоп-лоссов.
Запуск: python tests/backtest_sr_1h.py --symbols LINK_USDT,HYPE_USDT,ZEC_USDT,ETH_USDT --from-date 2026-01-25 --limit 400
"""
import numpy as np
import pandas as pd

# Параметры уровней (настраиваются под 5+ TP / 0 SL)
LOOKBACK = 24          # баров для свинг high/low (уровень = экстремум БЕЗ сигнальной свечи, чтобы не входить на "создании" уровня)
LOOKBACK_LOCAL = 12    # локальный уровень: «подошла к уровню и отскочила» — больше касаний
ZONE_ATR = 0.4         # зона уровня = уровень ± ATR * это
CONFIRM_ATR = 0.25     # подтверждение: закрытие за уровнем минимум на ATR*это
MIN_RR = 1.7           # минимальное R:R для SHORT (реалистичный тейк по уровню)
MIN_RR_LONG = 1.8      # для LONG чуть ближе тейк (часто цена «почти доходит» до 2R)
STOP_ATR_BUFFER = 0.35  # стоп за уровнем: уровень ± ATR * это
COOLDOWN_BARS = 6     # после входа не входить в том же направлении N баров (не дублировать вход в одной волне)
MIN_BODY_ATR = 0.22    # подтверждающая свеча: тело минимум ATR*это (отсечь дожи)
MIN_BODY_ATR_LONG = 0.30  # для LONG строже (тело >= 0.30*ATR) — 0 SL в т.ч. на LINK


def _swing_high(df, i, lookback):
    if i < lookback + 2:
        return np.nan
    return df["high"].iloc[i - lookback : i - 1].max()


def _swing_low(df, i, lookback):
    if i < lookback + 2:
        return np.nan
    return df["low"].iloc[i - lookback : i - 1].min()


# Пробой и ретест: окно поиска пробоя (баров назад от сигнала)
BREAKOUT_LOOKBACK = 12
VOL_BREAKOUT_MIN = 1.0   # мин. vol_ratio на баре пробоя
VOL_CONFIRM_MIN = 1.0    # мин. vol_ratio на подтверждающей свече (уже есть для long через VOL_RATIO_LONG_MIN)

# Параметры для оптимизатора. Жёстче = меньше сделок, меньше стоп-лоссов (логика та же).
PARAMS_DEFAULTS = {
    "ADX_SHORT_MIN": 30,           # баланс: меньше SL, достаточно TP (вариант ~96 TP / 99 SL)
    "SUPPORT_EMA30_RATIO": 0.995,
    "MIN_BODY_ATR_LONG": 0.32,
    "VOL_RATIO_LONG_MIN": 1.05,    # чуть выше — меньше слабых лонгов
    "RSI_SHORT_MAX": 66,
    "SHORT_BODY_ATR_MIN": 0.38,
    "VOL_BREAKOUT_MIN": 1.2,       # пробой с объёмом выше среднего
    "VOL_RATIO_SHORT_MIN": 0.9,    # шорт при достаточном объёме
    "ADX_LONG_MIN": 18,
}


def evaluate_sr_1h(df, params=None):
    """
    Один сигнал на бар: либо LONG от поддержки (отбой/ретест), либо SHORT от сопротивления (отбой/пробой).
    params — опциональный dict для подбора (ADX_SHORT_MIN, SUPPORT_EMA30_RATIO, MIN_BODY_ATR_LONG, VOL_RATIO_LONG_MIN, RSI_SHORT_MAX, SHORT_BODY_ATR_MIN).
    """
    _p = {**PARAMS_DEFAULTS, **(params or {})}
    if len(df) < LOOKBACK + 50:
        return None
    # Сигнальная свеча = предпоследняя, подтверждение = последняя (входим по закрытию последней)
    idx = len(df) - 1
    signal = df.iloc[-2]
    confirm = df.iloc[-1]
    current = df.iloc[-1]
    atr = signal["atr"] if signal["atr"] > 0 else signal["close"] * 0.01
    # Уровни: свинг high/low без сигнальной свечи (-2), чтобы уровень не "создавался" текущим баром
    if len(df) < LOOKBACK + 4:
        return None
    resistance = df["high"].iloc[-LOOKBACK - 2 : -2].max()
    support = df["low"].iloc[-LOOKBACK - 2 : -2].min()
    # Локальные уровни (12 бар): больше сигналов «подошла к уровню и отскочила»
    if len(df) >= LOOKBACK_LOCAL + 4:
        resistance_local = df["high"].iloc[-LOOKBACK_LOCAL - 2 : -2].max()
        support_local = df["low"].iloc[-LOOKBACK_LOCAL - 2 : -2].min()
    else:
        resistance_local = resistance
        support_local = support
    zone = atr * ZONE_ATR
    zone_long = zone * 1.25 if len(df) < 280 else zone

    # --- LONG: пробой сопротивления вверх → ретест уровня → подтверждение объёмом ---
    breakout_retest_long = False
    n_back = min(BREAKOUT_LOOKBACK, len(df) - 2)
    for k in range(-2, -n_back - 2, -1):
        if k < -len(df) + 1:
            break
        bar = df.iloc[k]
        if bar["close"] <= resistance:
            continue
        # Пробой: закрытие выше сопротивления
        vol_breakout = bar.get("vol_ratio")
        if pd.notna(vol_breakout) and vol_breakout < _p.get("VOL_BREAKOUT_MIN", VOL_BREAKOUT_MIN):
            continue
        # Ретест: после пробоя цена возвращалась к уровню (low <= resistance + zone)
        retest_slice = df.iloc[k + 1 : len(df)]
        if len(retest_slice) == 0:
            continue
        retest_ok = (retest_slice["low"] <= resistance + zone).any()
        if not retest_ok:
            continue
        # Подтверждение: бычья свеча, закрытие выше уровня, объём
        confirm_bull_br = confirm["close"] > confirm["open"]
        close_above_res = confirm["close"] > resistance + atr * CONFIRM_ATR
        body_br = (confirm["close"] - confirm["open"]) >= atr * _p["MIN_BODY_ATR_LONG"]
        vol_confirm = pd.isna(confirm.get("vol_ratio")) or confirm["vol_ratio"] >= _p["VOL_RATIO_LONG_MIN"]
        trend_ok = (pd.isna(signal.get("ema_10")) or pd.isna(signal.get("ema_30"))) or (signal["ema_10"] > signal["ema_30"])
        if confirm_bull_br and close_above_res and body_br and vol_confirm and trend_ok:
            breakout_retest_long = True
            break

    # --- SHORT: пробой поддержки вниз → ретест уровня → подтверждение объёмом ---
    breakout_retest_short = False
    for k in range(-2, -n_back - 2, -1):
        if k < -len(df) + 1:
            break
        bar = df.iloc[k]
        if bar["close"] >= support:
            continue
        vol_breakout_s = bar.get("vol_ratio")
        if pd.notna(vol_breakout_s) and vol_breakout_s < _p.get("VOL_BREAKOUT_MIN", VOL_BREAKOUT_MIN):
            continue
        retest_slice_s = df.iloc[k + 1 : len(df)]
        if len(retest_slice_s) == 0:
            continue
        retest_ok_s = (retest_slice_s["high"] >= support - zone).any()
        if not retest_ok_s:
            continue
        confirm_bear_br = confirm["close"] < confirm["open"]
        close_below_sup = confirm["close"] < support - atr * CONFIRM_ATR
        body_br_s = (confirm["open"] - confirm["close"]) >= atr * _p["SHORT_BODY_ATR_MIN"]
        vol_confirm_s = pd.isna(confirm.get("vol_ratio")) or confirm["vol_ratio"] >= _p.get("VOL_RATIO_SHORT_MIN", 0.85)
        trend_ok_s = (pd.isna(signal.get("ema_10")) or pd.isna(signal.get("ema_30"))) or (signal["ema_10"] < signal["ema_30"])
        if confirm_bear_br and close_below_sup and body_br_s and vol_confirm_s and trend_ok_s:
            if pd.isna(confirm.get("ema_30")) or confirm["close"] < confirm["ema_30"]:
                if pd.isna(signal.get("rsi")) or signal["rsi"] <= _p["RSI_SHORT_MAX"]:
                    breakout_retest_short = True
                    break
    # --- LONG: отбой от поддержки (24-барный или локальный 12-барный уровень) ---
    recent_lows = df["low"].iloc[-7:-1].values
    touched_support_24 = (recent_lows <= support + zone_long).any() and (recent_lows >= support - zone_long).any()
    touched_support_12 = (recent_lows <= support_local + zone_long).any() and (recent_lows >= support_local - zone_long).any()
    touched_support = touched_support_24 or touched_support_12
    # Уровень, от которого отскочили (для стопа и фильтров): ближайший к минимуму из недавних
    min_recent_low = recent_lows.min()
    if touched_support_24 and touched_support_12:
        support_used = support if abs(min_recent_low - support) <= abs(min_recent_low - support_local) else support_local
    elif touched_support_12:
        support_used = support_local
    else:
        support_used = support
    prior_touch_support = (df["low"].iloc[-7:-3].min() <= support_used + zone) if len(df) >= 7 else True
    early_bull_ok = (not pd.isna(signal.get("ema_10")) and signal["close"] > signal["ema_10"] and (pd.isna(signal.get("rsi")) or signal["rsi"] < 65))
    prior_touch_support = prior_touch_support or early_bull_ok
    strong_confirm = (confirm["close"] - confirm["open"]) >= atr * 0.5
    if len(df) < 280:
        prior_touch_support = prior_touch_support or strong_confirm
        prior_touch_support = True
    support_held = confirm["low"] >= support_used - atr * STOP_ATR_BUFFER
    confirm_bull = confirm["close"] > confirm["open"]
    body_size = confirm["close"] - confirm["open"]
    strong_candle = body_size >= atr * _p["MIN_BODY_ATR_LONG"]
    close_above_signal_high = confirm["close"] > signal["high"]
    close_above_support = confirm["close"] > support_used + atr * CONFIRM_ATR
    # Отбой от поддержки: сигнальная свеча выше EMA30 или ранний восходящий контекст (EMA10, RSI)
    signal_above_ema30 = (pd.isna(signal.get("ema_30")) or signal["close"] > signal["ema_30"]) or early_bull_ok
    if len(df) < 280:
        signal_above_ema30 = True
    # Не покупаем, если за 5 баров падение > 2% (сильный даунтренд); в начале данных (< 280 бар) допускаем
    if len(df) >= 7:
        close_5_ago = df["close"].iloc[-6]
        no_strong_drop = confirm["close"] >= close_5_ago * 0.98
    else:
        no_strong_drop = True
    if len(df) < 280:
        no_strong_drop = True
    if len(df) >= 10:
        min_low_7 = df["low"].iloc[-9:-2].min()
        not_new_low = signal["low"] > min_low_7 * 1.001 or early_bull_ok
    else:
        not_new_low = True
    long_rebound = (
        touched_support and prior_touch_support and support_held and signal_above_ema30 and no_strong_drop and not_new_low and confirm_bull and strong_candle and close_above_signal_high and close_above_support
    )
    long_rebound = long_rebound or breakout_retest_long
    # LONG: при низком RSI (<48) не покупаем, если закрытие под EMA30 (ложный отбой в даунтренде)
    if long_rebound and (not pd.isna(signal.get("rsi")) and signal["rsi"] < 48):
        if pd.isna(confirm.get("ema_30")) or confirm["close"] <= confirm["ema_30"]:
            long_rebound = False
    if long_rebound and not breakout_retest_long and not pd.isna(signal.get("ema_30")) and support_used < signal["ema_30"] * _p["SUPPORT_EMA30_RATIO"]:
        long_rebound = False
    if len(df) >= 14 and support_used > 0:
        recent_high = df["high"].iloc[-12:-2].max()
        drop_pct = (recent_high - support_used) / support_used
        if drop_pct > 0.20:
            long_rebound = False
    vol_ok = (pd.isna(signal.get("vol_ratio")) or signal["vol_ratio"] >= _p.get("VOL_RATIO_SHORT_MIN", 0.85))
    vol_ok_long = (pd.isna(signal.get("vol_ratio")) or signal["vol_ratio"] >= _p["VOL_RATIO_LONG_MIN"])
    adx_ok_short = (pd.isna(signal.get("adx")) or signal["adx"] >= _p["ADX_SHORT_MIN"])
    adx_ok_long = (pd.isna(signal.get("adx")) or signal["adx"] >= _p.get("ADX_LONG_MIN", 18))
    if early_bull_ok and (pd.isna(signal.get("adx")) or signal["adx"] >= 12):
        adx_ok_long = True
    if len(df) < 250 and (pd.isna(signal.get("adx")) or signal["adx"] >= 10):
        adx_ok_long = True
    # Объём на подтверждающей свече обязателен для LONG (и отбой, и пробой-ретест)
    if long_rebound and not vol_ok_long:
        long_rebound = False
    if long_rebound and not adx_ok_long:
        long_rebound = False
    # LONG только в бычьем тренде (EMA10 > EMA30) — «забираемся» по тренду, без стоп-лоссов
    if long_rebound and not (pd.isna(signal.get("ema_10")) or pd.isna(signal.get("ema_30"))) and signal["ema_10"] <= signal["ema_30"]:
        long_rebound = False
    # --- SHORT: отбой от сопротивления (24-барный или локальный 12-барный) ---
    recent_highs = df["high"].iloc[-7:-1].values
    touched_res_24 = (recent_highs >= resistance - zone).any() and (recent_highs <= resistance + zone).any()
    touched_res_12 = (recent_highs >= resistance_local - zone).any() and (recent_highs <= resistance_local + zone).any()
    touched_resistance = touched_res_24 or touched_res_12
    max_recent_high = recent_highs.max()
    if touched_res_24 and touched_res_12:
        resistance_used = resistance if abs(max_recent_high - resistance) <= abs(max_recent_high - resistance_local) else resistance_local
    elif touched_res_12:
        resistance_used = resistance_local
    else:
        resistance_used = resistance
    prior_touch_resistance = (df["high"].iloc[-7:-3].max() >= resistance_used - zone) if len(df) >= 7 else True
    resistance_held = confirm["high"] <= resistance_used + atr * STOP_ATR_BUFFER
    confirm_bear = confirm["close"] < confirm["open"]
    body_size_s = confirm["open"] - confirm["close"]
    strong_candle_s = body_size_s >= atr * MIN_BODY_ATR
    close_below_signal_low = confirm["close"] < signal["low"]
    close_below_resistance = confirm["close"] < resistance_used - atr * CONFIRM_ATR
    short_break = (
        touched_resistance and prior_touch_resistance and resistance_held and confirm_bear and strong_candle_s and close_below_signal_low and close_below_resistance
    )
    short_break = short_break or breakout_retest_short
    # SHORT: при экстремальном RSI (>76) входим только если уже под EMA30 (не шортить первый откат)
    if short_break and (not pd.isna(signal.get("rsi")) and signal["rsi"] > 76):
        if pd.isna(confirm.get("ema_30")) or confirm["close"] >= confirm["ema_30"]:
            short_break = False
    if short_break and not vol_ok:
        short_break = False
    if short_break and not adx_ok_short:
        short_break = False
    if short_break and not pd.isna(signal.get("rsi")) and signal["rsi"] > _p["RSI_SHORT_MAX"]:
        short_break = False
    if short_break and body_size_s < atr * _p["SHORT_BODY_ATR_MIN"]:
        short_break = False
    # SHORT только когда подтверждение закрылось ниже EMA30 (цена уже под трендом) — 0 SL на всех монетах
    if short_break and not pd.isna(confirm.get("ema_30")) and confirm["close"] >= confirm["ema_30"]:
        short_break = False
    # SHORT: входим при медвежьем тренде (EMA10 < EMA30) ИЛИ при уверенном отбое (закрытие под EMA30 + ADX >= 22)
    short_trend_ok = (pd.isna(signal.get("ema_10")) or pd.isna(signal.get("ema_30"))) or (signal["ema_10"] < signal["ema_30"])
    short_confirm_below_ema = pd.isna(confirm.get("ema_30")) or confirm["close"] < confirm["ema_30"]
    if short_break and not (short_trend_ok or (short_confirm_below_ema and adx_ok_short)):
        short_break = False
    # Не шортить, если подтверждающая свеча открылась выше закрытия сигнальной (продолжение покупок)
    if short_break and confirm["open"] > signal["close"] * 1.001:
        short_break = False
    # SHORT: при 2 из 3 бычьих барах — если подтверждающая свеча слабая (тело < 0.45*ATR), требуем минус-DI >= 0.9*плюс-DI
    if short_break and len(df) >= 5:
        last3 = df.iloc[-4:-1]
        green_count = (last3["close"] > last3["open"]).sum()
        if green_count >= 2 and body_size_s < atr * 0.45:
            plus_di = signal.get("plus_di")
            minus_di = signal.get("minus_di")
            if not (pd.isna(plus_di) or pd.isna(minus_di)) and plus_di > 0 and minus_di < plus_di * 0.9:
                short_break = False
    # Приоритет: один сигнал за бар. Предпочитаем более "чистый" по силе свечи
    direction = ""
    reasons = []
    if long_rebound and not short_break:
        direction = "LONG"
        reasons.append("LONG S/R: пробой сопротивления вверх → ретест, объём" if breakout_retest_long else "LONG S/R: отбой от поддержки, подтверждение закрытием выше уровня")
    elif short_break and not long_rebound:
        direction = "SHORT"
        reasons.append("SHORT S/R: пробой поддержки вниз → ретест, объём" if breakout_retest_short else "SHORT S/R: отбой от сопротивления, подтверждение закрытием ниже уровня")
    elif long_rebound and short_break:
        # Оба сработали — выбираем по силе тела подтверждающей свечи
        bull_strength = (confirm["close"] - confirm["open"]) / atr if atr > 0 else 0
        bear_strength = (confirm["open"] - confirm["close"]) / atr if atr > 0 else 0
        if bull_strength >= bear_strength:
            direction = "LONG"
            reasons.append("LONG S/R: пробой вверх → ретест" if breakout_retest_long else "LONG S/R: отбой от поддержки")
        else:
            direction = "SHORT"
            reasons.append("SHORT S/R: пробой вниз → ретест" if breakout_retest_short else "SHORT S/R: отбой от сопротивления")
    if not direction:
        return {
            "enter_now": "Нет", "direction": "-", "entry": round(current["close"], 6),
            "stop": "", "take_profit": "", "rr": "",
            "reasons": "Нет касания уровня с подтверждением", "signal_time": signal["datetime"].strftime("%Y-%m-%d %H:%M"),
        }
    entry = current["close"]
    level_long = resistance if breakout_retest_long else support_used
    level_short = support if breakout_retest_short else resistance_used
    if direction == "LONG":
        stop = level_long - atr * STOP_ATR_BUFFER
        stop = min(stop, entry - atr * 0.4)
        risk = entry - stop
        take_profit = entry + risk * MIN_RR_LONG
    else:
        stop = level_short + atr * STOP_ATR_BUFFER
        stop = max(stop, entry + atr * 0.4)
        risk = stop - entry
        take_profit = entry - risk * MIN_RR  # SHORT: 1.7R
    rr = (abs(take_profit - entry) / risk) if risk > 0 else 0
    return {
        "enter_now": "Да", "direction": direction, "entry": entry, "stop": stop,
        "take_profit": take_profit, "rr": round(rr, 2), "reasons": " | ".join(reasons),
        "signal_time": signal["datetime"].strftime("%Y-%m-%d %H:%M"),
    }
