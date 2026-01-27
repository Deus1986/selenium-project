import time
import pandas as pd
import numpy as np
import requests
import warnings

warnings.filterwarnings('ignore')

DEBUG = False


def get_high_volume_symbols(min_volume=10000000):
    url = "https://contract.mexc.com/api/v1/contract/ticker"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        symbols = []
        if 'data' in data:
            for item in data["data"]:
                if (item["amount24"] > min_volume and
                    item['symbol'].endswith('_USDT') and
                    item['symbol'] != 'USDT_USDT'):
                    symbols.append({
                        'symbol': item['symbol'],
                        'volume_24h': item['amount24'],
                        'price_change_percent': float(item['riseFallRate']) * 100,
                        'last_price': float(item['lastPrice'])
                    })
        return sorted(symbols, key=lambda x: x['volume_24h'], reverse=True)
    except Exception as e:
        print(f"⚠️ Ошибка получения монет: {e}")
        return []


class ReversalPatternScanner:
    def __init__(self):
        self.min_volume = 10000000
        self.timeframe = "Min5"
        self.volume_threshold = 2.5
        self.candles_to_analyze = 8
        self.channel_candles = 30
        self.min_channel_points = 2
        self.rsi_period = 14
        self.poc_lookback = 50
        self.request_delay = 0.25

    def get_candles_simple(self, symbol, interval="Min5", limit=100):
        url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
        params = {"interval": interval, "limit": limit}
        for _ in range(3):
            try:
                response = requests.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('code') == 0:
                        return data
            except Exception:
                time.sleep(0.7)
        return None

    def create_dataframe_from_dict(self, data):
        if not data or not data.get('data'):
            return None
        raw_data = data['data']
        rows = []
        try:
            if isinstance(raw_data, dict) and 'time' in raw_data:
                for i in range(len(raw_data['time'])):
                    rows.append({
                        'timestamp': raw_data['time'][i],
                        'open': float(raw_data['open'][i]),
                        'high': float(raw_data['high'][i]),
                        'low': float(raw_data['low'][i]),
                        'close': float(raw_data['close'][i]),
                        'volume': float(raw_data['vol'][i])
                    })
            elif isinstance(raw_data, list):
                for item in raw_data:
                    if isinstance(item, list) and len(item) >= 6:
                        rows.append({
                            'timestamp': item[0],
                            'open': float(item[1]),
                            'high': float(item[2]),
                            'low': float(item[3]),
                            'close': float(item[4]),
                            'volume': float(item[5])
                        })
            if not rows:
                return None
            df = pd.DataFrame(rows)
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp']).reset_index(drop=True)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
            df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna().reset_index(drop=True)
            return df if len(df) > 15 else None
        except Exception:
            return None

    def add_indicators(self, df):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        df['poc'] = np.nan
        if len(df) >= self.poc_lookback:
            recent = df.tail(self.poc_lookback).copy()
            price_min = recent['low'].min()
            price_max = recent['high'].max()
            if price_max > price_min:
                bin_width = max((price_max - price_min) / 50, 0.0001)
                bins = np.arange(price_min, price_max + bin_width, bin_width)
                if len(bins) > 2:
                    recent['price_bin'] = pd.cut(recent['close'], bins=bins, labels=False)
                    vol_profile = recent.groupby('price_bin')['volume'].sum()
                    if not vol_profile.empty:
                        max_bin = vol_profile.idxmax()
                        poc_price = (bins[int(max_bin)] + bins[int(max_bin) + 1]) / 2
                        df['poc'] = poc_price
        return df

    def is_pinbar(self, row, threshold=2.0):
        body = abs(row['close'] - row['open'])
        upper_shadow = row['high'] - max(row['open'], row['close'])
        lower_shadow = min(row['open'], row['close']) - row['low']
        total_range = row['high'] - row['low']
        if total_range == 0 or body == 0:
            return None
        upper_ratio = upper_shadow / body
        lower_ratio = lower_shadow / body
        if (upper_ratio >= threshold and lower_ratio < 1.0 and
            row['close'] < row['open'] and (row['close'] - row['low']) <= total_range * 0.35):
            return "BEARISH_PINBAR"
        if (lower_ratio >= threshold and upper_ratio < 1.0 and
            row['close'] > row['open'] and (row['high'] - row['close']) <= total_range * 0.35):
            return "BULLISH_PINBAR"
        return None

    def is_engulfing(self, row, prev_row):
        if prev_row is None:
            return None
        if (prev_row['close'] > prev_row['open'] and
            row['close'] < row['open'] and
            row['open'] >= prev_row['close'] and
            row['close'] <= prev_row['open']):
            return "BEARISH_ENGULFING"
        if (prev_row['close'] < prev_row['open'] and
            row['close'] > row['open'] and
            row['open'] <= prev_row['close'] and
            row['close'] >= prev_row['open']):
            return "BULLISH_ENGULFING"
        return None

    def is_hammer_or_hanging(self, row, threshold=2.0):
        body = abs(row['close'] - row['open'])
        upper_shadow = row['high'] - max(row['open'], row['close'])
        lower_shadow = min(row['open'], row['close']) - row['low']
        if body == 0:
            return None
        lower_ratio = lower_shadow / body
        upper_ratio = upper_shadow / body
        if lower_ratio >= threshold and upper_ratio < 1.0:
            return "HAMMER" if row['close'] > row['open'] else "HANGING_MAN"
        return None

    def detect_rsi_divergence(self, df, i):
        if i < 10 or 'rsi' not in df.columns or df['rsi'].isna().sum() > 5:
            return None
        try:
            window = df.iloc[i-5:i+1]
            if len(window) < 6:
                return None
            prices = np.where(window['close'] > window['open'], window['high'], window['low']).values
            rsi_vals = window['rsi'].values
            curr_price, curr_rsi = prices[-1], rsi_vals[-1]
            peaks = []
            for j in range(1, len(prices)-1):
                if prices[j] > prices[j-1] and prices[j] > prices[j+1]:
                    peaks.append((prices[j], rsi_vals[j]))
            if peaks:
                max_price_prev, max_rsi_prev = max(peaks, key=lambda x: x[0])
                if curr_price > max_price_prev * 1.001 and curr_rsi < max_rsi_prev - 1.5:
                    return "BEARISH_DIVERGENCE"
                if curr_price < max_price_prev * 0.999 and curr_rsi > max_rsi_prev + 1.5:
                    return "BULLISH_DIVERGENCE"
        except:
            pass
        return None

    def calculate_entry_sl_tp(self, pattern, row, df, is_bullish, is_bearish):
        current_price = df['close'].iloc[-1]

        if is_bullish:
            entry_low = min(row['open'], row['close'])
            entry_high = max(row['open'], row['close'])
        else:
            entry_low = max(row['open'], row['close']) * 0.998
            entry_high = row['high'] * 0.999

        if pattern.get('near_poc', False) and not pd.isna(df['poc'].iloc[-1]):
            poc = df['poc'].iloc[-1]
            if row['low'] <= poc <= row['high']:
                entry_low = min(entry_low, poc)
                entry_high = max(entry_high, poc)

        entry_mid = (entry_low + entry_high) / 2

        sl_buffer = 0.002
        if is_bullish:
            sl = row['low'] * (1 - sl_buffer)
            risk = entry_mid - sl
        else:
            sl = row['high'] * (1 + sl_buffer)
            risk = sl - entry_mid

        if risk <= 0:
            return None

        tp1 = entry_mid + (2 * risk if is_bullish else -2 * risk)
        tp2 = entry_mid + (3 * risk if is_bullish else -3 * risk)

        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        if is_bullish:
            tp1 = min(tp1, recent_high * 0.995)
            tp2 = min(tp2, recent_high * 0.99)
        else:
            tp1 = max(tp1, recent_low * 1.005)
            tp2 = max(tp2, recent_low * 1.01)

        signal_status = "АКТУАЛЕН"
        signal_valid = True
        side = "BUY" if is_bullish else "SELL"

        if is_bullish:
            if current_price > entry_high * 1.003:
                signal_status = "ПРОСРОЧЕН (цена ушла вверх)"
                signal_valid = False
            elif current_price < entry_low:
                signal_status = "ОЖИДАНИЕ ОТКАТА"
        else:
            if current_price < entry_low * 0.997:
                signal_status = "ПРОСРОЧЕН (цена ушла вниз)"
                signal_valid = False
            elif current_price > entry_high:
                signal_status = "ОЖИДАНИЕ ОТКАТА"

        rr1 = abs(tp1 - entry_mid) / risk
        rr2 = abs(tp2 - entry_mid) / risk

        return {
            'side': side,
            'entry_zone': (round(entry_low, 8), round(entry_high, 8)),
            'entry_mid': round(entry_mid, 8),
            'sl': round(sl, 8),
            'tp1': round(tp1, 8),
            'tp2': round(tp2, 8),
            'risk': round(risk, 8),
            'risk_percent': round(risk / entry_mid * 100, 2),
            'rr1': round(rr1, 1),
            'rr2': round(rr2, 1),
            'signal_valid': signal_valid,
            'signal_status': signal_status
        }

    def detect_channel_reversals(self, df, channel_candles=None, min_channel_points=None):
        if channel_candles is None:
            channel_candles = self.channel_candles
        if min_channel_points is None:
            min_channel_points = self.min_channel_points

        if len(df) < channel_candles:
            return []

        patterns = []

        highs = df['high'].values
        peaks = []
        for i in range(5, len(highs) - 5):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2] and
                highs[i] >= np.percentile(highs[i-5:i+6], 80)):
                peaks.append((i, highs[i]))

        lows = df['low'].values
        bottoms = []
        for i in range(5, len(lows) - 5):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2] and
                lows[i] <= np.percentile(lows[i-5:i+6], 20)):
                bottoms.append((i, lows[i]))

        if len(peaks) >= min_channel_points:
            n_peaks = min(4, len(peaks))
            recent_peaks = peaks[-n_peaks:]
            x_vals = np.array([p[0] for p in recent_peaks])
            y_vals = np.array([p[1] for p in recent_peaks])
            try:
                slope_high, intercept_high = np.polyfit(x_vals, y_vals, 1)
                y_pred = slope_high * x_vals + intercept_high
                ss_res = np.sum((y_vals - y_pred) ** 2)
                ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
                r2_high = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                if slope_high < -0.00005 and r2_high >= 0.4:
                    for offset in range(1, 6):
                        i = len(df) - offset
                        if i < 10:
                            continue
                        channel_high_at_i = slope_high * i + intercept_high
                        current_high = df['high'].iloc[i]
                        tolerance = channel_high_at_i * 0.003
                        if abs(current_high - channel_high_at_i) <= tolerance:
                            if recent_peaks:
                                last_peak_idx, last_peak_price = recent_peaks[-1]
                                if i > last_peak_idx + 3:
                                    pullback = (last_peak_price - df['low'].iloc[last_peak_idx:i].min()) / last_peak_price
                                    if pullback > 0.015:
                                        touch_conf = 1.0 - abs(current_high - channel_high_at_i) / tolerance
                                        patterns.append({
                                            'type': 'SECOND_TOP_CHANNEL',
                                            'line_price': channel_high_at_i,
                                            'index': i,
                                            'price': current_high,
                                            'touch_confidence': max(0.5, min(1.0, touch_conf)),
                                            'slope': slope_high,
                                            'r2': r2_high,
                                            'pullback_pct': pullback * 100
                                        })
            except:
                pass

        if len(bottoms) >= min_channel_points:
            n_bottoms = min(4, len(bottoms))
            recent_bottoms = bottoms[-n_bottoms:]
            x_vals = np.array([b[0] for b in recent_bottoms])
            y_vals = np.array([b[1] for b in recent_bottoms])
            try:
                slope_low, intercept_low = np.polyfit(x_vals, y_vals, 1)
                y_pred = slope_low * x_vals + intercept_low
                ss_res = np.sum((y_vals - y_pred) ** 2)
                ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
                r2_low = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                if slope_low > 0.00005 and r2_low >= 0.4:
                    for offset in range(1, 6):
                        i = len(df) - offset
                        if i < 10:
                            continue
                        channel_low_at_i = slope_low * i + intercept_low
                        current_low = df['low'].iloc[i]
                        tolerance = channel_low_at_i * 0.003
                        if abs(current_low - channel_low_at_i) <= tolerance:
                            if recent_bottoms:
                                last_bottom_idx, last_bottom_price = recent_bottoms[-1]
                                if i > last_bottom_idx + 3:
                                    pullup = (df['high'].iloc[last_bottom_idx:i].max() - last_bottom_price) / last_bottom_price
                                    if pullup > 0.015:
                                        touch_conf = 1.0 - abs(current_low - channel_low_at_i) / tolerance
                                        patterns.append({
                                            'type': 'SECOND_BOTTOM_CHANNEL',
                                            'line_price': channel_low_at_i,
                                            'index': i,
                                            'price': current_low,
                                            'touch_confidence': max(0.5, min(1.0, touch_conf)),
                                            'slope': slope_low,
                                            'r2': r2_low,
                                            'pullback_pct': pullup * 100
                                        })
            except:
                pass

        return patterns

    def analyze_patterns(self, df):
        if len(df) < 35:
            return []
        df = self.add_indicators(df)
        avg_vol = df['volume'].tail(30).mean()
        if avg_vol <= 0:
            return []
        results = []

        for offset in range(1, min(self.candles_to_analyze + 1, len(df) - 15)):
            i = len(df) - offset
            if i < 15:
                continue
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            vol_ratio = row['volume'] / avg_vol
            if vol_ratio < self.volume_threshold:
                continue

            patterns = []
            pin = self.is_pinbar(row)
            if pin: patterns.append(pin)
            engulf = self.is_engulfing(row, prev_row)
            if engulf: patterns.append(engulf)
            hammer = self.is_hammer_or_hanging(row)
            if hammer: patterns.append(hammer)
            if not patterns:
                continue

            divergence = self.detect_rsi_divergence(df, i) if offset > 1 else None

            near_poc = False
            poc_dist_pct = 0.0
            if not pd.isna(df['poc'].iloc[-1]):
                poc = df['poc'].iloc[-1]
                poc_dist_pct = abs(row['close'] - poc) / poc * 100
                if poc_dist_pct <= 2.0:
                    near_poc = True

            is_bullish = any(p in patterns for p in ["BULLISH_PINBAR", "BULLISH_ENGULFING", "HAMMER"])
            is_bearish = any(p in patterns for p in ["BEARISH_PINBAR", "BEARISH_ENGULFING", "HANGING_MAN"])

            setup = None
            if is_bullish or is_bearish:
                setup = self.calculate_entry_sl_tp({
                    'near_poc': near_poc,
                    'poc_distance_percent': poc_dist_pct
                }, row, df, is_bullish, is_bearish)

            pattern_info = {
                'datetime': row['datetime'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'volume_ratio': round(vol_ratio, 2),
                'age_index': offset,
                'patterns': patterns,
                'divergence': divergence,
                'near_poc': near_poc,
                'poc_distance_percent': round(poc_dist_pct, 2),
                'is_bullish': is_bullish,
                'is_bearish': is_bearish,
                'trade_setup': setup
            }

            if DEBUG or (setup and setup['signal_valid']):
                results.append(pattern_info)

        channel_patterns = self.detect_channel_reversals(df)
        for cp in channel_patterns:
            i = cp['index']
            row = df.iloc[i]
            vol_ratio = row['volume'] / avg_vol
            if vol_ratio < self.volume_threshold:
                continue

            near_poc = False
            poc_dist_pct = 0.0
            if not pd.isna(df['poc'].iloc[-1]):
                poc = df['poc'].iloc[-1]
                dist = abs(cp['price'] - poc) / poc * 100
                if dist <= 2.0:
                    near_poc = True
                    poc_dist_pct = dist

            is_bullish = cp['type'] == 'SECOND_BOTTOM_CHANNEL'
            is_bearish = cp['type'] == 'SECOND_TOP_CHANNEL'

            setup = self.calculate_entry_sl_tp({
                'near_poc': near_poc,
                'poc_distance_percent': poc_dist_pct
            }, row, df, is_bullish, is_bearish)

            pattern_info = {
                'datetime': row['datetime'],
                'open': row['open'],
                'high': cp['price'] if is_bearish else row['high'],
                'low': cp['price'] if is_bullish else row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'volume_ratio': round(vol_ratio, 2),
                'age_index': len(df) - i,
                'patterns': [cp['type']],
                'divergence': None,
                'near_poc': near_poc,
                'poc_distance_percent': round(poc_dist_pct, 2),
                'is_bullish': is_bullish,
                'is_bearish': is_bearish,
                'trade_setup': setup,
                'channel_details': {
                    'slope': cp['slope'],
                    'r2': cp['r2'],
                    'pullback_pct': round(cp['pullback_pct'], 1),
                    'touch_confidence': cp['touch_confidence']
                }
            }

            if DEBUG or (setup and setup['signal_valid']):
                results.append(pattern_info)

        return results

    def analyze_symbol(self, symbol):
        try:
            data = self.get_candles_simple(symbol, self.timeframe, 100)
            if not data:
                return None
            df = self.create_dataframe_from_dict(data)
            if df is None or len(df) < 30:
                return None
            patterns = self.analyze_patterns(df)
            if not patterns:
                return None
            return {
                'symbol': symbol,
                'patterns': patterns,
                'current_price': df['close'].iloc[-1],
                'poc': df['poc'].iloc[-1] if 'poc' in df.columns else None
            }
        except Exception:
            return None

    def scan_for_reversals(self, symbol_count=100, filter_side=None):
        print(f"🎯 СКАНЕР: топ-{symbol_count} | ТФ: {self.timeframe} | Канал: {self.channel_candles} свечей")
        symbols_data = get_high_volume_symbols(self.min_volume)
        symbols = [item['symbol'] for item in symbols_data[:symbol_count]]
        print(f"📊 Анализ {len(symbols)} монет...")

        results = []
        for sym in symbols:
            res = self.analyze_symbol(sym)
            if res:
                results.append(res)
            time.sleep(self.request_delay)

        buy_count = sell_count = 0
        for r in results:
            for p in r['patterns']:
                setup = p.get('trade_setup')
                if setup and setup['signal_valid']:
                    if setup['side'] == "BUY":
                        buy_count += 1
                    else:
                        sell_count += 1
        print(f"\n📈 BUY: {buy_count} | 📉 SELL: {sell_count} | Всего: {buy_count + sell_count}")

        if not results:
            print("❌ Нет сигналов")
            return []

        def score(r):
            valid = [p for p in r['patterns'] if p.get('trade_setup', {}).get('signal_valid')]
            if not valid:
                return 0
            best = max(valid, key=lambda x: x['volume_ratio'])
            s = best['volume_ratio']
            if best['divergence']: s *= 1.3
            if best['near_poc']: s *= 1.15
            if 'SECOND_TOP_CHANNEL' in best['patterns'] or 'SECOND_BOTTOM_CHANNEL' in best['patterns']:
                s *= 1.25
            return s

        results.sort(key=score, reverse=True)

        shown = 0
        for r in results:
            valid_patterns = [
                p for p in r['patterns']
                if p.get('trade_setup') and p['trade_setup']['signal_valid'] and
                (filter_side is None or p['trade_setup']['side'] == filter_side)
            ]
            if not valid_patterns:
                continue

            best = max(valid_patterns, key=lambda x: x['volume_ratio'])
            setup = best['trade_setup']
            entry_low, entry_high = setup['entry_zone']

            print(f"\n{shown+1}. **{r['symbol']}** @ {r['current_price']:.6f}")
            print(f"   🪄 {', '.join(best['patterns'])} | x{best['volume_ratio']:.1f}")
            if best['divergence']:
                print(f"   📉 {best['divergence']}")
            if best.get('channel_details'):
                cd = best['channel_details']
                direction = "↓" if 'TOP' in best['patterns'][0] else "↑"
                print(f"   📈 Канал {direction}: наклон {cd['slope']:+.6f}, R²={cd['r2']}, движение {cd['pullback_pct']}%, точность {cd['touch_confidence']}")
            if best['near_poc'] and r['poc']:
                print(f"   🎯 POC: {r['poc']:.4f} (Δ{best['poc_distance_percent']:.1f}%)")
            print(f"   {setup['side']} {entry_low:.4f}–{entry_high:.4f} (mid: {setup['entry_mid']:.4f})")
            print(f"   🛑 SL: {setup['sl']:.4f}")
            print(f"   🎯 TP1: {setup['tp1']:.4f} (RR {setup['rr1']}x)")
            print(f"   💡 {setup['signal_status']}")

            shown += 1
            if shown >= 15:
                break

        return results

    def get_detailed_analysis(self, symbol):
        print(f"\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ {symbol}...")
        res = self.analyze_symbol(symbol)
        if not res:
            print("❌ Нет данных")
            return

        print(f"📊 Текущая цена: {res['current_price']:.6f}")
        if res['poc']:
            print(f"📍 POC (50 свечей): {res['poc']:.6f}")

        has_valid = False
        for pat in res['patterns']:
            setup = pat.get('trade_setup')
            if not setup:
                continue

            ts = pat['datetime'].strftime('%d.%m %H:%M')
            age_str = f"{pat['age_index']} свеча назад"
            print(f"\n📌 {ts} | {age_str}")
            print(f"   Паттерны: {', '.join(pat['patterns'])} | x{pat['volume_ratio']:.1f}")
            if pat['divergence']:
                print(f"   Дивергенция: {pat['divergence']}")
            if pat.get('channel_details'):
                cd = pat['channel_details']
                arrow = "↓" if 'TOP' in pat['patterns'][0] else "↑"
                print(f"   Детали канала {arrow}: наклон={cd['slope']:+.6f}, R²={cd['r2']}, движение={cd['pullback_pct']}%, точность={cd['touch_confidence']}")
            if setup['signal_valid']:
                has_valid = True
                entry_low, entry_high = setup['entry_zone']
                print(f"   ✅ {setup['side']}: {entry_low:.6f}–{entry_high:.6f} (mid: {setup['entry_mid']:.4f})")
                print(f"   🛑 SL: {setup['sl']:.6f} ({setup['risk_percent']}% риска)")
                print(f"   🎯 TP1: {setup['tp1']:.6f} (RR {setup['rr1']}x)")
                print(f"   🎯 TP2: {setup['tp2']:.6f} (RR {setup['rr2']}x)")
                print(f"   💡 {setup['signal_status']}")
            else:
                print(f"   ❌ {setup['signal_status']}")

        if not has_valid:
            print("\n⚠️ Нет актуальных сэтапов")

    def export_signals_to_csv(self, results, filename="signals.csv"):
        import csv
        rows = []
        for r in results:
            for pat in r['patterns']:
                setup = pat.get('trade_setup')
                if setup and setup['signal_valid']:
                    row = {
                        'symbol': r['symbol'],
                        'side': setup['side'],
                        'entry_low': setup['entry_zone'][0],
                        'entry_high': setup['entry_zone'][1],
                        'entry_mid': setup['entry_mid'],
                        'sl': setup['sl'],
                        'tp1': setup['tp1'],
                        'tp2': setup['tp2'],
                        'rr1': setup['rr1'],
                        'rr2': setup['rr2'],
                        'volume_ratio': pat['volume_ratio'],
                        'patterns': ";".join(pat['patterns']),
                        'divergence': pat['divergence'] or "",
                        'near_poc': pat['near_poc'],
                        'status': setup['signal_status']
                    }
                    if pat.get('channel_details'):
                        cd = pat['channel_details']
                        row.update({
                            'channel_type': 'TOP' if 'TOP' in pat['patterns'][0] else 'BOTTOM',
                            'channel_slope': cd['slope'],
                            'channel_r2': cd['r2'],
                            'move_pct': cd['pullback_pct'],
                            'touch_confidence': cd['touch_confidence']
                        })
                    rows.append(row)
        if rows:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"✅ Экспортировано {len(rows)} сигналов в {filename}")
        else:
            print("❌ Нет сигналов для экспорта")


def main():
    scanner = ReversalPatternScanner()
    print("🚀 СКАНЕР: ВТОРАЯ ВЕРШИНА + ВТОРОЕ ДНО В КАНАЛЕ (исправлено, работает!)")
    print("=" * 76)

    while True:
        print(f"\nТекущие настройки:")
        print(f"  ТФ: {scanner.timeframe} | Канал: {scanner.channel_candles} свечей")
        print(f"  Паттерны: последние {scanner.candles_to_analyze} свечей")
        print("\n1 — Сканировать топ-100 (все)")
        print("2 — Только BUY (включая SECOND_BOTTOM_CHANNEL)")
        print("3 — Только SELL (включая SECOND_TOP_CHANNEL)")
        print("4 — Детальный анализ")
        print("5 — Экспорт в CSV")
        print("6 — Настроить параметры")
        print("7 — Выход")
        choice = input("\n→ ").strip()

        if choice == "1":
            scanner.scan_for_reversals(100)
        elif choice == "2":
            scanner.scan_for_reversals(100, "BUY")
        elif choice == "3":
            scanner.scan_for_reversals(100, "SELL")
        elif choice == "4":
            sym = input("Символ (e.g. ARBUSDT): ").strip().upper()
            if sym and not sym.endswith('_USDT'):
                sym += '_USDT'
            if sym:
                scanner.get_detailed_analysis(sym)
        elif choice == "5":
            print("Сканирование для экспорта...")
            results = scanner.scan_for_reversals(50)
            scanner.export_signals_to_csv(results)
        elif choice == "6":
            print("\n--- НАСТРОЙКИ ---")
            try:
                tf = input(f"Таймфрейм (Min1/Min5/Min15/Min60/Min240) [текущий: {scanner.timeframe}]: ").strip()
                if tf:
                    valid_tfs = ["Min1", "Min5", "Min15", "Min30", "Min60", "Min240"]
                    if tf in valid_tfs:
                        scanner.timeframe = tf
                    else:
                        print("⚠️ Неверный таймфрейм — оставлено как есть.")

                cc = input(f"Свечей для анализа канала (число, по умолч. 30) [текущий: {scanner.channel_candles}]: ").strip()
                if cc.isdigit():
                    scanner.channel_candles = int(cc)

                nc = input(f"Мин. точек для канала (2–4) [текущий: {scanner.min_channel_points}]: ").strip()
                if nc.isdigit() and 2 <= int(nc) <= 4:
                    scanner.min_channel_points = int(nc)

                ca = input(f"Свечей для паттернов (1–15) [текущий: {scanner.candles_to_analyze}]: ").strip()
                if ca.isdigit() and 1 <= int(ca) <= 15:
                    scanner.candles_to_analyze = int(ca)

                vt = input(f"Мин. объём (x от среднего, напр. 2.5) [текущий: {scanner.volume_threshold}]: ").strip()
                if vt.replace('.', '').isdigit():
                    scanner.volume_threshold = float(vt)

                print("✅ Настройки обновлены.")
            except Exception as e:
                print(f"❌ Ошибка: {e}")
        elif choice == "7":
            print("✅ Выход")
            break
        else:
            print("Неверный выбор")


if __name__ == "__main__":
    main()