import json
import threading
import time
import datetime
import requests
import warnings

import pandas as pd
import pandas_ta as ta
import websocket

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
SYMBOL = 'XAU/USD'
TIMEFRAME = '5min'
LIMIT = 150                 # 拉取的K线数量
TWELVEDATA_API_KEY = '448e2ece2d694647bf506939595c28e4'

DINGTALK_WEBHOOK = ''
ENABLE_NOTIFY = bool(DINGTALK_WEBHOOK)

MACD_FAST   = 6
MACD_SLOW   = 13
MACD_SIGNAL = 5
RSI_PERIOD  = 14

KDJ_PERIOD = 9
KDJ_SIGNAL = 3

DIVERGENCE_LOOKBACK = 30
TREND_CANDLE_COUNT  = 4    # 共振前置条件：前N根同向K线

MACD_WARN_BARS = 2         # MACD柱状图连续收缩根数
KDJ_WARN_GAP   = 5         # KDJ K/D距离阈值

print(f"🤖 机器人启动：监控 {SYMBOL} 黄金 ({TIMEFRAME}K线) MACD+KDJ共振信号...")
print(f"   MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL}) | KDJ({KDJ_PERIOD},{KDJ_SIGNAL},{KDJ_SIGNAL})")
print(f"   实时价格: WebSocket | K线数据: REST API（仅新K线时请求）\n")


# ================= WebSocket 实时价格 =================
_ws_price: float = None        # WebSocket 推送的最新价格（线程间共享）
_ws_price_time: str = None     # WebSocket 推送的价格对应北京时间（字符串）
_ws_connected: bool = False

_CST_OFFSET = datetime.timezone(datetime.timedelta(hours=8))  # 北京时间 UTC+8


def _on_ws_open(ws):
    global _ws_connected
    _ws_connected = True
    ws.send(json.dumps({"action": "subscribe", "params": {"symbols": SYMBOL}}))
    print("✅ WebSocket 已连接，实时价格推送中...")


def _on_ws_message(ws, message):
    global _ws_price, _ws_price_time
    data = json.loads(message)
    if data.get('event') == 'price' and 'price' in data:
        _ws_price = float(data['price'])
        ts = data.get('timestamp')
        if ts:
            cst_dt = datetime.datetime.fromtimestamp(int(ts), tz=_CST_OFFSET)
            _ws_price_time = cst_dt.strftime('%H:%M:%S')
        else:
            _ws_price_time = datetime.datetime.now().strftime('%H:%M:%S')


def _on_ws_error(ws, error):
    global _ws_connected
    _ws_connected = False
    print(f"⚠️ WebSocket 错误: {error}")


def _on_ws_close(ws, close_status_code, close_msg):
    global _ws_connected
    _ws_connected = False
    print("⚠️ WebSocket 断开，5秒后重连...")


def start_websocket():
    """后台线程启动 WebSocket，断线自动重连"""
    def _run():
        while True:
            try:
                ws = websocket.WebSocketApp(
                    f"wss://ws.twelvedata.com/v1/quotes/price?apikey={TWELVEDATA_API_KEY}",
                    on_open=_on_ws_open,
                    on_message=_on_ws_message,
                    on_error=_on_ws_error,
                    on_close=_on_ws_close,
                )
                ws.run_forever()
            except Exception as e:
                print(f"⚠️ WebSocket 异常: {e}")
            time.sleep(5)

    threading.Thread(target=_run, daemon=True).start()


def get_realtime_price():
    """
    获取实时价格，返回 (price, time_str) 元组。
    优先使用 WebSocket 推送数据；不可用时降级到 REST /quote 接口，
    time_str 取 last_quote_at（最新报价时间）转换后的北京时间字符串。
    """
    if _ws_price is not None:
        return _ws_price, _ws_price_time

    try:
        resp = requests.get(
            'https://api.twelvedata.com/quote',
            params={'symbol': SYMBOL, 'apikey': TWELVEDATA_API_KEY},
            timeout=6
        )
        data = resp.json()
        price = float(data['close'])
        # last_quote_at：最新报价的 Unix 时间戳（秒）
        ts = data.get('last_quote_at') or data.get('timestamp')
        if ts:
            cst_dt = datetime.datetime.fromtimestamp(int(ts), tz=_CST_OFFSET)
            time_str = cst_dt.strftime('%H:%M:%S')
        else:
            time_str = None
        return price, time_str
    except Exception as e:
        print(f"⚠️ 实时价格获取失败: {e}")
        return None, None


# ================= 数据获取与指标计算 =================
def fetch_data():
    """从 Twelve Data REST API 获取历史K线数据"""
    try:
        resp = requests.get(
            'https://api.twelvedata.com/time_series',
            params={
                'symbol': SYMBOL,
                'interval': TIMEFRAME,
                'outputsize': LIMIT,
                'timezone': 'Asia/Shanghai',
                'apikey': TWELVEDATA_API_KEY,
            },
            timeout=10
        )
        j = resp.json()
        if j.get('status') != 'ok':
            print(f"❌ K线数据获取失败: {j.get('message', j)}")
            return None

        records = j['values'][::-1]
        df = pd.DataFrame(records)
        df.rename(columns={'datetime': 'timestamp'}, inplace=True)
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col])
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0) if 'volume' in df.columns else 0
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].dropna(subset=['close']).reset_index(drop=True)

    except Exception as e:
        print(f"❌ K线数据获取失败: {e}")
        return None


def calculate_indicators(df):
    """计算 MACD、RSI、KDJ 指标"""
    df = pd.concat([
        df,
        df.ta.macd(fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL),
        df.ta.rsi(length=RSI_PERIOD),
        df.ta.kdj(length=KDJ_PERIOD, signal=KDJ_SIGNAL),
    ], axis=1)
    df.rename(columns={
        f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}':  'macd',
        f'MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}': 'signal_line',
        f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}': 'histogram',
        f'RSI_{RSI_PERIOD}':        'rsi',
        f'K_{KDJ_PERIOD}_{KDJ_SIGNAL}': 'kdj_k',
        f'D_{KDJ_PERIOD}_{KDJ_SIGNAL}': 'kdj_d',
        f'J_{KDJ_PERIOD}_{KDJ_SIGNAL}': 'kdj_j',
    }, inplace=True)
    return df


# ================= K线缓存 =================
_cached_df: pd.DataFrame = None
_cached_candle_ts: str = None


def _expected_candle_ts():
    """计算当前应已完成的K线时间戳（当前5分钟窗口的上一个窗口）"""
    now = datetime.datetime.now()
    window_start = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
    return (window_start - datetime.timedelta(minutes=4)).strftime('%Y-%m-%d %H:%M:%S')


def refresh_kline_if_needed():
    """若出现新K线则重新拉取并计算指标，否则复用缓存"""
    global _cached_df, _cached_candle_ts
    expected = _expected_candle_ts()
    if _cached_df is None or expected != _cached_candle_ts:
        print(f"🔄 新K线({expected})，拉取全量数据...")
        df_new = fetch_data()
        if df_new is None:
            return False
        _cached_df = calculate_indicators(df_new)
        _cached_candle_ts = expected
    return True


# ================= 信号检测 =================
def _detect_macd_signals(last, prev, lookback):
    """检测 MACD 信号集合：水上/水下金叉/死叉 + 顶/底背离"""
    hit = set()
    macd_val, sig_val = last['macd'], last['signal_line']
    prev_macd, prev_sig = prev['macd'], prev['signal_line']

    if any(pd.isna(v) for v in [macd_val, sig_val, prev_macd, prev_sig]):
        return hit

    if prev_macd < prev_sig and macd_val > sig_val:
        hit.add('水上金叉' if macd_val > 0 else '水下金叉')
    elif prev_macd > prev_sig and macd_val < sig_val:
        hit.add('水上死叉' if macd_val > 0 else '水下死叉')

    if not lookback.empty:
        valid = lookback.dropna(subset=['macd'])
        if not valid.empty:
            hi_idx = valid['high'].idxmax()
            if last['high'] > valid.loc[hi_idx, 'high'] and macd_val < valid.loc[hi_idx, 'macd'] and macd_val > 0:
                hit.add('顶背离')
            lo_idx = valid['low'].idxmin()
            if last['low'] < valid.loc[lo_idx, 'low'] and macd_val > valid.loc[lo_idx, 'macd'] and macd_val < 0:
                hit.add('底背离')
    return hit


def _detect_kdj_signals(last, prev):
    """检测 KDJ 信号集合：金叉 / 死叉"""
    hit = set()
    k, d, pk, pd_ = last['kdj_k'], last['kdj_d'], prev['kdj_k'], prev['kdj_d']
    if any(pd.isna(v) for v in [k, d, pk, pd_]):
        return hit
    if pk < pd_ and k > d:
        hit.add('金叉')
    elif pk > pd_ and k < d:
        hit.add('死叉')
    return hit


def _has_trend_candles(df, sig_type, count=TREND_CANDLE_COUNT):
    """
    校验已完成K线前 count 根的方向是否符合信号预期：
      金叉/底背离 → 前N根全为阴线；死叉/顶背离 → 前N根全为阳线
    """
    window = df.iloc[-(count + 2):-2]
    if len(window) < count:
        return False
    if '金叉' in sig_type or sig_type == '底背离':
        return (window['close'] < window['open']).all()
    if '死叉' in sig_type or sig_type == '顶背离':
        return (window['close'] > window['open']).all()
    return False


def check_signals(df):
    """
    MACD+KDJ 共振检测。
    返回 (notify_signals, macd_hit, kdj_hit)：
      - notify_signals: 满足所有条件的通知信号列表
      - macd_hit / kdj_hit: 各自单独命中集合（用于日志）
    """
    if len(df) < MACD_SLOW + MACD_SIGNAL + DIVERGENCE_LOOKBACK:
        return [], set(), set()

    last    = df.iloc[-2]
    prev    = df.iloc[-3]
    lookback = df.iloc[-(DIVERGENCE_LOOKBACK + 2):-2].copy().reset_index(drop=True)

    macd_hit = _detect_macd_signals(last, prev, lookback)
    kdj_hit  = _detect_kdj_signals(last, prev)

    common = set()
    if {s for s in macd_hit if '金叉' in s} and '金叉' in kdj_hit:
        common |= {s for s in macd_hit if '金叉' in s}
    if {s for s in macd_hit if '死叉' in s} and '死叉' in kdj_hit:
        common |= {s for s in macd_hit if '死叉' in s}
    common |= {s for s in macd_hit if '背离' in s}

    signal_configs = {
        '水上金叉': ('🟢', '【MACD水上金叉+KDJ共振 - 看多延续】'),
        '水下金叉': ('🟢', '【MACD水下金叉+KDJ共振 - 底部反转看多】'),
        '水上死叉': ('🔴', '【MACD水上死叉+KDJ共振 - 顶部反转看空】'),
        '水下死叉': ('🔴', '【MACD水下死叉+KDJ共振 - 看空延续】'),
        '顶背离':   ('⚠️', '【MACD顶背离 - 潜在下跌风险】'),
        '底背离':   ('🌟', '【MACD底背离 - 潜在上涨机会】'),
    }

    notify_signals = []
    for sig_type in common:
        if not _has_trend_candles(df, sig_type):
            required = '阴线' if ('金叉' in sig_type or sig_type == '底背离') else '阳线'
            print(f"   ⚡ [{sig_type}] 前{TREND_CANDLE_COUNT}根K线不全为{required}，跳过通知")
            continue
        emoji, title = signal_configs[sig_type]
        notify_signals.append({
            'type': sig_type,
            'message': (
                f"{emoji} {title}\n"
                f"时间: {last['timestamp']}\n"
                f"价格: {last['close']:.2f} USD\n"
                f"MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL}): {last['macd']:.4f} | Signal: {last['signal_line']:.4f}\n"
                f"KDJ({KDJ_PERIOD},{KDJ_SIGNAL},{KDJ_SIGNAL}): K={last['kdj_k']:.1f} D={last['kdj_d']:.1f} J={last['kdj_j']:.1f}\n"
                f"RSI: {last['rsi']:.1f}"
            )
        })
    return notify_signals, macd_hit, kdj_hit


def check_warnings(df):
    """
    预警检测：MACD 柱状图收敛 + KDJ K/D 两线靠近。
    返回 [(来源, 方向, 详情), ...]
    """
    result = []
    if len(df) < MACD_WARN_BARS + 3:
        return result

    last = df.iloc[-2]
    hists = df['histogram'].iloc[-(MACD_WARN_BARS + 2):-1].dropna().tolist()
    if len(hists) >= MACD_WARN_BARS + 1:
        converging = all(abs(hists[i]) < abs(hists[i - 1]) for i in range(len(hists) - MACD_WARN_BARS, len(hists)))
        if converging:
            direction = '金叉' if hists[-1] < 0 else '死叉'
            result.append(('MACD', direction, f"柱状图连续{MACD_WARN_BARS}根收缩，hist={hists[-1]:.4f}"))

    k, d = last['kdj_k'], last['kdj_d']
    if not pd.isna(k) and not pd.isna(d):
        gap = k - d
        if abs(gap) < KDJ_WARN_GAP:
            result.append(('KDJ', '金叉' if gap < 0 else '死叉', f"|K-D|={abs(gap):.1f}，两线即将交叉"))
    return result


# ================= 通知 =================
def send_notification(message):
    """发送钉钉通知（未配置时静默）"""
    if not ENABLE_NOTIFY:
        return
    try:
        resp = requests.post(
            DINGTALK_WEBHOOK,
            json={"msgtype": "text", "text": {"content": f"[黄金机器人]\n{message}"}},
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        print(f"✅ 钉钉通知已发送，状态码: {resp.status_code}")
    except Exception as e:
        print(f"❌ 钉钉通知失败: {e}")


def handle_resonance_notify(sig_type, candle_ts, message, alerted_signals):
    """
    统一处理共振预警和共振信号的打印与通知。
    - 同K线同 sig_type 已处理 → 完全跳过
    - 实际共振且同K线已发过同方向预警 → 打印但不重复通知
    """
    key = f"{sig_type}_{candle_ts}"
    if key in alerted_signals:
        return

    print(f"\n{'=' * 55}")
    print(message)
    print('=' * 55)

    if sig_type.startswith('共振预警_'):
        send_notification(message)
    else:
        direction = '金叉' if '金叉' in sig_type else ('死叉' if '死叉' in sig_type else '')
        warn_key = f"共振预警_{direction}_{candle_ts}" if direction else None
        if warn_key and warn_key in alerted_signals:
            print("   ℹ️ 同根K线已发过共振预警，不重复通知")
        else:
            send_notification(message)

    alerted_signals.add(key)


# ================= 主循环业务方法 =================
def print_market_status(df, price_info):
    """打印当前行情：实时价 + 已完成K线 OHLC + 指标概览"""
    latest = df.iloc[-2]

    def safe(col):
        v = latest[col]
        return v if not pd.isna(v) else float('nan')

    macd_val = safe('macd')
    sig_val  = safe('signal_line')
    macd_trend = "↑多" if macd_val > sig_val else "↓空"

    rsi_val = safe('rsi')
    if rsi_val >= 70:   rsi_trend = "超买⚠️"
    elif rsi_val >= 50: rsi_trend = "偏多↑"
    elif rsi_val >= 30: rsi_trend = "偏空↓"
    else:               rsi_trend = "超卖⚠️"

    try:
        cs = datetime.datetime.strptime(latest['timestamp'], '%Y-%m-%d %H:%M:%S')
        candle_range = f"{cs.strftime('%H:%M')} ~ {(cs + datetime.timedelta(minutes=5)).strftime('%H:%M')}"
    except Exception:
        candle_range = latest['timestamp']

    realtime_price, price_time = price_info
    rt_str = f"{realtime_price:.2f}" if realtime_price else "N/A"
    source = "WebSocket" if (_ws_connected and _ws_price) else "REST"
    time_tag = f"({source} @ {price_time} 北京时间)" if price_time else f"({source})"
    print(f"   ┌─ 实时价: {rt_str} USD {time_tag}")
    print(f"   ├─ 已完成K线 [{candle_range}]  开:{latest['open']:.2f}  高:{latest['high']:.2f}  低:{latest['low']:.2f}  收:{latest['close']:.2f}")
    print(
        f"   ├─ MACD: {macd_val:.3f}/{sig_val:.3f} {macd_trend} | "
        f"KDJ: K{safe('kdj_k'):.1f}/D{safe('kdj_d'):.1f}/J{safe('kdj_j'):.1f} | "
        f"RSI: {rsi_val:.1f}({rsi_trend})"
    )


def process_warnings(df, candle_ts, alerted_signals):
    """预警检测与通知"""
    warn_list = check_warnings(df)
    if not warn_list:
        return

    macd_dirs = {w[1] for w in warn_list if w[0] == 'MACD'}
    kdj_dirs  = {w[1] for w in warn_list if w[0] == 'KDJ'}
    common_dirs = macd_dirs & kdj_dirs

    for source, direction, detail in warn_list:
        emoji = '🟢' if direction == '金叉' else '🔴'
        print(f"   ⏰ [{source}预警] {emoji}{direction}即将发生 | {detail}")

    last = df.iloc[-2]
    for d in common_dirs:
        emoji = '🟢' if d == '金叉' else '🔴'
        print(f"   🚨 【共振预警】MACD+KDJ 同时收敛，{emoji}{d}或将共振！")
        warn_msg = (
            f"🚨 【共振预警 - {emoji}{d}或将发生】\n"
            f"时间: {last['timestamp']}\n"
            f"价格: {last['close']:.2f} USD\n"
            f"MACD柱状图: {last['histogram']:.4f}（收敛中）\n"
            f"KDJ K={last['kdj_k']:.1f} D={last['kdj_d']:.1f} |K-D|={abs(last['kdj_k'] - last['kdj_d']):.1f}（两线靠近）\n"
            f"RSI: {last['rsi']:.1f}"
        )
        handle_resonance_notify(f"共振预警_{d}", candle_ts, warn_msg, alerted_signals)


def process_signals(df, candle_ts, alerted_signals):
    """信号检测、日志打印与通知"""
    notify_signals, macd_hit, kdj_hit = check_signals(df)

    macd_tag_map = {
        '水上金叉': '🟢水上金叉', '水下金叉': '🟢水下金叉',
        '水上死叉': '🔴水上死叉', '水下死叉': '🔴水下死叉',
        '顶背离': '⚠️顶背离', '底背离': '🌟底背离',
    }
    kdj_tag_map = {'金叉': '🟢金叉', '死叉': '🔴死叉'}

    if macd_hit:
        print(f"   📌 MACD单独信号: {' '.join(macd_tag_map.get(s, s) for s in macd_hit)}")
    if kdj_hit:
        print(f"   📌 KDJ单独信号:  {' '.join(kdj_tag_map.get(s, s) for s in kdj_hit)}")
    if not macd_hit and not kdj_hit:
        print("   📊 暂无信号，继续监控...")

    for sig_info in notify_signals:
        handle_resonance_notify(sig_info['type'], candle_ts, sig_info['message'], alerted_signals)


# ================= 工具函数 =================
def is_weekend():
    """判断是否为周末（UTC时间），黄金市场周末休市"""
    return datetime.datetime.utcnow().weekday() >= 5


def sleep_until_next_half_minute():
    """阻塞到下一个 HH:MM:30"""
    now = datetime.datetime.now()
    target = now.replace(second=30, microsecond=0)
    if now >= target:
        target += datetime.timedelta(minutes=1)
    time.sleep(max((target - now).total_seconds(), 0))


# ================= 主循环 =================
alerted_signals: set = set()

start_websocket()

while True:
    try:
        if is_weekend():
            print("⏸️  周末黄金市场休市，1小时后重试...")
            time.sleep(3600)
            break
        if not refresh_kline_if_needed():
            print("   K线获取失败，等待下一个 :30 重试...")
            sleep_until_next_half_minute()
            continue

        df = _cached_df
        candle_ts = df.iloc[-2]['timestamp']

        print_market_status(df, get_realtime_price())
        process_warnings(df, candle_ts, alerted_signals)
        process_signals(df, candle_ts, alerted_signals)

        if len(alerted_signals) > 100:
            alerted_signals = set(list(alerted_signals)[-50:])

    except KeyboardInterrupt:
        print("\n\n👋 用户中断，机器人已停止。")
        break
    except Exception as e:
        print(f"❌ 主循环异常: {e}")

    sleep_until_next_half_minute()
