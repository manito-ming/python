import base64
import datetime
import hashlib
import hmac
import time
import urllib.parse
import warnings

import pandas as pd
import pandas_ta as ta
import requests

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
SYMBOL              = 'XAU/USD'
TIMEFRAME           = '1day'
LIMIT               = 200                # 拉取的日K数量，保证MA52有足够历史
TWELVEDATA_API_KEY  = '448e2ece2d694647bf506939595c28e4'

# 钉钉机器人配置（在群 → 智能群助手 → 自定义机器人 → 安全设置选"加签"获取）
DINGTALK_WEBHOOK    = ''   # 完整 Webhook 地址，例如 https://oapi.dingtalk.com/robot/send?access_token=xxx
DINGTALK_SECRET     = ''   # 加签 Secret，例如 SECxxx；不启用加签则留空

# Server酱 Turbo 微信推送（sct.ftqq.com 登录后获取 SendKey，SCT 开头）
SERVERCHAN_KEY      = 'SCT351227TUkPByGmf6v6mRlEWz2xSqtkH'   # 填入 SendKey 后自动启用微信推送，免费 5条/天

ENABLE_NOTIFY       = bool(DINGTALK_WEBHOOK) or bool(SERVERCHAN_KEY)

MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9

MA_PERIODS   = [7, 14, 52]   # 短期、中期、长期均线

RUN_HOUR     = 9
RUN_MINUTE   = 30             # 每日 09:30 执行

_CST_OFFSET  = datetime.timezone(datetime.timedelta(hours=8))

print(f"📅 日K趋势分析启动：{SYMBOL}")
print(f"   MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL}) | MA{MA_PERIODS}")
print(f"   每日 {RUN_HOUR:02d}:{RUN_MINUTE:02d} 北京时间执行\n")


# ================= 数据获取 =================
def fetch_daily_data():
    """从 Twelve Data 拉取日K线数据"""
    try:
        resp = requests.get(
            'https://api.twelvedata.com/time_series',
            params={
                'symbol':     SYMBOL,
                'interval':   TIMEFRAME,
                'outputsize': LIMIT,
                'timezone':   'Asia/Shanghai',
                'apikey':     TWELVEDATA_API_KEY,
            },
            timeout=15
        )
        j = resp.json()
        if j.get('status') != 'ok':
            print(f"❌ 日K数据获取失败: {j.get('message', j)}")
            return None

        records = j['values'][::-1]   # 升序排列（旧→新）
        df = pd.DataFrame(records)
        df.rename(columns={'datetime': 'date'}, inplace=True)
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col])
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0) if 'volume' in df.columns else 0
        return df[['date', 'open', 'high', 'low', 'close', 'volume']].dropna(subset=['close']).reset_index(drop=True)

    except Exception as e:
        print(f"❌ 日K数据获取失败: {e}")
        return None


# ================= 指标计算 =================
def calculate_indicators(df):
    """计算 MACD 和 MA7/MA14/MA52"""
    # MACD
    macd_df = df.ta.macd(fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    df = pd.concat([df, macd_df], axis=1)
    df.rename(columns={
        f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}':  'macd',
        f'MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}': 'signal_line',
        f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}': 'histogram',
    }, inplace=True)

    # MA7 / MA14 / MA52
    for p in MA_PERIODS:
        df[f'ma{p}'] = df['close'].rolling(window=p).mean()

    return df


# ================= 趋势判断 =================
def _ma_trend(df):
    """
    均线趋势判断：
      多头排列 → ma7 > ma14 > ma52
      空头排列 → ma7 < ma14 < ma52
      否则返回 '震荡'
    """
    last = df.iloc[-1]
    ma7, ma14, ma52 = last['ma7'], last['ma14'], last['ma52']
    if pd.isna(ma7) or pd.isna(ma14) or pd.isna(ma52):
        return '数据不足'
    if ma7 > ma14 > ma52:
        return '多头排列'
    if ma7 < ma14 < ma52:
        return '空头排列'
    return '震荡'


def _ma_price_position(df):
    """判断收盘价相对于各均线的位置"""
    last  = df.iloc[-1]
    close = last['close']
    positions = []
    for p in MA_PERIODS:
        ma_val = last[f'ma{p}']
        if pd.isna(ma_val):
            continue
        tag = '上方' if close > ma_val else '下方'
        positions.append(f"MA{p}({ma_val:.1f}){tag}")
    return positions


def _macd_state(df):
    """
    MACD 状态判断：
      - 多空方向（macd > signal → 多头）
      - 位置（水上/水下）
      - 金叉/死叉（本日与昨日对比）
      - 柱状图趋势（扩张/收缩）
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]

    if any(pd.isna(v) for v in [last['macd'], last['signal_line'], prev['macd'], prev['signal_line']]):
        return {}

    macd_val = last['macd']
    sig_val  = last['signal_line']
    prev_macd = prev['macd']
    prev_sig  = prev['signal_line']

    result = {}
    result['direction']  = '多头' if macd_val > sig_val else '空头'
    result['position']   = '水上' if macd_val > 0 else '水下'
    result['histogram']  = last['histogram']
    result['prev_histogram'] = prev['histogram']

    if prev_macd < prev_sig and macd_val > sig_val:
        result['cross'] = '金叉'
    elif prev_macd > prev_sig and macd_val < sig_val:
        result['cross'] = '死叉'
    else:
        result['cross'] = None

    if not pd.isna(last['histogram']) and not pd.isna(prev['histogram']):
        result['hist_trend'] = '扩张' if abs(last['histogram']) > abs(prev['histogram']) else '收缩'
    else:
        result['hist_trend'] = None

    return result


def _detect_divergence(df, lookback=60):
    """
    顶背离：价格创新高但 MACD 未创新高（macd > 0）
    底背离：价格创新低但 MACD 未创新低（macd < 0）
    """
    window = df.iloc[-(lookback + 1):-1].dropna(subset=['macd']).copy()
    if window.empty:
        return []

    last    = df.iloc[-1]
    signals = []

    if not pd.isna(last['macd']) and last['macd'] > 0:
        hi_idx = window['high'].idxmax()
        if last['high'] > window.loc[hi_idx, 'high'] and last['macd'] < window.loc[hi_idx, 'macd']:
            signals.append('顶背离')

    if not pd.isna(last['macd']) and last['macd'] < 0:
        lo_idx = window['low'].idxmin()
        if last['low'] < window.loc[lo_idx, 'low'] and last['macd'] > window.loc[lo_idx, 'macd']:
            signals.append('底背离')

    return signals


def _overall_trend(ma_trend, macd_state):
    """综合均线排列和MACD给出整体趋势判断"""
    if not macd_state:
        return ma_trend
    macd_dir = macd_state.get('direction', '')
    if ma_trend == '多头排列' and macd_dir == '多头':
        return '强势上涨'
    if ma_trend == '空头排列' and macd_dir == '空头':
        return '强势下跌'
    if ma_trend == '多头排列' and macd_dir == '空头':
        return '上涨回调'
    if ma_trend == '空头排列' and macd_dir == '多头':
        return '下跌反弹'
    return '震荡整理'


# ================= 通知 =================
def _build_webhook_url():
    """若配置了加签 Secret，拼接 timestamp+sign 参数后返回完整 URL"""
    if not DINGTALK_SECRET:
        return DINGTALK_WEBHOOK
    timestamp = str(round(time.time() * 1000))
    string_to_sign = f"{timestamp}\n{DINGTALK_SECRET}"
    hmac_code = hmac.new(
        DINGTALK_SECRET.encode('utf-8'),
        string_to_sign.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
    return f"{DINGTALK_WEBHOOK}&timestamp={timestamp}&sign={sign}"


def _send_dingtalk(message):
    """发送钉钉通知（未配置时跳过）"""
    if not DINGTALK_WEBHOOK:
        return
    try:
        url = _build_webhook_url()
        resp = requests.post(
            url,
            json={"msgtype": "text", "text": {"content": f"[黄金日K分析]\n{message}"}},
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        print(f"✅ 钉钉通知已发送，状态码: {resp.status_code}")
    except Exception as e:
        print(f"❌ 钉钉通知失败: {e}")


def _send_serverchan(title, message):
    """通过 Server酱 Turbo 发送微信消息（未配置时跳过），免费 5条/天"""
    if not SERVERCHAN_KEY:
        return
    try:
        resp = requests.post(
            f'https://sctapi.ftqq.com/{SERVERCHAN_KEY}.send',
            data={'title': title, 'desp': message},
            timeout=5
        )
        data = resp.json()
        if data.get('data', {}).get('errno') == 0 or data.get('code') == 0:
            print(f"✅ Server酱微信推送成功")
        else:
            print(f"⚠️ Server酱推送返回: {data}")
    except Exception as e:
        print(f"❌ Server酱推送失败: {e}")


def send_notification(title, message):
    """同时发送钉钉 + Server酱微信推送（各自未配置时静默）"""
    if not ENABLE_NOTIFY:
        return
    _send_dingtalk(f"{title}\n{message}")
    _send_serverchan(title, message)


# ================= 主分析任务 =================
def run_daily_analysis():
    """拉取日K、计算指标、打印趋势报告并发送通知"""
    now = datetime.datetime.now(_CST_OFFSET)
    print(f"\n{'=' * 60}")
    print(f"📅 日K趋势分析  {now.strftime('%Y-%m-%d %H:%M:%S')} 北京时间")
    print('=' * 60)

    df = fetch_daily_data()
    if df is None:
        print("❌ 数据获取失败，跳过本次分析")
        return

    df = calculate_indicators(df)

    last         = df.iloc[-1]
    prev         = df.iloc[-2]
    ma_trend     = _ma_trend(df)
    price_pos    = _ma_price_position(df)
    macd_state   = _macd_state(df)
    divergences  = _detect_divergence(df)
    trend        = _overall_trend(ma_trend, macd_state)

    # 趋势 emoji
    trend_emoji = {
        '强势上涨': '🚀', '上涨回调': '📈',
        '强势下跌': '💥', '下跌反弹': '📉',
        '震荡整理': '🔄',
    }.get(trend, '❓')

    # ---- 打印报告 ----
    print(f"   日期:    {last['date']}")
    print(f"   收盘价:  {last['close']:.2f} USD  "
          f"(开:{last['open']:.2f} 高:{last['high']:.2f} 低:{last['low']:.2f})")
    print(f"   前日收:  {prev['close']:.2f} USD  "
          f"涨跌: {last['close'] - prev['close']:+.2f} "
          f"({(last['close'] / prev['close'] - 1) * 100:+.2f}%)")

    print(f"\n── 均线指标 ──────────────────────────────")
    for p in MA_PERIODS:
        val = last[f'ma{p}']
        tag = '↑ 价格在上方' if last['close'] > val else '↓ 价格在下方'
        print(f"   MA{p:<3}: {val:.2f}  {tag}")
    print(f"   均线排列: {ma_trend}")

    print(f"\n── MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL}) ──────────────────────────")
    if macd_state:
        cross_tag = f"  ⚡{macd_state['cross']}!" if macd_state.get('cross') else ''
        print(f"   MACD:     {last['macd']:.4f}  Signal: {last['signal_line']:.4f}{cross_tag}")
        print(f"   柱状图:   {last['histogram']:.4f}  ({macd_state.get('hist_trend', '-')})")
        print(f"   方向/位置: {macd_state['direction']} / {macd_state['position']}")

    if divergences:
        for d in divergences:
            emoji = '⚠️' if d == '顶背离' else '🌟'
            print(f"   {emoji} 检测到【{d}】信号！")

    print(f"\n── 综合趋势判断 ───────────────────────────")
    print(f"   {trend_emoji} {trend}")
    print('=' * 60)

    # ---- 构建通知消息 ----
    cross_line = ''
    if macd_state and macd_state.get('cross'):
        cross_line = f"MACD {macd_state['cross']}！\n"

    div_line = ''
    if divergences:
        div_line = '⚠️ 背离信号: ' + ' / '.join(divergences) + '\n'

    ma_line = ' | '.join(price_pos)

    title = f"{trend_emoji} 黄金日K {last['date']} 【{trend}】"
    body = (
        f"收盘: {last['close']:.2f} USD  "
        f"({(last['close'] / prev['close'] - 1) * 100:+.2f}%)\n"
        f"\n均线: {ma_line}\n"
        f"均线排列: {ma_trend}\n"
        f"\nMACD: {last['macd']:.4f} / Signal: {last['signal_line']:.4f}\n"
        f"方向: {macd_state.get('direction', '-')} | 位置: {macd_state.get('position', '-')} | "
        f"柱状图: {macd_state.get('hist_trend', '-')}\n"
        f"{cross_line}"
        f"{div_line}"
    )
    send_notification(title, body)


# ================= 定时调度 =================
def _seconds_until_next_run():
    """计算距下一个 09:30 还有多少秒"""
    now = datetime.datetime.now(_CST_OFFSET)
    target = now.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)
    if now >= target:
        target += datetime.timedelta(days=1)
    return (target - now).total_seconds()


def _is_trading_day():
    """周末跳过（黄金现货周末休市）"""
    return datetime.datetime.now(_CST_OFFSET).weekday() < 5


# ================= 入口 =================
def test_run():
    """立即执行一次分析，用于测试验证"""
    print("🧪 测试模式：立即执行一次日K分析...\n")
    run_daily_analysis()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_run()
        sys.exit(0)

    while True:
        wait_secs = _seconds_until_next_run()
        next_run  = datetime.datetime.now(_CST_OFFSET) + datetime.timedelta(seconds=wait_secs)
        print(f"⏰ 下次执行时间: {next_run.strftime('%Y-%m-%d %H:%M:%S')} 北京时间  (等待 {wait_secs/3600:.1f} 小时)")
        time.sleep(wait_secs)

        if not _is_trading_day():
            print("⏸️  今日为周末，跳过分析...")
            continue

        try:
            run_daily_analysis()
        except Exception as e:
            print(f"❌ 分析任务异常: {e}")
