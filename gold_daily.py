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

RSI_PERIOD   = 14
RSI_OB       = 70             # RSI 超涨阈值
RSI_OS       = 30             # RSI 超跌阈值

KDJ_PERIOD   = 9
KDJ_SIGNAL   = 3
KDJ_OB       = 80             # KDJ K 超涨阈值
KDJ_OS       = 20             # KDJ K 超跌阈值

RUN_HOUR     = 10
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
    """计算 MACD、MA7/MA14/MA52、RSI、KDJ"""
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

    # RSI
    rsi_df = df.ta.rsi(length=RSI_PERIOD)
    df['rsi'] = rsi_df

    # KDJ
    kdj_df = df.ta.kdj(length=KDJ_PERIOD, signal=KDJ_SIGNAL)
    df = pd.concat([df, kdj_df], axis=1)
    df.rename(columns={
        f'K_{KDJ_PERIOD}_{KDJ_SIGNAL}': 'kdj_k',
        f'D_{KDJ_PERIOD}_{KDJ_SIGNAL}': 'kdj_d',
        f'J_{KDJ_PERIOD}_{KDJ_SIGNAL}': 'kdj_j',
    }, inplace=True)

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


def _rsi_state(df):
    """
    RSI 状态判断：
      > 70 超涨  |  50~70 偏多  |  30~50 偏空  |  < 30 超跌
    返回 dict: value / zone / cross_50（穿越50中轴）
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]
    rsi  = last['rsi']
    if pd.isna(rsi):
        return {}

    if rsi >= RSI_OB:
        zone = '超涨'
    elif rsi >= 50:
        zone = '偏多'
    elif rsi >= RSI_OS:
        zone = '偏空'
    else:
        zone = '超跌'

    cross_50 = None
    if not pd.isna(prev['rsi']):
        if prev['rsi'] < 50 <= rsi:
            cross_50 = '上穿50'
        elif prev['rsi'] >= 50 > rsi:
            cross_50 = '下穿50'

    return {'value': rsi, 'zone': zone, 'cross_50': cross_50}


def _kdj_state(df):
    """
    KDJ 状态判断：
      K > 80 超涨  |  K < 20 超跌  |  J > 100 极端超涨  |  J < 0 极端超跌
    返回 dict: k / d / j / zone / cross（金叉/死叉）
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]
    k, d, j = last.get('kdj_k'), last.get('kdj_d'), last.get('kdj_j')
    if pd.isna(k) or pd.isna(d):
        return {}

    if j > 100 or k > KDJ_OB:
        zone = '超涨'
    elif j < 0 or k < KDJ_OS:
        zone = '超跌'
    elif k > 50:
        zone = '偏多'
    else:
        zone = '偏空'

    cross = None
    pk, pd_ = prev.get('kdj_k'), prev.get('kdj_d')
    if not pd.isna(pk) and not pd.isna(pd_):
        if pk < pd_ and k > d:
            cross = '金叉'
        elif pk > pd_ and k < d:
            cross = '死叉'

    return {'k': k, 'd': d, 'j': j, 'zone': zone, 'cross': cross}


def _comprehensive_conclusion(ma_trend, macd_state, rsi_state, kdj_state, divergences):
    """
    积分制综合结论：
      趋势类指标（MA/MACD）决定主方向，
      RSI/KDJ 判断超涨超跌，背离提供反转预警。

    分值范围：满分 ±9，正为多，负为空。
    """
    score = 0
    details = []

    # ── 均线排列（±2）──
    if ma_trend == '多头排列':
        score += 2
        details.append(('多', 'MA多头排列'))
    elif ma_trend == '空头排列':
        score -= 2
        details.append(('空', 'MA空头排列'))

    # ── MACD（±3）──
    if macd_state:
        if macd_state.get('direction') == '多头':
            score += 1
            details.append(('多', 'MACD多头'))
        else:
            score -= 1
            details.append(('空', 'MACD空头'))

        if macd_state.get('position') == '水上':
            score += 1
            details.append(('多', 'MACD水上'))
        else:
            score -= 1
            details.append(('空', 'MACD水下'))

        if macd_state.get('cross') == '金叉':
            score += 1
            details.append(('多', 'MACD金叉'))
        elif macd_state.get('cross') == '死叉':
            score -= 1
            details.append(('空', 'MACD死叉'))

    # ── RSI（±2）──
    if rsi_state:
        zone = rsi_state.get('zone', '')
        if zone == '偏多':
            score += 1
            details.append(('多', f"RSI偏多({rsi_state['value']:.1f})"))
        elif zone == '偏空':
            score -= 1
            details.append(('空', f"RSI偏空({rsi_state['value']:.1f})"))
        elif zone == '超涨':
            score += 1            # 超涨仍偏多，但加预警
            details.append(('警', f"RSI超涨({rsi_state['value']:.1f})⚠️"))
        elif zone == '超跌':
            score -= 1
            details.append(('警', f"RSI超跌({rsi_state['value']:.1f})⚠️"))

        if rsi_state.get('cross_50') == '上穿50':
            score += 1
            details.append(('多', 'RSI上穿50'))
        elif rsi_state.get('cross_50') == '下穿50':
            score -= 1
            details.append(('空', 'RSI下穿50'))

    # ── KDJ（±2）──
    if kdj_state:
        zone = kdj_state.get('zone', '')
        if zone == '偏多':
            score += 1
            details.append(('多', f"KDJ偏多(K{kdj_state['k']:.0f})"))
        elif zone == '偏空':
            score -= 1
            details.append(('空', f"KDJ偏空(K{kdj_state['k']:.0f})"))
        elif zone == '超涨':
            score += 1
            details.append(('警', f"KDJ超涨(K{kdj_state['k']:.0f}/J{kdj_state['j']:.0f})⚠️"))
        elif zone == '超跌':
            score -= 1
            details.append(('警', f"KDJ超跌(K{kdj_state['k']:.0f}/J{kdj_state['j']:.0f})⚠️"))

        if kdj_state.get('cross') == '金叉':
            score += 1
            details.append(('多', 'KDJ金叉'))
        elif kdj_state.get('cross') == '死叉':
            score -= 1
            details.append(('空', 'KDJ死叉'))

    # ── 背离（±2，反转信号）──
    for d in divergences:
        if d == '底背离':
            score += 2
            details.append(('多', '底背离🌟'))
        elif d == '顶背离':
            score -= 2
            details.append(('空', '顶背离⚠️'))

    # ── 超涨超跌综合判断 ──
    ob_warn = (rsi_state.get('zone') == '超涨') or (kdj_state.get('zone') == '超涨')
    os_warn = (rsi_state.get('zone') == '超跌') or (kdj_state.get('zone') == '超跌')

    # ── 结论映射 ──
    if score >= 6:
        conclusion = '强烈看多'
        emoji = '🚀🚀'
    elif score >= 3:
        conclusion = '看多'
        emoji = '🚀'
    elif score >= 1:
        conclusion = '偏多'
        emoji = '📈'
    elif score <= -6:
        conclusion = '强烈看空'
        emoji = '💥💥'
    elif score <= -3:
        conclusion = '看空'
        emoji = '💥'
    elif score <= -1:
        conclusion = '偏空'
        emoji = '📉'
    else:
        conclusion = '震荡观望'
        emoji = '🔄'

    # 叠加超涨超跌修饰
    if ob_warn and score > 0:
        conclusion += '（超涨警惕回调）'
    elif os_warn and score < 0:
        conclusion += '（超跌关注反弹）'

    return {
        'score':      score,
        'conclusion': conclusion,
        'emoji':      emoji,
        'ob_warn':    ob_warn,
        'os_warn':    os_warn,
        'details':    details,
    }


# ================= 通知 =================
def _score_to_bar(score):
    """将积分转为直观的信号强度描述，满分 ±9"""
    strength_map = [
        (7,  '█████ 极强看多'),
        (4,  '████░ 强烈看多'),
        (2,  '███░░ 看多'),
        (1,  '██░░░ 偏多'),
        (-1, '░░░░░ 震荡观望'),
        (-2, '░░░██ 偏空'),
        (-4, '░░███ 看空'),
        (-7, '░████ 强烈看空'),
    ]
    for threshold, label in strength_map:
        if score >= threshold:
            return label
    return '█████ 极强看空'
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

    last        = df.iloc[-1]
    prev        = df.iloc[-2]
    ma_trend    = _ma_trend(df)
    price_pos   = _ma_price_position(df)
    macd_state  = _macd_state(df)
    rsi_state   = _rsi_state(df)
    kdj_state   = _kdj_state(df)
    divergences = _detect_divergence(df)
    conclusion  = _comprehensive_conclusion(ma_trend, macd_state, rsi_state, kdj_state, divergences)

    pct = (last['close'] / prev['close'] - 1) * 100

    # ── 打印报告 ────────────────────────────────────────
    print(f"   日期:    {last['date']}")
    print(f"   收盘价:  {last['close']:.2f} USD  "
          f"(开:{last['open']:.2f} 高:{last['high']:.2f} 低:{last['low']:.2f})")
    print(f"   前日收:  {prev['close']:.2f} USD  "
          f"涨跌: {last['close'] - prev['close']:+.2f} ({pct:+.2f}%)")

    # 均线
    print(f"\n── 均线指标 ──────────────────────────────")
    for p in MA_PERIODS:
        val = last[f'ma{p}']
        tag = '↑ 价格上方' if last['close'] > val else '↓ 价格下方'
        print(f"   MA{p:<3}: {val:.2f}  {tag}")
    print(f"   均线排列: {ma_trend}")

    # MACD
    print(f"\n── MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL}) ──────────────────────────")
    if macd_state:
        cross_tag = f"  ⚡{macd_state['cross']}!" if macd_state.get('cross') else ''
        print(f"   MACD:      {last['macd']:.4f}  Signal: {last['signal_line']:.4f}{cross_tag}")
        print(f"   柱状图:    {last['histogram']:.4f}  ({macd_state.get('hist_trend', '-')})")
        print(f"   方向/位置: {macd_state['direction']} / {macd_state['position']}")

    if divergences:
        for d in divergences:
            emoji = '⚠️' if d == '顶背离' else '🌟'
            print(f"   {emoji} 检测到【MACD {d}】信号！")

    # RSI
    print(f"\n── RSI({RSI_PERIOD}) ─────────────────────────────")
    if rsi_state:
        zone_emoji = {'超涨': '🔴', '偏多': '🟢', '偏空': '🟡', '超跌': '🟣'}.get(rsi_state['zone'], '')
        cross_tag  = f"  ⚡{rsi_state['cross_50']}" if rsi_state.get('cross_50') else ''
        print(f"   RSI: {rsi_state['value']:.2f}  {zone_emoji} {rsi_state['zone']}{cross_tag}")
        print(f"   参考: 超涨>{RSI_OB}  超跌<{RSI_OS}")

    # KDJ
    print(f"\n── KDJ({KDJ_PERIOD},{KDJ_SIGNAL},{KDJ_SIGNAL}) ─────────────────────────────")
    if kdj_state:
        zone_emoji = {'超涨': '🔴', '偏多': '🟢', '偏空': '🟡', '超跌': '🟣'}.get(kdj_state['zone'], '')
        cross_tag  = f"  ⚡{kdj_state['cross']}" if kdj_state.get('cross') else ''
        print(f"   K: {kdj_state['k']:.1f}  D: {kdj_state['d']:.1f}  J: {kdj_state['j']:.1f}"
              f"  {zone_emoji} {kdj_state['zone']}{cross_tag}")
        print(f"   参考: 超涨 K>{KDJ_OB} 或 J>100  超跌 K<{KDJ_OS} 或 J<0")

    # 综合结论
    print(f"\n── 综合结论（信号强度: {_score_to_bar(conclusion['score'])}）──────────")
    for side, desc in conclusion['details']:
        icon = '▲' if side == '多' else ('▽' if side == '空' else '◆')
        print(f"   {icon} {desc}")
    print(f"\n   {conclusion['emoji']} 【{conclusion['conclusion']}】")
    print('=' * 60)

    # ── 构建通知消息 ─────────────────────────────────────
    ma_line    = ' | '.join(price_pos)
    cross_line = f"MACD {macd_state['cross']}！\n" if macd_state and macd_state.get('cross') else ''
    div_line   = 'MACD背离: ' + ' / '.join(divergences) + '\n' if divergences else ''

    rsi_line = ''
    if rsi_state:
        rsi_line = f"RSI({RSI_PERIOD}): {rsi_state['value']:.1f} [{rsi_state['zone']}]\n"

    kdj_line = ''
    if kdj_state:
        kdj_line = (f"KDJ({KDJ_PERIOD}): K{kdj_state['k']:.1f} D{kdj_state['d']:.1f} "
                    f"J{kdj_state['j']:.1f} [{kdj_state['zone']}]\n")

    title = f"{conclusion['emoji']} 黄金日K {last['date']}【{conclusion['conclusion']}】"
    body  = (
        f"收盘: {last['close']:.2f} USD ({pct:+.2f}%)\n"
        f"\n均线: {ma_line}\n"
        f"排列: {ma_trend}\n"
        f"\nMACD: {last['macd']:.3f}/{last['signal_line']:.3f} "
        f"[{macd_state.get('direction','-')}/{macd_state.get('position','-')}]\n"
        f"{cross_line}"
        f"\nRSI: {rsi_line}"
        f"\nKDJ: {kdj_line}"
        f"{div_line}"
        f"\n信号强度: {_score_to_bar(conclusion['score'])}"
    )
    send_notification(title, body)


# ================= 定时调度 =================
def _seconds_until_next_run():
    """计算距下一个 10:30 还有多少秒"""
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
