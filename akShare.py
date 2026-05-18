import akshare as ak
import pandas as pd


def get_gold_price_akshare():
    # 伦敦金在新浪的代码通常是 HF_XAU
    # 如果 HF_XAU 不行，可以尝试 "XAU" 或 "伦敦金"，但 HF_XAU 最稳定
    symbol_name = "HF_XAU"

    try:
        print(f"🔄 正在获取 {symbol_name} 实时行情...")

        # 调用接口，必须传入 symbol 参数
        df = ak.futures_foreign_commodity_realtime(symbol=symbol_name)

        if df is not None and not df.empty:
            print(f"💰 [AkShare] 伦敦金 ({df}) 实时报价:")

            # 打印完整数据以便查看字段
            # 通常包含: symbol, price, open, high, low, change, change_percent, update_time
            current_price = df['price'].iloc[0]
            update_time = df['update_time'].iloc[0]
            change_pct = df['change_percent'].iloc[0]

            print("-" * 30)
            print(f"💰 [AkShare] 伦敦金 ({symbol_name}) 实时报价:")
            print(f"   当前价格: ${current_price}")
            print(f"   涨跌幅:   {change_pct}%")
            print(f"   更新时间: {update_time}")
            print("-" * 30)

            return current_price
        else:
            print(f"❌ 未获取到 {symbol_name} 的数据，返回为空。")
            return None

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        print("💡 提示：可能是 AkShare 版本过旧或新浪接口临时调整。")
        print("   建议执行: pip install --upgrade akshare")
        return None


if __name__ == "__main__":
    get_gold_price_akshare()