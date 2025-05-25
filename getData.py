import yfinance as yf
import pandas as pd
import numpy as np

def calculate_indicators(df):
    # 移動平均線
    df['ma5'] = df['Close'].rolling(window=5).mean()
    df['ma10'] = df['Close'].rolling(window=10).mean()
    df['ma20'] = df['Close'].rolling(window=20).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # 布林帶
    df['bb_mid'] = df['Close'].rolling(window=20).mean()
    df['bb_std'] = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

    # 成交量變化
    df['volume_change'] = df['Volume'].pct_change()

    # 技術指標滯後值與差值
    indicators = ['ma5', 'ma10', 'ma20', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower']
    for ind in indicators:
        for lag in [1, 2, 3]:
            df[f'{ind}_t-{lag}'] = df[ind].shift(lag)
            df[f'{ind}_diff_t-{lag}'] = df[ind] - df[ind].shift(lag)

    return df

def add_target(df):
    # 隔日收盤價與今日比較決定漲跌（1：漲，0：跌）
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df

def get_data():
    df = yf.download("BTC-USD", start="2020-01-01", interval="1d")
    df = calculate_indicators(df)
    df = add_target(df)
    df.dropna(inplace=True)  # 移除有NA的行
    df.to_csv('btc_features.csv')
    print("📁 資料已儲存至 btc_features.csv，共有欄位:", df.shape[1])

if __name__ == "__main__":
    get_data()
