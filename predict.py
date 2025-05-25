import pandas as pd
import numpy as np
import joblib
import sys
import os

def calculate_technical_indicators(df):
    # 計算移動平均線
    for window in [5, 10, 20, 50]:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
        df[f'MA{window}_slope'] = df[f'MA{window}'].pct_change()
    
    # 計算布林帶
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    
    # 計算價格變化率
    for window in [1, 3, 5, 10]:
        df[f'Price_change_{window}d'] = df['Close'].pct_change(window)
        df[f'Volume_change_{window}d'] = df['Volume'].pct_change(window)
    
    # 計算價格波動率
    for window in [5, 10, 20]:
        df[f'Volatility_{window}d'] = df['Close'].rolling(window=window).std() / df['Close'].rolling(window=window).mean()
    
    # 計算價格動量
    for window in [5, 10, 20]:
        df[f'Momentum_{window}d'] = df['Close'] - df['Close'].shift(window)
    
    # 計算價格趨勢
    df['Trend_short'] = np.where(df['MA5'] > df['MA20'], 1, 0)
    df['Trend_medium'] = np.where(df['MA20'] > df['MA50'], 1, 0)
    
    # 計算 RSI 變化
    df['RSI_change'] = df['RSI'].pct_change()
    
    # 計算 MACD 變化
    df['MACD_change'] = df['MACD'].pct_change()
    df['MACD_signal_change'] = df['MACD_signal'].pct_change()
    
    # 計算價格與移動平均線的差距
    for window in [5, 10, 20, 50]:
        df[f'Price_MA{window}_diff'] = (df['Close'] - df[f'MA{window}']) / df[f'MA{window}']
    
    return df

def clean_data(X):
    # 處理無限值
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # 處理異常值
    for col in X.columns:
        # 計算四分位數
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # 定義異常值的界限
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # 將異常值替換為界限值
        X[col] = X[col].clip(lower_bound, upper_bound)
    
    # 填充空值
    X = X.ffill().bfill()
    
    return X

def predict_price(data):
    # 檢查模型文件是否存在
    if not os.path.exists('btc_price_predictor.joblib') or not os.path.exists('scaler.joblib'):
        raise FileNotFoundError("找不到模型文件，請先運行 train_model.py")

    # 載入模型和 scaler
    model = joblib.load('btc_price_predictor.joblib')
    scaler = joblib.load('scaler.joblib')
    
    # 標準化數據
    data_scaled = scaler.transform(data)
    
    # 預測
    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)
    
    return prediction, probability

if __name__ == "__main__":
    try:
        # 檢查數據文件是否存在
        if not os.path.exists("btc_rsi_macd.csv"):
            raise FileNotFoundError("找不到 btc_rsi_macd.csv 文件")

        # 讀取數據
        df = pd.read_csv("btc_rsi_macd.csv", header=None, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_signal', 'Target'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # 計算技術指標
        df = calculate_technical_indicators(df)
        
        # 準備特徵
        features = [col for col in df.columns if col not in ['Target']]
        
        # 檢查特徵是否存在
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"數據中缺少以下特徵: {missing_features}")
        
        X = df[features].iloc[-1:].values  # 只取最後一筆數據
        
        # 預測
        prediction, probability = predict_price(X)
        
        print(f"預測結果: {'上漲' if prediction[0] == 1 else '下跌'}")
        print(f"上漲機率: {probability[0][1]:.2%}")
        print(f"下跌機率: {probability[0][0]:.2%}")
    except Exception as e:
        print(f"錯誤: {str(e)}", file=sys.stderr)
        sys.exit(1)
