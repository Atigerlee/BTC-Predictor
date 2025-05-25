# 比特幣價格預測模型

這是一個使用機器學習預測比特幣價格走勢的專案。該模型使用 XGBoost 算法，結合多個技術指標來預測比特幣的價格變動。

## 功能特點

- 自動獲取比特幣歷史數據
- 計算多個技術指標（MA、RSI、MACD、布林帶等）
- 使用 XGBoost 進行價格預測
- 提供模型評估指標和視覺化結果
- 支持模型保存和加載

## 安裝步驟

1. 克隆專案：
```bash
git clone [您的專案URL]
cd Price_Predict
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 獲取數據：
```bash
python getData.py
```

2. 訓練模型：
```bash
python train.py
```

3. 使用模型進行預測：
```bash
python predict.py
```

## 專案結構

- `getData.py`: 獲取比特幣數據並計算技術指標
- `train.py`: 訓練 XGBoost 模型
- `predict.py`: 使用訓練好的模型進行預測
- `requirements.txt`: 專案依賴包列表
- `Data/`: 存放數據文件的目錄

## 模型特點

- 使用多個技術指標作為特徵
- 採用網格搜索進行參數優化
- 使用交叉驗證確保模型穩定性
- 提供特徵重要性分析
- 生成 ROC 曲線等評估圖表

## 注意事項

- 模型預測結果僅供參考，不構成投資建議
- 請確保有足夠的磁盤空間存儲數據文件
- 建議使用 Python 3.8 或更高版本

## 授權

[請在此處添加您的授權信息] 