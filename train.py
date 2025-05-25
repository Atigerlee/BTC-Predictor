import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


df = pd.read_csv('btc_features.csv')

# show the columns
print("columns：", df.columns.tolist())
print("\nshape：", df.shape)

# delete useless columns
columns_to_drop = ['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume']
existing_columns = [col for col in columns_to_drop if col in df.columns]
if existing_columns:
    df = df.drop(columns=existing_columns)
print("\ndrop columns：", df.shape)

# fix price data
if 'Price' in df.columns:
    df['Price'] = df['Price'].astype(str).str.replace('$', '').str.replace(',', '')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# convert other features to numeric
for column in df.columns:
    if column not in ['target', 'Price']:
        df[column] = pd.to_numeric(df[column], errors='coerce')

# show the data types
print("\ndata types：")
print(df.dtypes)

# check NaN values
print("\nNaN values：")
print(df.isna().sum())

# delete NaN values in feature columns, keep target column
feature_columns = [col for col in df.columns if col != 'target']
df = df.dropna(subset=feature_columns)
print("\ndrop NaN values in feature columns：", df.shape)

# check the values of target column
print("\ntarget column values：")
print(df['target'].unique())

# if target column is all NaN, recalculate the target
if df['target'].isna().all():
    print("\nrecalculate the target...")
    # use the change of Close price to calculate the target
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    # delete the last row's NaN value
    df = df.dropna(subset=['target'])

print("\nprocessed data shape：", df.shape)
print("target column values distribution：")
print(df['target'].value_counts())

if df.shape[0] == 0:
    raise ValueError("data is empty, check the data")

# add technical indicators
def add_technical_indicators(df):
    # moving average
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['std20'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (df['std20'] * 2)
    df['Lower'] = df['MA20'] - (df['std20'] * 2)
    
    # price change rate
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
    
    # volatility
    df['Volatility'] = df['Price_Change'].rolling(window=20).std()
    
    return df

# add 
df = add_technical_indicators(df)

# handle NaN values
feature_columns = [col for col in df.columns if col != 'target']
df = df.dropna(subset=feature_columns)

# target
if df['target'].isna().all():
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna(subset=['target'])

# standardization
scaler = StandardScaler()
X = df.drop(columns=['Close', 'target'])
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# split data 20% for test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['target'], test_size=0.2, shuffle=False)

# define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

# GridSearchCV find the best parameters
model = XGBClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# use the best parameters
best_model = grid_search.best_estimator_
print("\nbest parameters：", grid_search.best_params_)

# cross validation
cv_scores = cross_val_score(best_model, X_scaled, df['target'], cv=5)
print("\ncross validation scores：", cv_scores)
print("average cross validation scores：", cv_scores.mean())

# predict
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# evaluate
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nconfusion matrix：")
print(confusion_matrix(y_test, y_pred))
print("\nclassification report：")
print(classification_report(y_test, y_pred))

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()

# feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# save model and scaler
joblib.dump(best_model, 'btc_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\nmodel and scaler saved")
