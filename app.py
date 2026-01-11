# ---------------------------
# app.py - Taiwan Stock Prediction System
# ---------------------------
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# ---------------------------
# 1️⃣ Sidebar: User Inputs
# ---------------------------
st.sidebar.title("Settings")

tickers_input = st.sidebar.text_input(
    "Enter Taiwan Stock Symbols (comma-separated, e.g., 2330.TW,2317.TW,2454.TW):",
    "2330.TW,2317.TW,2454.TW"
)

tickers = [t.strip() for t in tickers_input.split(",") if t.strip() != ""]

selected_ticker = st.sidebar.selectbox(
    "Select a stock to analyze:",
    tickers
)

lags = st.sidebar.slider("Number of Lag Days:", 1, 10, 5)
test_size = st.sidebar.slider("Test Set Size (days):", 30, 200, 50)

# ---------------------------
# 2️⃣ Load stock data
# ---------------------------
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, start='2020-01-01', end='2025-01-01')
    df = df[['Close']].reset_index()
    return df

# ---------------------------
# 3️⃣ Feature Engineering
# ---------------------------
def create_features(df, lags=5):
    df = df.copy()
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['Close'].shift(i)
    df['rolling_mean'] = df['Close'].shift(1).rolling(window=3).mean()
    df['rolling_std'] = df['Close'].shift(1).rolling(window=3).std()
    df = df.dropna()
    return df

# ---------------------------
# 4️⃣ Process selected stock
# ---------------------------
ticker = selected_ticker
st.header(f"Stock: {ticker}")

data = load_data(ticker)
if data.empty:
    st.warning("No data found for this stock.")
    st.stop()

data_feat = create_features(data, lags=lags)

X = data_feat.drop(['Close', 'Date'], axis=1)
y = data_feat['Close']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=test_size
)

# ---------------------------
# 5️⃣ Train models
# ---------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

try:
    arima_model = ARIMA(data['Close'][:-len(X_test)], order=(5, 1, 0))
    arima_result = arima_model.fit()
    pred_arima = arima_result.forecast(steps=len(X_test))
except:
    pred_arima = np.zeros(len(y_test))
    st.warning("ARIMA model failed.")

rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

xgb = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)

svr = SVR(kernel='rbf', C=100, epsilon=0.01)
svr.fit(X_train, y_train)
pred_svr = svr.predict(X_test)

# ---------------------------
# 6️⃣ Evaluation
# ---------------------------
models = {
    'Linear Regression': pred_lr,
    'ARIMA': pred_arima,
    'Random Forest': pred_rf,
    'XGBoost': pred_xgb,
    'SVR': pred_svr
}

performance = []
for name, pred in models.items():
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    performance.append({'Model': name, 'RMSE': rmse, 'MAE': mae})

perf_df = pd.DataFrame(performance)
best_model_name = perf_df.loc[perf_df['RMSE'].idxmin(), 'Model']

st.subheader(f"Best Model: {best_model_name}")
st.table(perf_df)

# ---------------------------
# 7️⃣ Visualization
# ---------------------------
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual', color='black')
plt.plot(models[best_model_name], label=f'Best Model ({best_model_name})', color='red')
plt.xlabel("Test Days")
plt.ylabel("Close Price")
plt.title(f"{ticker} - Best Model Prediction")
plt.legend()
st.pyplot(plt)

with st.expander("Show All Model Predictions"):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Actual', color='black')
    for name, pred in models.items():
        plt.plot(pred, label=name)
    plt.xlabel("Test Days")
    plt.ylabel("Close Price")
    plt.title(f"{ticker} - All Model Predictions")
    plt.legend()
    st.pyplot(plt)
