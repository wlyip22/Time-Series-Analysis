# ---------------------------
# app.py - Taiwan Stock Prediction System (with Future Forecast)
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
# Sidebar
# ---------------------------
st.sidebar.title("Settings")

tickers_input = st.sidebar.text_input(
    "Enter Taiwan Stock Symbols (comma-separated):",
    "2330.TW,2317.TW,2454.TW"
)
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
ticker = st.sidebar.selectbox("Select a stock:", tickers)

lags = st.sidebar.slider("Number of Lag Days:", 1, 10, 5)
test_size = st.sidebar.slider("Test Set Size (days):", 30, 200, 50)
forecast_days = st.sidebar.slider("Forecast Future Days:", 1, 30, 5)

# ---------------------------
# Data loading
# ---------------------------
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, start="2020-01-01", end="2025-01-01")
    return df[['Close']].reset_index()

def create_features(df, lags=5, include_rolling=True):
    df = df.copy()

    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['Close'].shift(i)

    if include_rolling:
        df['rolling_mean'] = df['Close'].shift(1).rolling(window=3).mean()
        df['rolling_std'] = df['Close'].shift(1).rolling(window=3).std()

    return df.dropna()


# ---------------------------
# Load & prepare data
# ---------------------------
st.header(f"Stock: {ticker}")
data = load_data(ticker)
data_feat = create_features(data, lags)

X = data_feat.drop(['Date', 'Close'], axis=1)
y = data_feat['Close']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=test_size
)

# ---------------------------
# Train models (backtesting)
# ---------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    "SVR": SVR(kernel="rbf", C=100, epsilon=0.01)
}

predictions = {}
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    results.append({
        "Model": name,
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "MAE": mean_absolute_error(y_test, pred)
    })

perf_df = pd.DataFrame(results)
best_model_name = perf_df.loc[perf_df["RMSE"].idxmin(), "Model"]
best_model = models[best_model_name]

st.subheader(f"Best Model: {best_model_name}")
st.table(perf_df)

# ---------------------------
# Future Forecast (NEW)
# ---------------------------
data_lag_only = create_features(data, lags=lags, include_rolling=False)

X_full = data_lag_only.drop(['Date', 'Close'], axis=1)
y_full = data_lag_only['Close']

best_model.fit(X_full, y_full)

last_lags = X_full.iloc[-1].to_numpy(dtype=float).copy()
future_preds = []

for _ in range(forecast_days):
    next_price = float(best_model.predict(last_lags.reshape(1, -1))[0])
    future_preds.append(next_price)

    # shift lag features safely
    last_lags[1:] = last_lags[:-1]
    last_lags[0] = next_price

# ---------------------------
# Visualization
# ---------------------------
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual", color="black")
plt.plot(predictions[best_model_name], label="Backtest Prediction", color="red")

future_x = np.arange(len(y_test), len(y_test) + forecast_days)
plt.plot(future_x, future_preds, linestyle="--", label="Future Forecast")

plt.title(f"{ticker} - Backtesting & Future Forecast")
plt.xlabel("Days")
plt.ylabel("Close Price")
plt.legend()
st.pyplot(plt)
