# ---------------------------
# app.py - Taiwan Stock Prediction System (Safe ML + Future Forecast)
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

def create_features(df, lags, include_rolling=True):
    df = df.copy()
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['Close'].shift(i)
    if include_rolling:
        df['rolling_mean'] = df['Close'].shift(1).rolling(3).mean()
        df['rolling_std'] = df['Close'].shift(1).rolling(3).std()
    return df.dropna()

# ---------------------------
# Load & prepare data
# ---------------------------
st.header(f"Stock: {ticker}")
data = load_data(ticker)

if data.empty:
    st.error("No data found for this stock.")
else:
    # Features for ML backtesting (includes rolling)
    data_feat = create_features(data, lags, include_rolling=True)
    X = data_feat.drop(['Date', 'Close'], axis=1)
    y = data_feat['Close']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=test_size
    )

    # ---------------------------
    # Train ML models
    # ---------------------------
    ml_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        "SVR": SVR(kernel="rbf", C=100, epsilon=0.01)
    }

    predictions = {}
    results = []

    for name, model in ml_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        results.append({
            "Model": name,
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "MAE": mean_absolute_error(y_test, pred)
        })

    # ---------------------------
    # ARIMA baseline (optional)
    # ---------------------------
    try:
        arima_model = ARIMA(data['Close'][:-len(X_test)], order=(5, 1, 0))
        arima_result = arima_model.fit()
        pred_arima = arima_result.forecast(steps=len(X_test))
        predictions["ARIMA"] = pred_arima
        results.append({
            "Model": "ARIMA",
            "RMSE": np.sqrt(mean_squared_error(y_test, pred_arima)),
            "MAE": mean_absolute_error(y_test, pred_arima)
        })
    except:
        st.warning("ARIMA failed or not enough data.")

    perf_df = pd.DataFrame(results)
    best_model_name = perf_df.loc[perf_df["RMSE"].idxmin(), "Model"]
    st.subheader(f"Best Model: {best_model_name}")
    st.table(perf_df)

    # ---------------------------
    # Future Forecast (Corrected & Robust Version)
    # ---------------------------
    future_preds = []

    if best_model_name in ml_models:
        best_model = ml_models[best_model_name]
        
        # 1. Refit on all available data using the SAME feature set as training
        best_model.fit(X, y)

        # 2. Start with the very last available row from your data
        # We convert it to a DataFrame to keep column names
        current_batch = X.iloc[-1:].copy()

        # 3. Recursive future prediction
        for _ in range(forecast_days):
            # Ensure the input is a DataFrame with correct feature names
            # This prevents the "Feature name mismatch" or "TypeError"
            pred_value = best_model.predict(current_batch)[0]
            next_price = float(pred_value)
            future_preds.append(next_price)

            # --- Update Features for the Next Day ---
            new_row = current_batch.iloc[0].copy()
            
            # Update Lags: Shift values (e.g., lag_1 becomes the new next_price)
            # lag_5 = lag_4, lag_4 = lag_3...
            for i in range(lags, 1, -1):
                new_row[f'lag_{i}'] = new_row[f'lag_{i-1}']
            new_row['lag_1'] = next_price
            
            # Update Rolling Statistics (Important if include_rolling=True)
            if 'rolling_mean' in new_row:
                # We calculate mean of the most recent 'lags'
                recent_values = [new_row[f'lag_{i}'] for i in range(1, 4)] # 3-day window
                new_row['rolling_mean'] = np.mean(recent_values)
                new_row['rolling_std'] = np.std(recent_values)
            
            # Prepare for next iteration
            current_batch = pd.DataFrame([new_row])
    else:
        st.warning(f"Future forecast for {best_model_name} is skipped.")

    # ---------------------------
    # Visualization
    # ---------------------------
    plt.figure(figsize=(10,5))
    plt.plot(y_test.values, label="Actual", color="black")

    if best_model_name in ml_models:
        plt.plot(predictions[best_model_name], label="Backtest Prediction", color="red")
        future_x = np.arange(len(y_test), len(y_test) + forecast_days)
        plt.plot(future_x, future_preds, linestyle="--", label="Future Forecast")
    else:
        plt.plot(predictions.get(best_model_name, []), label="Backtest Prediction", color="red")

    plt.title(f"{ticker} - Backtesting & Future Forecast")
    plt.xlabel("Days")
    plt.ylabel("Close Price")
    plt.legend()
    st.pyplot(plt)
