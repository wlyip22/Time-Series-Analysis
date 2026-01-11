# ---------------------------
# app.py - Taiwan Stock Prediction + Fundamentals + Future Forecast + Historical EPS/Revenue
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
# Sidebar - User Inputs
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
# Load stock price data
# ---------------------------
@st.cache_data
def load_price_data(symbol):
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
# Load data
# ---------------------------
st.header(f"Stock: {ticker}")
price_data = load_price_data(ticker)

if price_data.empty:
    st.error("No price data found for this stock.")
else:
    # ---------------------------
    # Fundamental Analysis (latest snapshot)
    # ---------------------------
    st.subheader("Fundamental Analysis (Latest)")
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info

        fundamentals = {
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "EPS": info.get("trailingEPS", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A"),
            "Revenue": info.get("totalRevenue", "N/A"),
            "Profit Margin": info.get("profitMargins", "N/A"),
        }

        # Format numbers
        for k, v in fundamentals.items():
            if isinstance(v, (int, float)):
                if k in ["Market Cap", "Revenue"]:
                    fundamentals[k] = f"{v/1e9:.2f} B"
                elif k in ["Dividend Yield", "Profit Margin"]:
                    fundamentals[k] = f"{v*100:.2f}%"
                else:
                    fundamentals[k] = round(v,2)

        st.table(pd.DataFrame(fundamentals.items(), columns=["Metric", "Value"]))
    except:
        st.warning("Latest fundamental data not available for this stock.")

    # ---------------------------
    # Historical fundamentals (EPS and Revenue trends)
    # ---------------------------
    st.subheader("Historical Fundamentals (Annual)")

    try:
        fin = ticker_obj.financials.T  # transpose: columns = Net Income, Revenue, etc.
        shares_outstanding = info.get("sharesOutstanding", None)

        # EPS calculation: EPS = Net Income / Shares
        if shares_outstanding:
            fin['EPS'] = fin['Net Income'] / shares_outstanding
        else:
            fin['EPS'] = np.nan

        # Keep only EPS and Revenue
        hist_fund = fin[['EPS','Total Revenue']].rename(columns={'Total Revenue':'Revenue'})
        hist_fund = hist_fund.sort_index()  # chronological order

        st.line_chart(hist_fund)

    except:
        st.warning("Historical fundamentals not available.")

    # ---------------------------
    # Feature Engineering for ML
    # ---------------------------
    data_feat = create_features(price_data, lags, include_rolling=True)
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
    # ARIMA baseline
    # ---------------------------
    try:
        arima_model = ARIMA(price_data['Close'][:-len(X_test)], order=(5,1,0))
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

    # ---------------------------
    # Model performance
    # ---------------------------
    perf_df = pd.DataFrame(results)
    best_model_name = perf_df.loc[perf_df["RMSE"].idxmin(), "Model"]
    st.subheader(f"Best Model: {best_model_name}")
    st.table(perf_df)

    # ---------------------------
    # Future Forecast (ML models only)
    # ---------------------------
    future_preds = []
    if best_model_name in ml_models:
        best_model = ml_models[best_model_name]
        best_model.fit(X, y)  # Refit on full dataset

        current_batch = X.iloc[-1:].copy()
        for _ in range(forecast_days):
            pred_output = best_model.predict(current_batch)
            next_price = float(np.array(pred_output).flatten()[0])
            future_preds.append(next_price)

            # Update lag features
            new_row_data = current_batch.iloc[0].copy()
            for i in range(lags, 1, -1):
                new_row_data[f'lag_{i}'] = new_row_data[f'lag_{i-1}']
            new_row_data['lag_1'] = next_price

            if 'rolling_mean' in new_row_data:
                lookback = [new_row_data[f'lag_{i}'] for i in range(1, min(4,lags+1))]
                new_row_data['rolling_mean'] = np.mean(lookback)
                new_row_data['rolling_std'] = np.std(lookback)

            current_batch = pd.DataFrame([new_row_data])
    else:
        st.warning(f"Future forecast for {best_model_name} (ARIMA) not supported.")

    # ---------------------------
    # Visualization
    # ---------------------------
    plt.figure(figsize=(10,5))
    plt.plot(y_test.values, label="Actual", color="black")

    if best_model_name in ml_models:
        plt.plot(predictions[best_model_name], label="Backtest Prediction", color="red")
        future_x = np.arange(len(y_test), len(y_test)+forecast_days)
        plt.plot(future_x, future_preds, linestyle="--", label="Future Forecast")
    else:
        plt.plot(predictions.get(best_model_name, []), label="Backtest Prediction", color="red")

    plt.title(f"{ticker} - Backtesting & Future Forecast")
    plt.xlabel("Days")
    plt.ylabel("Close Price")
    plt.legend()
    st.pyplot(plt)
