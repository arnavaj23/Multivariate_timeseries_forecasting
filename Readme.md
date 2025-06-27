📈 Time Series Forecasting with LSTM (Stock & Pollution Data)

This repository showcases multivariate and univariate time series forecasting using LSTM (Long Short-Term Memory) networks on real-world datasets:

    🏭 Air Pollution Prediction using Beijing PM2.5 dataset

    💼 Stock Price Forecasting using Microsoft stock prices from Yahoo Finance

🚀 What You'll Learn

✅ How to:

    Clean and preprocess real-world time series datasets

    Convert time series into windowed supervised learning format

    Build deep LSTM architectures for forecasting

    Train with both Mean Squared Error (MSE) and Huber Loss

    Evaluate performance using:

        R² Score

        MAE, MAPE, RMSE

    Implement recursive prediction for multi-step forecasting

    Visualize training/validation/test predictions vs actual values

📊 Datasets Used

    Air Pollution (PM2.5)
    Source: UCI / Kaggle

        Features: Temperature, Wind Speed, Direction, Dew Point, etc.

        Target: pollution (PM2.5 concentration)

    Microsoft Stock Prices
    Source: Yahoo Finance

        Features: Close price only

        Target: Future closing price

🧠 Model Architectures

    LSTM with:

        Multiple stacked LSTM layers (256 → 128 → 64)

        Dense layers for final regression output

    Window size: n=50 (for both stock and pollution datasets)

model = Sequential([
    LSTM(256, return_sequences=True),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

    Loss: Huber Loss or MSE

    Optimizer: Adam (lr=0.001)

📈 Evaluation Metrics

Each model is evaluated using:

    ✅ R² Score

    ✅ Mean Absolute Error (MAE)

    ✅ Root Mean Squared Error (RMSE)

    ✅ Mean Absolute Percentage Error (MAPE)

🔁 Recursive Forecasting

Implemented recursive loop to simulate real-world multi-step forecasting where each prediction is used as input for the next step.

for t in test_dates:
    pred = model.predict(last_window)
    last_window = update_with(pred)

📉 Visualizations

    Matplotlib plots for:

        Training/Validation/Test predictions

        Recursive predictions

        Error metrics over time

✍️ Author

Arnav
Time Series & Deep Learning Enthusiast
