import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from darts import TimeSeries
from darts.models import ExponentialSmoothing, ARIMA, Prophet, NaiveSeasonal
from darts.metrics import mape
from datetime import datetime, timedelta
import os

# Ensure data directory exists
os.makedirs('../data', exist_ok=True)

def download_stock_data(ticker, start_date, end_date):
    """Download stock data from Yahoo Finance"""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def create_time_series(data, column='Close'):
    """Create a Darts TimeSeries from the stock data"""
    # Extract only the specified column to avoid multi-dimensional issues
    df = pd.DataFrame(data[column])
    
    # Stock market data typically has a business day frequency
    # We set fill_missing_dates=True to handle any gaps in the data
    series = TimeSeries.from_dataframe(df, freq='B', fill_missing_dates=True)
    return series

def train_and_predict(series, forecast_horizon, models=None):
    """Train multiple models and make predictions"""
    if models is None:
        models = {
            'Exponential Smoothing': ExponentialSmoothing(),
            'ARIMA': ARIMA(),
            'Prophet': Prophet(),
            'Naive Seasonal': NaiveSeasonal(K=7)  # Assuming weekly seasonality
        }
    
    # Split the data into training and validation sets
    train, val = series.split_before(series.time_index[-forecast_horizon])
    
    predictions = {}
    for name, model in models.items():
        print(f"Training {name} model...")
        model.fit(train)
        pred = model.predict(forecast_horizon)
        predictions[name] = pred
        
        # Calculate MAPE for validation set
        if val is not None:
            try:
                error = mape(val, pred)
                print(f"{name} MAPE: {error:.2f}%")
            except:
                print(f"Could not calculate MAPE for {name}")
    
    return train, val, predictions

def plot_predictions(train, val, predictions, ticker, output_path=None, title=None):
    """Plot the training data, validation data, and predictions"""
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    train.plot(label='Training Data')
    
    # Plot validation data if available
    if val is not None:
        val.plot(label='Validation Data')
    
    # Plot predictions
    for name, pred in predictions.items():
        pred.plot(label=f'{name} Prediction')
    
    if title is None:
        title = f'{ticker} Stock Price Prediction'
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot if output path is provided
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    
    plt.show()

def predict_stock(ticker, forecast_days=30, history_years=2, save_plot=True):
    """Main function to download data, train models and predict stock prices"""
    # Define date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*history_years)).strftime('%Y-%m-%d')
    
    print(f"Downloading {ticker} stock data from {start_date} to {end_date}...")
    data = download_stock_data(ticker, start_date, end_date)
    
    if data.empty:
        print(f"No data found for ticker {ticker}. Please check if it's a valid stock symbol.")
        return None, None, None
    
    print(f"Creating time series for {ticker}...")
    series = create_time_series(data)
    
    print(f"Training models and making predictions for the next {forecast_days} days...")
    train, val, predictions = train_and_predict(series, forecast_days)
    
    output_path = None
    if save_plot:
        output_path = f'../data/{ticker}_prediction.png'
    
    print("Plotting predictions...")
    plot_predictions(train, val, predictions, ticker, output_path)
    
    # Extract the most recent prediction for each model
    latest_predictions = {}
    for name, pred in predictions.items():
        value = pred.values()[-1][0]
        # Skip NaN values
        if not np.isnan(value):
            latest_predictions[name] = value
    
    # Get the current price
    current_price = float(data['Close'].iloc[-1].item())
    
    # Calculate the average predicted price only from valid predictions
    if latest_predictions:
        avg_prediction = sum(latest_predictions.values()) / len(latest_predictions)
        
        # Determine if the stock is predicted to go up or down
        prediction_direction = "UP" if avg_prediction > current_price else "DOWN"
        prediction_change = ((avg_prediction - current_price) / current_price) * 100
    else:
        avg_prediction = np.nan
        prediction_direction = "UNKNOWN"
        prediction_change = np.nan
    
    print(f"\nCurrent {ticker} price: ${current_price:.2f}")
    print(f"Model predictions for {forecast_days} days in the future:")
    for name, price in latest_predictions.items():
        print(f"  {name}: ${price:.2f}")
    print(f"Average prediction: ${avg_prediction:.2f}")
    print(f"Predicted direction: {prediction_direction} ({prediction_change:.2f}%)")
    
    return data, series, predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict stock prices using time series forecasting models')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., NVDA, AAPL, MSFT)')
    parser.add_argument('--days', type=int, default=30, help='Number of days to forecast (default: 30)')
    parser.add_argument('--years', type=int, default=2, help='Years of historical data to use (default: 2)')
    parser.add_argument('--no-plot', action='store_true', help='Do not save the plot')
    
    args = parser.parse_args()
    
    predict_stock(
        ticker=args.ticker,
        forecast_days=args.days,
        history_years=args.years,
        save_plot=not args.no_plot
    ) 