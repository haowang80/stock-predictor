import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from darts import TimeSeries
from darts.models import ExponentialSmoothing, ARIMA, Prophet
from darts.metrics import mape
from datetime import datetime, timedelta

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')

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
            'Prophet': Prophet()
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
            error = mape(val, pred)
            print(f"{name} MAPE: {error:.2f}%")
    
    return train, val, predictions

def plot_predictions(train, val, predictions, title='NVDA Stock Price Prediction'):
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
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('../data/nvda_prediction.png')
    plt.show()

def main():
    # Define parameters
    ticker = 'NVDA'
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    forecast_horizon = 30  # Predict next 30 days
    
    print(f"Downloading {ticker} stock data from {start_date} to {end_date}...")
    data = download_stock_data(ticker, start_date, end_date)
    
    print(f"Creating time series for {ticker}...")
    series = create_time_series(data)
    
    print(f"Training models and making predictions for the next {forecast_horizon} days...")
    train, val, predictions = train_and_predict(series, forecast_horizon)
    
    print("Plotting predictions...")
    plot_predictions(train, val, predictions, title=f'{ticker} Stock Price Prediction')
    
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
    print(f"Model predictions for {forecast_horizon} days in the future:")
    for name, price in latest_predictions.items():
        print(f"  {name}: ${price:.2f}")
    print(f"Average prediction: ${avg_prediction:.2f}")
    print(f"Predicted direction: {prediction_direction} ({prediction_change:.2f}%)")
    
    return data, series, predictions

if __name__ == "__main__":
    main() 