# NVIDIA Stock Predictor

A simple stock price prediction system for NVIDIA (NVDA) built with the [Darts](https://github.com/unit8co/darts) time series forecasting library.

## Overview

This project uses historical NVIDIA stock data to predict future price movements. It implements multiple forecasting models:

- Exponential Smoothing
- ARIMA (AutoRegressive Integrated Moving Average)
- Prophet (Facebook's time series forecasting tool)

The system evaluates each model's performance using the Mean Absolute Percentage Error (MAPE) metric and provides a combined prediction based on all models.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/stock-predictor.git
cd stock-predictor
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Note: Prophet might require additional dependencies. If you encounter any issues, please refer to the [Prophet installation instructions](https://facebook.github.io/prophet/docs/installation.html).

## Usage

### NVIDIA Stock Prediction

Run the NVIDIA-specific script to download NVIDIA stock data, train the models, make predictions, and visualize the results:

```bash
cd src
python nvda_predictor.py
```

### Predicting Any Stock

For more flexibility, you can use either:

#### Using the Shell Script (Recommended)

```bash
# Run with default settings (NVDA, 30 days, 2 years history)
./run_predictor.sh

# Specify a different stock
./run_predictor.sh -t AAPL

# Customize days to forecast and years of history
./run_predictor.sh -t MSFT -d 60 -y 3
```

#### Using the Python Script Directly

```bash
cd src
python predict_stock.py NVDA
```

Additional options:
```bash
# Predict Apple stock for the next 60 days using 3 years of historical data
python predict_stock.py AAPL --days 60 --years 3

# Predict Microsoft stock without saving the plot
python predict_stock.py MSFT --no-plot
```

Both scripts will:
1. Download historical stock data
2. Train multiple forecasting models
3. Predict stock prices for the specified time period
4. Generate a plot of historical prices and predictions
5. Save the plot to `data/[TICKER]_prediction.png`
6. Print a summary of the predictions and the projected price direction

## Customization

You can modify the script to:

- Change the target stock by updating the `ticker` variable
- Adjust the prediction horizon by changing the `forecast_horizon` variable
- Add or remove forecasting models in the `train_and_predict` function
- Modify the date range for historical data with the `start_date` and `end_date` variables

## Disclaimer

This tool is for educational purposes only. Stock price predictions are inherently uncertain, and no forecasting method can guarantee accurate results. Always conduct your own research before making investment decisions. 