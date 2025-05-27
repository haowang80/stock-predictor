import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import os

# 基本的分析和预测库
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error

# 确保数据目录存在
os.makedirs('../data', exist_ok=True)

def download_stock_data(ticker, start_date, end_date):
    """从Yahoo Finance下载股票数据"""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def simple_moving_average(data, window=20):
    """计算简单移动平均线"""
    return data.rolling(window=window).mean()

def exponential_moving_average(data, span=20):
    """计算指数移动平均线"""
    return data.ewm(span=span, adjust=False).mean()

def predict_stock(ticker, forecast_days=30, history_years=2, save_plot=True):
    """主函数：下载数据、训练模型并预测股票价格"""
    # 定义日期范围
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*history_years)).strftime('%Y-%m-%d')
    
    print(f"Downloading {ticker} stock data from {start_date} to {end_date}...")
    data = download_stock_data(ticker, start_date, end_date)
    
    if data.empty:
        print(f"No data found for ticker {ticker}. Please check if it's a valid stock symbol.")
        return None, None
    
    # 使用收盘价
    close_prices = data['Close']
    
    # 计算移动平均线
    print("Calculating moving averages...")
    ma20 = simple_moving_average(close_prices, window=20)
    ma50 = simple_moving_average(close_prices, window=50)
    ema20 = exponential_moving_average(close_prices, span=20)
    
    # 尝试使用ARIMA模型进行预测
    print("Training ARIMA model...")
    predictions = {}
    
    try:
        # 提取训练数据的最后值，作为基准
        current_price = float(close_prices.iloc[-1])
        
        # 使用ARIMA模型进行预测
        # 避免使用日期索引，直接使用数值
        arima_model = ARIMA(close_prices.values, order=(5,1,0))
        arima_fit = arima_model.fit()
        
        # 预测未来N天
        arima_forecast = arima_fit.forecast(steps=forecast_days)
        
        # 创建日期索引
        last_date = close_prices.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')
        
        # 调整预测数据长度以匹配日期
        if len(arima_forecast) > len(future_dates):
            arima_forecast = arima_forecast[:len(future_dates)]
        elif len(arima_forecast) < len(future_dates):
            future_dates = future_dates[:len(arima_forecast)]
        
        # 创建预测Series
        arima_series = pd.Series(arima_forecast, index=future_dates)
        predictions['ARIMA'] = arima_series
        
    except Exception as e:
        print(f"ARIMA model failed: {e}")
    
    # 尝试使用指数平滑进行预测
    print("Training Exponential Smoothing model...")
    try:
        # 避免使用日期索引，直接使用数值
        exp_model = ExponentialSmoothing(
            close_prices.values, 
            trend='add',
            seasonal=None
        )
        exp_fit = exp_model.fit()
        
        # 预测未来N天
        exp_forecast = exp_fit.forecast(forecast_days)
        
        # 创建日期索引
        last_date = close_prices.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')
        
        # 调整预测数据长度以匹配日期
        if len(exp_forecast) > len(future_dates):
            exp_forecast = exp_forecast[:len(future_dates)]
        elif len(exp_forecast) < len(future_dates):
            future_dates = future_dates[:len(exp_forecast)]
        
        # 创建预测Series
        exp_series = pd.Series(exp_forecast, index=future_dates)
        predictions['Exponential Smoothing'] = exp_series
        
    except Exception as e:
        print(f"Exponential Smoothing model failed: {e}")
    
    # 简单线性回归预测
    print("Calculating simple linear regression...")
    try:
        # 创建简单的时间序列特征
        x = np.arange(len(close_prices)).reshape(-1, 1)
        y = close_prices.values
        
        # 使用numpy的polyfit进行线性回归
        slope, intercept = np.polyfit(x.flatten(), y, 1)
        
        # 预测未来N天
        future_x = np.arange(len(close_prices), len(close_prices) + forecast_days)
        linear_forecast = slope * future_x + intercept
        
        # 创建日期索引
        last_date = close_prices.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')
        
        # 调整预测数据长度以匹配日期
        if len(linear_forecast) > len(future_dates):
            linear_forecast = linear_forecast[:len(future_dates)]
        elif len(linear_forecast) < len(future_dates):
            future_dates = future_dates[:len(linear_forecast)]
        
        # 创建预测Series
        linear_series = pd.Series(linear_forecast, index=future_dates)
        predictions['Linear Regression'] = linear_series
        
    except Exception as e:
        print(f"Linear regression failed: {e}")
    
    # 绘制结果
    print("Plotting predictions...")
    plt.figure(figsize=(12, 6))
    
    # 绘制历史数据和移动平均线
    plt.plot(close_prices.index, close_prices.values, label='Historical Data', color='black')
    plt.plot(ma20.index, ma20.values, label='20-day MA', color='blue', alpha=0.7)
    plt.plot(ma50.index, ma50.values, label='50-day MA', color='orange', alpha=0.7)
    plt.plot(ema20.index, ema20.values, label='20-day EMA', color='purple', alpha=0.7)
    
    # 绘制预测数据
    colors = ['red', 'green', 'cyan']
    for i, (name, pred) in enumerate(predictions.items()):
        plt.plot(pred.index, pred.values, label=f'{name} Prediction', color=colors[i % len(colors)], linestyle='--')
    
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    if save_plot:
        output_path = f'../data/{ticker}_prediction.png'
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    
    plt.show()
    
    # 提取最近的预测结果
    latest_predictions = {}
    for name, pred in predictions.items():
        # 获取最后一个预测值
        if len(pred) > 0:
            value = pred.iloc[-1]
            if not np.isnan(value):
                latest_predictions[name] = value
    
    # 获取当前价格
    current_price = float(close_prices.iloc[-1])
    
    # 仅从有效预测中计算平均预测价格
    if latest_predictions:
        avg_prediction = sum(latest_predictions.values()) / len(latest_predictions)
        
        # 确定预测方向（上涨或下跌）
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
    
    return data, predictions

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