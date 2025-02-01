import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tkinter import *
from tkinter import messagebox
from datetime import datetime, timedelta

# 1. گرفتن داده‌های سهام از Yahoo Finance
def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# 2. پیش‌پردازش داده‌ها
def preprocess_data(data):
    data = data[['Close']]  # فقط قیمت بسته‌شدن رو می‌گیریم
    data['Prediction'] = data[['Close']].shift(-30)  # پیش‌بینی برای 30 روز آینده

    # مقیاس‌گذاری داده‌ها (برای بهتر شدن پیش‌بینی)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['Close']])

    # تقسیم داده‌ها به ویژگی‌ها (X) و برچسب‌ها (y)
    X = data_scaled[:-30]  # از داده‌های قیمت به‌جز 30 روز آخر استفاده می‌کنیم
    y = data['Prediction'][:-30]  # پیش‌بینی قیمت 30 روز آینده

    # تقسیم داده‌ها به بخش‌های آموزشی و آزمایشی
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler

# 3. ساخت مدل پیش‌بینی
def build_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)  # آموزش مدل با داده‌های آموزشی
    return model

# 4. رسم گراف پیش‌بینی‌ها
def plot_predictions(X_test, y_test, predictions):
    plt.figure(figsize=(10,6))
    plt.plot(y_test, color='blue', label='Real Prices')  # قیمت‌های واقعی
    plt.plot(predictions, color='red', label='Predicted Prices')  # قیمت‌های پیش‌بینی‌شده
    plt.title('Stock Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# 5. عملکرد اصلی برنامه
def run_app():
    def on_predict():
        # گرفتن ورودی‌های کاربر
        symbol = symbol_var.get()
        
        if not symbol:
            messagebox.showerror("Error", "Please select a stock symbol")
            return

        try:
            # تاریخ امروز و تاریخ یک سال قبل
            end_date = datetime.today().strftime('%Y-%m-%d')  # تاریخ امروز
            start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')  # تاریخ یک سال قبل

            # گرفتن داده‌ها از Yahoo Finance
            stock_data = get_stock_data(symbol, start_date, end_date)

            # پیش‌پردازش داده‌ها
            X_train, X_test, y_train, y_test, scaler = preprocess_data(stock_data)

            # ساخت و آموزش مدل
            model = build_model(X_train, y_train)

            # پیش‌بینی با استفاده از مدل
            predictions = model.predict(X_test)

            # نمایش گراف پیش‌بینی‌ها
            plot_predictions(X_test, y_test, predictions)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    # طراحی رابط کاربری
    root = Tk()
    root.title("draculaTrade - Stock Price Predictor")
    
    Label(root, text="Select Stock Symbol:").grid(row=0, column=0, pady=10)
    
    # نمادهای سهام موجود برای انتخاب
    symbols = ["AAPL", "GOOGL", "AMZN", "MSFT", "TSLA", "FB"]
    
    # ایجاد یک متغیر برای نگهداری نماد انتخابی
    symbol_var = StringVar(value=symbols[0])  # مقدار پیش‌فرض نماد اول است
    
    # ایجاد منوی کشویی برای انتخاب نماد
    symbol_menu = OptionMenu(root, symbol_var, *symbols)
    symbol_menu.grid(row=0, column=1, pady=10)

    predict_button = Button(root, text="Predict", command=on_predict)
    predict_button.grid(row=1, column=0, columnspan=2, pady=20)
    
    root.mainloop()

# اجرای برنامه
run_app()
