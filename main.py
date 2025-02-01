import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# Function to fetch stock data from Yahoo Finance
def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Function to preprocess stock data
def preprocess_data(data):
    data = data[['Close']]  # Only consider closing prices
    data['Prediction'] = data[['Close']].shift(-30)  # Predicting the next 30 days

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['Close']])

    # Split data into features (X) and labels (y)
    X = data_scaled[:-30]
    y = data['Prediction'][:-30]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler, data

# Function to build the prediction model
def build_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)  # Train the model
    return model

# Function to plot predictions with improved style
def plot_predictions(y_test, predictions, data):
    plt.figure(figsize=(12, 7))

    # Plot real prices with dates on x-axis
    plt.plot(data.index[-len(y_test):], y_test, color='blue', label='Real Prices', linewidth=2)

    # Plot predicted prices
    plt.plot(data.index[-len(y_test):], predictions, color='red', label='Predicted Prices', linewidth=2, linestyle='--')

    # Customize labels and title
    plt.title('Stock Price Prediction', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price', fontsize=12)

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45)

    plt.legend(loc='upper left')
    plt.tight_layout()  # Adjust the layout to prevent overlap
    plt.show()

# Main function to run the application
def run_app():
    def on_predict():
        symbol = symbol_var.get()
        
        if not symbol:
            messagebox.showerror("Error", "Please select a stock symbol")
            return

        try:
            end_date = datetime.today().strftime('%Y-%m-%d')  # Today's date
            start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')  # Date one year ago

            stock_data = get_stock_data(symbol, start_date, end_date)

            X_train, X_test, y_train, y_test, scaler, data = preprocess_data(stock_data)

            model = build_model(X_train, y_train)

            predictions = model.predict(X_test)

            plot_predictions(y_test, predictions, data)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    # Create the GUI interface
    root = Tk()
    root.title("draculaTrade - Stock Price Predictor")
    root.geometry("500x300")
    root.config(bg="#2c3e50")  # Set background color to dark blue

    # Add a frame for better structure
    frame = Frame(root, padx=20, pady=20, bg="#2c3e50")
    frame.pack(padx=20, pady=20)

    # Heading Label
    Label(frame, text="Stock Price Predictor", font=("Arial", 18, "bold"), fg="white", bg="#2c3e50").grid(row=0, column=0, columnspan=2, pady=10)

    Label(frame, text="Select Stock Symbol:", font=("Arial", 12), fg="white", bg="#2c3e50").grid(row=1, column=0, pady=10)
    
    symbols = ["AAPL", "GOOGL", "AMZN", "MSFT", "TSLA", "META"]
    symbol_var = StringVar(value=symbols[0])
    
    symbol_menu = OptionMenu(frame, symbol_var, *symbols)
    symbol_menu.config(width=20)
    symbol_menu.grid(row=1, column=1, pady=10)

    # Predict Button with stylish design
    predict_button = Button(frame, text="Predict", font=("Arial", 12, "bold"), command=on_predict, bg='#3498db', fg='white', padx=10, pady=5)
    predict_button.grid(row=2, column=0, columnspan=2, pady=20)

    root.mainloop()

# Run the application
if __name__ == "__main__":
    run_app()
