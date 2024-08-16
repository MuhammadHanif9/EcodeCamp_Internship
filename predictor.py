import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from datetime import datetime

# Load your trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Alpha Vantage API configuration
api_key = '3BNGP5G55N6QWYQN'

def fetch_data(symbol, api_key, interval='1min', outputsize='compact'):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    
    time_series_key = f'Time Series ({interval})'
    time_series = data.get(time_series_key, {})

    df = pd.DataFrame.from_dict(time_series, orient='index')
    df = df.rename(columns={
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. volume': 'volume'
    })

    # Convert index to datetime and sort
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df=df.dropna()

    # Convert columns to appropriate data types
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    
    # Calculate Moving Average
    df['Moving_Avg'] = df['close'].rolling(window=3).mean()
    df = df.dropna()  # Drop NaN values resulting from rolling window
    
    return df

# Streamlit App
st.title('Stock Price Prediction')

symbol = st.text_input('Enter Stock Symbol (e.g., AAPL, MSFT)', 'AAPL')

if st.button('Predict'):
    st.write(f'Fetching data for {symbol}...')
    data = fetch_data(symbol, api_key)

    if not data.empty:
        st.write(f"Data fetched for {symbol}. Here's a preview:")
        st.write(data.tail())  # Show the last few rows of data

        # Prepare the features for prediction (last 3 time steps)
        last_3_rows = data.tail(3)
        features = last_3_rows[['open', 'high', 'low', 'close', 'volume', 'Moving_Avg']].values

        # Reshape to match the input shape expected by the model (1, 3, 6)
        features = features.reshape(1, 3, 6)

        # Make prediction
        predicted_price = model.predict(features)[0]  # Extracting the scalar value from the array

        st.write(f"Predicted next stock price for {symbol}: ${predicted_price:.2f}")
    else:
        st.write('Failed to fetch data. Please check the stock symbol or try again later.')
