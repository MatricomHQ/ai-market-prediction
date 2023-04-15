# LSTM Model for SPY Prices Prediction 
This Python script uses LSTM (Long Short-Term Memory) model to predict the direction of prices for SPY (SPDR S&P 500 ETF Trust) stocks. 
It loads the time series dataset from a CSV file, 
preprocesses it using Scikit-learn's MinMaxScaler, 
trains the LSTM model, evaluates its performance, and calculates the percentage of correct direction decisions. 

## Dependencies 
* Pandas 
* Numpy 
* Tensorflow 
* Scikit-learn 
* Datetime 
* Logging 

## Usage
```
python python lstm_model.py
```
## Data
The CSV file "finals/data/SPY_2023-04-14_time_series.csv" 
contains the SPY stock prices time series data. 
The columns in the CSV file are:

```
timestamp
spy_price
volatility
expiration
realized_volatility
implied_volatility
```

## License
This project is licensed under the terms of the MIT license.