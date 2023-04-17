import pandas as pd
import requests
import json
from datetime import datetime, time as dtime
import pandas as pd
import io
import time
import concurrent.futures
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

from rich.console import Console
spy_current_price = 0

scaler = MinMaxScaler()
model = Sequential()

api_type = "sandbox" # "api"
import csv


def backtest(predictions, actual_prices, buy_sell_directions):
    transactions = []
    current_position = 0

    for i in range(len(buy_sell_directions)):
        if buy_sell_directions[i] == "buy" and current_position <= 0:
            if current_position < 0:
                transactions.append(("buy_to_close", actual_prices[i], i))
                current_position = 0
            current_position += 100
            transactions.append(("buy_to_open", actual_prices[i], i))
        elif buy_sell_directions[i] == "sell" and current_position >= 0:
            if current_position > 0:
                transactions.append(("sell_to_close", actual_prices[i], i))
                current_position = 0
            current_position -= 100
            transactions.append(("sell_to_open", actual_prices[i], i))

    # Close the final position
    if current_position > 0:
        transactions.append(("sell_to_close", actual_prices[-1], len(actual_prices) - 1))
    elif current_position < 0:
        transactions.append(("buy_to_close", actual_prices[-1], len(actual_prices) - 1))

    return transactions

# Load data
file_path = "finals/data/SPY_2023-04-14_time_series.csv"
csv_file = pd.read_csv(file_path, parse_dates=["timestamp"])
csv_file["expiration"] = csv_file["expiration"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").timestamp())
csv_file.set_index("timestamp", inplace=True)

# Preprocessing

scaled_data = scaler.fit_transform(csv_file.values)
X, y = scaled_data[:, 1:], scaled_data[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Reshape data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

console = Console()

console.print(f"[bold]X_test shape:[/bold] {X_test.shape}")
console.print(f"[bold]y_test shape:[/bold] {y_test.shape}")

# Create and compile LSTM model with Adagrad optimizer

model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))

model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.005), loss='mean_squared_error')

# Train the model ## UPDATE TO 24
model.fit(X_train, y_train, epochs=1, batch_size=48, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
console.print(f"[bold]Loss:[/bold] ${loss:.2f}", style="green")

# Make predictions
predictions = model.predict(X_test)

console.print(f"predictions shape: [bold]{predictions.shape}[/bold]")

# Inverse transform the predictions
predictions = scaler.inverse_transform(np.hstack([predictions, X_test.reshape(X_test.shape[0], -1)]))[:, 0]

# Compare actual and predicted values
_, _, _, _, test_indices, _ = train_test_split(X, y, csv_file.index, test_size=0.2, random_state=44)

actual_vs_predicted = pd.DataFrame({"Actual": csv_file.loc[test_indices, "spy_price"].values[:predictions.shape[0]], "Predicted": predictions})

# Determine next direction (up or down)
next_direction = ["buy" if p > a else "sell" for a, p in zip(actual_vs_predicted["Actual"], actual_vs_predicted["Predicted"])]

# Convert actual prices to numpy array
actual_prices = actual_vs_predicted["Actual"].values

transactions = backtest(predictions, actual_prices, next_direction)

# Print transaction details
initial_bank = 100000
available_funds = initial_bank
open_positions = []
trade_qty = 200

console.rule("Transaction Details")
for transaction in transactions:
    console.print(f"{transaction[0]} Qty {trade_qty} at: [bold]${transaction[1]:.2f}[/bold] - (Pos  {actual_vs_predicted.index[transaction[2]]})")
    if transaction[0] == "buy_to_open":
        cost = trade_qty * transaction[1]
        available_funds -= cost
        open_positions.append(("buy", transaction[1]))
    elif transaction[0] == "sell_to_open":
        revenue = trade_qty * transaction[1]
        available_funds += revenue
        open_positions.append(("sell", transaction[1]))
    elif transaction[0] == "sell_to_close":
        revenue = trade_qty * transaction[1]
        available_funds += revenue
        open_position = open_positions.pop()
        pnl = revenue - trade_qty * open_position[1]
        console.print(f"Closed transaction P&L: [bold]${pnl:.2f}[/bold]")
    elif transaction[0] == "buy_to_close":
        cost = trade_qty * transaction[1]
        available_funds -= cost
        open_position = open_positions.pop()
        pnl = trade_qty * open_position[1] - cost
        console.print(f"Closed transaction P&L: [bold]${pnl:.2f}[/bold]")

console.rule("Final Results")

console.print(f"Available funds: [bold]${available_funds:.2f}[/bold]")

total_pnl = available_funds - initial_bank
console.print(f"Total Profit and Loss: [bold]${total_pnl:.2f}[/bold]")

# Determine if the model got each direction right or wrong
got_direction_right = []
for i in range(1, len(next_direction)):
    if next_direction[i] == next_direction[i-1]:
        got_direction_right.append("right")
    else:
        got_direction_right.append("wrong")

# Calculate percentage of right decisions
right_decisions = got_direction_right.count("right")
total_decisions = len(got_direction_right)
percentage_right = (right_decisions / total_decisions) * 100

console.print(f"Percentage of right decisions: [bold]{percentage_right:.2f}%[/bold]")


import requests
import time


import yfinance as yf

def get_vix_value():
    vix_ticker = yf.Ticker("^VIX")
    vix_data = vix_ticker.history(period="1d")
    vix_value = vix_data["Close"].iloc[-1]
    vix_value = round(vix_value, 2)
    print(vix_value)
    return vix_value

import requests



#### REAL-TIME DATA COLLECTION CODE:
def process_expiration(expiration):
    global vix_current_price
    global spy_current_price
    chains_params = {
        'symbol': symbol,
        'expiration': expiration
    }
    put_call_ratio = 0
    put_notional = 0
    call_notional = 0
    try:
        chains_response = requests.get(chains_endpoint, headers=headers, params=chains_params)
        chains_data = json.loads(chains_response.text)
        puts = [option for option in chains_data['options']['option'] if option['option_type'] == 'put']
        calls = [option for option in chains_data['options']['option'] if option['option_type'] == 'call']
        put_notional = sum([put['bid'] * put['volume'] * put['contract_size'] for put in puts])
        call_notional = sum([call['bid'] * call['volume'] * call['contract_size'] for call in calls])
        put_call_ratio = put_notional / call_notional
    
    except:
        print("Error with API - Market probably closed.")
    
    quote_params = {'symbols': symbol + ',' + vix_symbol}
    quote_response = requests.get(quote_endpoint, headers=headers, params=quote_params)
    quote_data = json.loads(quote_response.text)
    spy_current_price = quote_data['quotes']['quote'][0]['last']
    #quote_data['quotes']['quote'][1]['last']
    #vix_current_price = quote_data['quotes']['quote'][1]['last']
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    
    new_row = {'timestamp': timestamp,
               'spy_price': spy_current_price,
               'vix': vix_current_price,
               'expiration': expiration,
               'pcr': put_call_ratio,
               'put_notional': put_notional,
               'call_notional': call_notional}
    
    return new_row



columns = ['timestamp', 'spy_price', 'vix', 'expiration', 'pcr', 'put_notional', 'call_notional']
expirations_endpoint = f'https://{api_type}.tradier.com/v1/markets/options/expirations'
chains_endpoint = f'https://{api_type}.tradier.com/v1/markets/options/chains'
quote_endpoint = f'https://{api_type}.tradier.com/v1/markets/quotes'
symbol = 'SPY'
vix_symbol = 'VIX'
expirations_params = {
    'symbol': symbol,
    'includeAllRoots': 'true',
    'strikes': 'false'
}

headers = {
    'Authorization': 'Bearer lHV9xbhDC4cW02rr7WqULDQGsWbG',
    'Accept': 'application/json'
}

def predict_next(new_data):
    # Preprocessing the new data
    scaled_new_data = scaler.transform(new_data.values)
    X_new = scaled_new_data[:, 1:]
    
    # Reshape data for LSTM
    X_new = np.reshape(X_new, (X_new.shape[0], 1, X_new.shape[1]))
    
    # Make predictions
    new_predictions = model.predict(X_new)
    
    # Inverse transform the predictions
    new_predictions = scaler.inverse_transform(np.hstack([new_predictions, X_new.reshape(X_new.shape[0], -1)]))[:, 0]
  
    # Determine next direction (up or down)
    actual_price = new_data.iloc[-1]["spy_price"]
    predicted_price = new_predictions[-1]
    print("Prediction--- Spy current price:", actual_price,"Predicted price:", predicted_price)
    if predicted_price > actual_price:
        return "buy"
    else:
        return "sell"

#### Tradier integration...

import requests
import time

API_KEY = 'lHV9xbhDC4cW02rr7WqULDQGsWbG'
ACCOUNT_NUMBER = 'VA57676810'

def monitor_order_status(order_id):
    endpoint = f'https://{api_type}.tradier.com/v1'
    headers = {'Authorization': f'Bearer {API_KEY}', 'Accept': 'application/json'}

    filled = False

    while not filled:
        order_status_url = f'{endpoint}/accounts/{ACCOUNT_NUMBER}/orders/{order_id}'
        order_status_response = requests.get(order_status_url, headers=headers)

        if order_status_response.status_code == 200:
            order_status = order_status_response.json()['status']

            if order_status == 'filled':
                filled = True
                print('Order filled!')
            else:
                print(f'Order status: {order_status}')
        else:
            print('Failed to get order status.')
            print(order_status_response.content)
            return False

        time.sleep(10)  # Wait 10 seconds before checking the order status again

    return True


import re
import requests
import json

def decode_option_symbol(symbol):
    pattern = r'^(\w+)(\d{6})([CP])(\d{8})$'
    match = re.match(pattern, symbol)
    
    if match:
        underlying_symbol, yymmdd, option_type, strike_price = match.groups()
        option_type = "call" if option_type == "C" else "put"
        return underlying_symbol, yymmdd, option_type, int(strike_price) / 10000
    return None, None, None, None



def get_open_option_position_id(symbol, option_type):
    endpoint = f'https://{api_type}.tradier.com/v1'
    headers = {'Authorization': f'Bearer {API_KEY}', 'Accept': 'application/json'}

    positions_url = f'{endpoint}/accounts/{ACCOUNT_NUMBER}/positions'
    positions_response = requests.get(positions_url, headers=headers)

    if positions_response.status_code != 200:
        print('Error retrieving positions.')
        return None

    positions_data = positions_response.json()
    
    if not isinstance(positions_data, dict) or positions_data.get('positions') == 'null':
        return None

    positions = positions_data.get('positions', {}).get('position', [])

    # Ensure 'positions' is a list
    if not isinstance(positions, list):
        positions = [positions]

    for position in positions:
        #print("POSITION!", position)
        _, _, position_option_type, _ = decode_option_symbol(position['symbol'])
        #print("Option type:",position_option_type)
        # If the symbol could not be decoded or the position is not an option, skip it
        if position_option_type is None:
            continue

        if position_option_type.lower() == option_type.lower() and int(position['quantity']) != 0:
            return position['symbol']
    #print("Get Position ID", position)
    return None


def close_option_order(order_id, quantity=1):
    endpoint = f'https://{api_type}.tradier.com/v1'
    headers = {'Authorization': f'Bearer {API_KEY}', 'Accept': 'application/json'}

    order_url = f'{endpoint}/accounts/{ACCOUNT_NUMBER}/orders/{order_id}/close'
    order_payload = {'quantity': quantity}
    order_response = requests.post(order_url, headers=headers, data=order_payload)

    if order_response.status_code == 200:
        print(f'Order to close position {order_id} placed successfully!')
        return True
    else:
        print('Order placement failed.')
        print(order_response.content)
        return None

def get_open_order_ids(option_position_id):
    endpoint = f'https://{api_type}.tradier.com/v1'
    headers = {'Authorization': f'Bearer {API_KEY}', 'Accept': 'application/json'}

    orders_url = f'{endpoint}/accounts/{ACCOUNT_NUMBER}/orders'
    orders_response = requests.get(orders_url, headers=headers)

    open_order_ids = []

    if orders_response.status_code == 200:
        orders = orders_response.json()['orders']['order']
        for order in orders:
            if order.get('class') == 'option' and order.get('option_id') == option_position_id and order.get('status') in ['open', 'partially_filled']:
                open_order_ids.append(order.get('id'))
    else:
        print('Unable to get open orders.')
        print(orders_response.content)

    return open_order_ids


def cancel_open_orders(option_position_id):
    open_order_ids = get_open_order_ids(option_position_id)
    print(open_order_ids)

    endpoint = f'https://{api_type}.tradier.com/v1'
    headers = {'Authorization': f'Bearer {API_KEY}', 'Accept': 'application/json'}

    for order_id in open_order_ids:
        order_url = f'{endpoint}/accounts/{ACCOUNT_NUMBER}/orders/{order_id}/cancel'
        order_response = requests.post(order_url, headers=headers)

        if order_response.status_code == 200:
            print(f'Order {order_id} cancelled successfully.')
        else:
            print(f'Failed to cancel order {order_id}.')
            print(order_response.content)



def trade_option_position(action, quantity=50, order_type='market', duration='gtc'):
    endpoint = f'https://{api_type}.tradier.com/v1'
    headers = {'Authorization': f'Bearer {API_KEY}', 'Accept': 'application/json'}

    if action == 'buy':
        # Close existing put position and orders
        put_option_id = get_open_option_position_id(symbol, 'put')
        if put_option_id is not None:
            place_option_order(put_option_id, 'put', quantity, order_type, duration, 'sell')

        # Open new call position
        call_option = get_closest_option(symbol, 'call')
        if call_option is not None:
            call_option_id = place_option_order(call_option, 'call', quantity, order_type, duration, 'buy')
            return call_option_id

    elif action == 'sell':
        # Close existing call position and orders
        call_position_id = get_open_option_position_id(symbol, 'call')
        print("-----Call position ID is:", call_position_id)
        if call_position_id is not None:
            place_option_order(call_position_id, 'call', quantity, order_type, duration, 'sell')

        # Open new call position
        put_option = get_closest_option(symbol, 'put')
        if put_option is not None:
            put_option_id = place_option_order(put_option, 'put', quantity, order_type, duration, 'buy')
            return put_option_id

    elif action == 'close':
        # Close all existing positions
        call_option_id = get_open_option_position_id(symbol, 'call')
        put_option_id = get_open_option_position_id(symbol, 'put')
        if call_option_id is not None:
            place_option_order(call_option_id, 'call', quantity, order_type, duration, 'sell')

        if put_option_id is not None:
            place_option_order(put_option_id, 'put', quantity, order_type, duration, 'sell')

        return True

    else:
        print('Invalid action parameter.')
        return None

def get_closest_option(symbol, option_type):
    print("get_closest_option", symbol, option_type)
    endpoint = f'https://{api_type}.tradier.com/v1'
    headers = {'Authorization': f'Bearer {API_KEY}', 'Accept': 'application/json'}

    quote_url = f'{endpoint}/markets/quotes'
    quote_params = {'symbols': symbol}
    quote_response = requests.get(quote_url, headers=headers, params=quote_params)

    if quote_response.status_code == 200:
        quote_data = quote_response.json()['quotes']['quote']
        if isinstance(quote_data, list):
            quote_data = quote_data[0]
        price = float(quote_data['last'])
    else:
        print('Failed to get quote.')
        print(quote_response.content)
        return None

    exp_url = f'{endpoint}/markets/options/expirations'
    exp_params = {'symbol': symbol}
    exp_response = requests.get(exp_url, headers=headers, params=exp_params)

    if exp_response.status_code == 200:
        expiration = exp_response.json()['expirations']['date'][5]
    else:
        print('Failed to get expirations.')
        #print(exp_response.content)
        return None

    chains_url = f'{endpoint}/markets/options/chains'
    chains_params = {'symbol': symbol,
                     'expiration': expiration,
                     'type': option_type}
    chains_response = requests.get(chains_url, headers=headers, params=chains_params)

    if chains_response.status_code == 200:
        chains = chains_response.json()['options']['option']

        strikes = [float(option['strike']) for option in chains if option['strike'] != '0.00']
        closest_strike = min(strikes, key=lambda x: abs(x - price))

        closest_option = next(option for option in chains if float(option['strike']) == closest_strike)['symbol']
        print("Closestt optin", closest_option)
    else:
        print('Failed to get option chains.')
        print(chains_response.content)
        return None

    return closest_option

def get_existing_option_order(option_symbol, side):
    endpoint = f'https://{api_type}.tradier.com/v1'
    headers = {'Authorization': f'Bearer {API_KEY}', 'Accept': 'application/json'}

    orders_url = f'{endpoint}/accounts/{ACCOUNT_NUMBER}/orders'
    orders_response = requests.get(orders_url, headers=headers)

    if orders_response.status_code != 200:
        print('Error retrieving orders.')
        return None

    orders = orders_response.json()['orders']
    if orders is None:
        return None

    option_orders = orders['order']

    for order in option_orders:
        print(order)
        
        if order['option_symbol'] == option_symbol and order['side'] == side and (order['status'] == 'open' or order['status'] == 'pending'):
            return order['id']

    return None


def place_option_order(symbol, option_type, quantity, order_type, duration, action):
    print("PLACE OPTION ORDER")
    option = symbol
    if option is None:
        print('Unable to get closest option.')
        return None
    option_symbol = symbol
    #option_symbol =  option['symbol']
    side = None

    if action == 'buy':
        side = 'buy_to_open'
    elif action == 'sell':
        side = 'sell_to_close'

    # Check if there's already an open position in the same direction
    if(action == "buy"):
        existing_position_id = get_open_option_position_id(option_symbol, option_type)
        if existing_position_id is not None:
            print(f'Position with ID {existing_position_id} already exists for {option_symbol}.')
            return None

    # Check if there's an existing open order in the same direction
    #existing_order_id = get_existing_option_order(option_symbol, side)
    existing_order_id = None
    if existing_order_id is not None:
        print(f'Order with ID {existing_order_id} already exists for {option_symbol}.')
        return None

    price = None
    stop = None

    if order_type == 'limit' or order_type == 'stop_limit':
        price = option['ask']
    elif order_type == 'stop' or order_type == 'stop_market':
        stop = option['ask']

    endpoint = f'https://{api_type}.tradier.com/v1'
    headers = {'Authorization': f'Bearer {API_KEY}', 'Accept': 'application/json'}

    order_url = f'{endpoint}/accounts/{ACCOUNT_NUMBER}/orders'
    order_payload = {'class': 'option',
                     'symbol': symbol,
                     'option_symbol': option_symbol,
                     'side': side,
                     'quantity': quantity,
                     'type': order_type,
                     'duration': duration}

    if price is not None:
        order_payload['price'] = price

    if stop is not None:
        order_payload['stop'] = stop

    order_response = requests.post(order_url, headers=headers, data=order_payload)

    if order_response.status_code == 200:
        print('Order placed successfully!')

        return order_response.text
    else:
        print('Order placement failed.')
        print(order_response.content)
        return None

import csv
def log_prediction_to_csv(timestamp, prediction, spy_current_price):
    with open('trade-log.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, prediction, spy_current_price])

current_prediction = None

while True:
    global vix_current_price
    expirations_response = requests.get(expirations_endpoint, headers=headers, params=expirations_params)
    #print(expirations_response.text)
    expirations_data = json.loads(expirations_response.text)
    expiration_dates = expirations_data['expirations']['date']
    exp_list = expiration_dates[:5]
    vix_current_price = get_vix_value()
    timestamp = datetime.now()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for expiration in exp_list:
            futures.append(executor.submit(process_expiration, expiration))
        new_rows = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                new_rows.append(result)


    current_date = datetime.now().strftime('%Y-%m-%d') 
    
    df = pd.DataFrame(new_rows, columns=columns)
    print(df)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)  # move the file pointer to the beginning of the buffer
    csv_ = pd.read_csv(csv_buffer, parse_dates=["timestamp"])
    csv_["expiration"] = csv_["expiration"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").timestamp())
    csv_.set_index("timestamp", inplace=True)
    
    #df = pd.concat([df, new_df], ignore_index=True)
    prediction = predict_next(csv_)
    if current_prediction != prediction:
        current_prediction = prediction
        log_prediction_to_csv(timestamp, prediction, spy_current_price)
        
    ## TODO: In the AM check if its working by forcing "buy" and "sell" on the prediction.
    trade_option_position(prediction)
    print("Next prediction:", prediction)
    time.sleep(10)
    
    
    


