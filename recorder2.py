import requests
import json
from datetime import datetime, time as dtime
import pandas as pd
import time
import concurrent.futures
import os

timestamp = None
previous_row = None
def process_expiration(expiration):
    chains_params = {
        'symbol': symbol,
        'expiration': expiration
    }
    chains_response = requests.get(chains_endpoint, headers=headers, params=chains_params)
    chains_data = json.loads(chains_response.text)
    
    puts = [option for option in chains_data['options']['option'] if option['option_type'] == 'put']
    calls = [option for option in chains_data['options']['option'] if option['option_type'] == 'call']
    
    put_notional = sum([put['bid'] * put['volume'] * put['contract_size'] for put in puts])
    call_notional = sum([call['bid'] * call['volume'] * call['contract_size'] for call in calls])
    try:
        put_call_ratio = put_notional / call_notional
    except ZeroDivisionError:
        put_call_ratio = 0
    
    quote_params = {'symbols': symbol + ',' + vix_symbol}
    quote_response = requests.get(quote_endpoint, headers=headers, params=quote_params)
    quote_data = json.loads(quote_response.text)
    spy_current_price = quote_data['quotes']['quote'][0]['last']
    vix_current_price = quote_data['quotes']['quote'][1]['last']
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    new_row = {'timestamp': timestamp,
               'spy_price': spy_current_price,
               'vix': vix_current_price,
               'expiration': expiration,
               'pcr': put_call_ratio,
               'put_notional': put_notional,
               'call_notional': call_notional}
    
    global previous_row
    comparing_row = {k: v for k, v in new_row.items() if k != 'timestamp'}
    if previous_row is not None and comparing_row == previous_row or put_call_ratio==0:
        print("The data hasn't changed")
    else:
        print(new_row)
        previous_row = comparing_row
        return new_row


expirations_endpoint = 'https://api.tradier.com/v1/markets/options/expirations'
chains_endpoint = 'https://api.tradier.com/v1/markets/options/chains'
quote_endpoint = 'https://api.tradier.com/v1/markets/quotes'

symbol = 'SPY'
vix_symbol = 'VIX'

expirations_params = {
    'symbol': symbol,
    'includeAllRoots': 'true',
    'strikes': 'false'
}

headers = {
    'Authorization': 'Bearer EV6ZirBBeJU9Hoh7UfWlNPvZiA3h',
    'Accept': 'application/json'
}

columns = ['timestamp', 'spy_price', 'vix', 'expiration', 'pcr', 'put_notional', 'call_notional']


while True:

    expirations_response = requests.get(expirations_endpoint, headers=headers, params=expirations_params)
    expirations_data = json.loads(expirations_response.text)
    expiration_dates = expirations_data['expirations']['date']
    next_10_expirations = expiration_dates[:10]
    timestamp = datetime.now()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for expiration in next_10_expirations:
            futures.append(executor.submit(process_expiration, expiration))
        new_rows = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                new_rows.append(result)
    
    current_date = datetime.now().strftime('%Y-%m-%d') 
    csv_file = f"data/{symbol}_{current_date}_time_series.csv"  # Modify the CSV file name here
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=columns)  # Create a new DataFrame if file doesn't exist
    
    new_df = pd.DataFrame(new_rows, columns=columns)
    df = pd.concat([df, new_df], ignore_index=True)
    
    # Check if there are any new rows to append
    if not new_df.empty:
        df.to_csv(csv_file, index=False)
        print(f"{len(new_df)} new rows added to {csv_file}")
    else:
        print("No new data to write to CSV")
        
    # Check if the market is open every minute
    time.sleep(10)
