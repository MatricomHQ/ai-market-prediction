import requests
import json
from datetime import datetime
import pandas as pd
import time
import concurrent.futures

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
    put_call_ratio = put_notional / call_notional

    new_row = {'timestamp': timestamp,
               'spy_current_price': spy_current_price,
               'expiration': expiration,
               'put_call_ratio': put_call_ratio,
               'total_put_call_ratio': total_put_call_ratio,
               'total_put_call_ratio_percentage': total_put_call_ratio_percentage}

    return new_row

expirations_endpoint = 'https://api.tradier.com/v1/markets/options/expirations'
chains_endpoint = 'https://api.tradier.com/v1/markets/options/chains'
quote_endpoint = 'https://api.tradier.com/v1/markets/quotes'

symbol = 'SPY'

expirations_params = {
    'symbol': symbol,
    'includeAllRoots': 'true',
    'strikes': 'false'
}

headers = {
    'Authorization': 'Bearer EV6ZirBBeJU9Hoh7UfWlNPvZiA3h',
    'Accept': 'application/json'
}

csv_file = "time_series.csv"
columns = ['timestamp', 'spy_current_price', 'expiration', 'put_call_ratio', 'total_put_call_ratio', 'total_put_call_ratio_percentage']
df = pd.DataFrame(columns=columns)
df.to_csv(csv_file, index=False)

while True:
    expirations_response = requests.get(expirations_endpoint, headers=headers, params=expirations_params)
    expirations_data = json.loads(expirations_response.text)
    expiration_dates = expirations_data['expirations']['date']
    exp_dates = expiration_dates[:10]

    total_put_notional = 0
    total_call_notional = 0

    for expiration in exp_dates:
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

        total_put_notional += put_notional
        total_call_notional += call_notional

    total_put_call_ratio = total_put_notional / total_call_notional
    total_put_call_ratio_percentage = total_put_call_ratio * 100

    quote_params = {'symbols': symbol}
    quote_response = requests.get(quote_endpoint, headers=headers, params=quote_params)
    quote_data = json.loads(quote_response.text)
    spy_current_price = quote_data['quotes']['quote']['last']

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with concurrent.futures.ThreadPoolExecutor() as executor:
        new_rows = list(executor.map(process_expiration, next_10_expirations))

    df = pd.read_csv(csv_file)
    new_df = pd.DataFrame(new_rows, columns=columns)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(csv_file, index=False)

    time.sleep(10)
