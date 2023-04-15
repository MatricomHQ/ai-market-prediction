from typing import List
import requests
from datetime import datetime
import pandas as pd
import time
import concurrent.futures
from config import Config
from schema import OptionChainsResponse, OptionChain, QuoteResponse, RowData, ExpirationsResponse
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='runtime.log'
)


def process_expiration(expiration):
    """
    This function processes expiration data and calculates put/call ratios,
    and returns a new row of data if there are changes.

    :param expiration: The expiration parameter is a variable that represents the
    expiration date of an options contract. It is used in the function to retrieve
    options data for a specific expiration date
    :return: a dictionary with the following keys: 'timestamp', 'spy_price', 'vix',
    'expiration', 'pcr', 'put_notional', and 'call_notional'. The values for these keys
    are obtained by making API requests and performing calculations on the data received.
    If the data has not changed or the put/call ratio is 0, the function prints a message
    """
    chains_params = {
        'symbol': Config.symbol,
        'expiration': expiration
    }

    # we serialized the response using pydantic
    chains_response = requests.get(
        Config.chains_endpoint,
        headers=headers,
        params=chains_params
    )
    chains_data = OptionChainsResponse(**chains_response.json())

    def _get_options(_options: List[OptionChain], _type: str):
        for option in _options:
            if option.type == _type:
                yield option

    def _get_notional(_options: List[OptionChain]):
        for option in _options:
            yield option.bid * option.volume * option.contract_size

    # we replaced the unnecessary list comprehension with a generator expression
    puts = list(_get_options(chains_data.options, 'put'))
    calls = list(_get_options(chains_data.options, 'call'))
    put_notional = sum(list(_get_notional(puts)))
    call_notional = sum(list(_get_notional(calls)))

    """    
    while handling this with a try catch is not wrong, we can use a conditional expression
    to make the code more readable, also, we can explicitly check for 0 or cast to a boolean
    """
    # try:
    #     put_call_ratio = put_notional / call_notional
    # except ZeroDivisionError:
    #     put_call_ratio = 0

    put_call_ratio = put_notional / call_notional if call_notional else 0

    # used f string to inset variables into strings
    quote_params = {
        'symbols': f"{Config.symbol},{Config.vix_symbol}"
    }

    # serialized the response using pydantic
    quote_response = requests.get(
        Config.quote_endpoint,
        headers=headers,
        params=quote_params
    )
    quote_data = QuoteResponse(**quote_response.json())

    # we used list comprehension to get the last price of the quotes
    spy_current_price, vix_current_price = [quote.last for quote in quote_data.quotes]

    new_row = RowData(
        spy_price=spy_current_price,
        vix=vix_current_price,
        expiration=expiration,
        pcr=put_call_ratio,
        put_notional=put_notional,
        call_notional=call_notional
    )

    def compare_rows(_new_row, previous_row):
        """
        This function compares the new row of data with the previous row of data.
        :param _new_row:  The new row of data
        :param previous_row:  The previous row of data
        :return:  The new row of data if the data has changed, otherwise, it returns None
        """

        # explicitly casting _new_row.pcr to a bool lets us check the truthy for None and 0
        if previous_row == _new_row or not bool(_new_row.pcr):
            logging.info("The data hasn't changed")
            return None
        return new_row

    # not used because the original method did nothing,
    # explicitly pass the old row data to the function
    # previous_row = compare_rows(new_row, previous_row)

    new_row.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return new_row


expirations_params = {
    'symbol': Config.symbol,
    'includeAllRoots': 'true',
    'strikes': 'false'
}

headers = {
    'Authorization': f'Bearer {Config.token}',
    'Accept': 'application/json'
}

columns = ['timestamp', 'spy_price', 'vix', 'expiration', 'pcr', 'put_notional', 'call_notional']

while True:
    """
    Used json() method of response object to load JSON directly instead of using json.loads().
    Removed unnecessary variable expirations_data.
    Combined lines 3 and 4 to just one line.
    Used executor.map() instead of list comprehension and submit().
    yields the result of the future only if it's truthy.
    """
    expirations_response = requests.get(
        Config.expirations_endpoint, headers=headers, params=expirations_params
    )
    expirations_data = ExpirationsResponse(**expirations_response.json())
    expiration_dates = [expiration.date for expiration in expirations_data.expirations[:10]]

    with concurrent.futures.ThreadPoolExecutor() as executor:

        # using list comprehension and mapping the function
        futures = list(executor.map(process_expiration, expiration_dates))

        # using a generator expression to simplify the logic
        # note that the map function returns a generator containing the results of the futures
        new_rows = (result for result in futures if result)

        current_date = datetime.now().strftime('%Y-%m-%d')
        csv_file = f"data/{Config.symbol}_{current_date}_time_series.csv"  # Modify the CSV file name here

        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            df = pd.DataFrame(columns=columns)  # Create a new DataFrame if file doesn't exist

        new_df = pd.DataFrame(new_rows, columns=columns)
        df = pd.concat([df, new_df], ignore_index=True)

        # Check if there are any new rows to append
        if not new_df.empty:
            df.to_csv(csv_file, index=False)
            logging.info(f"{len(new_df)} new rows added to {csv_file}")
        else:
            logging.info("No new data to write to CSV")

        # Check if the market is open every minute
    time.sleep(10)
