from types import SimpleNamespace
from typing import List, Union, Optional, Iterable, Any
import requests
from datetime import datetime
import pandas as pd
import time
import concurrent.futures
from config import Config
from app.data.schema import OptionChainsResponse, OptionChain, QuoteResponse, RowData, ExpirationsResponse
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='runtime.log'
)


def fetch_api_data(url: str, params: dict, response_model: Any) -> Any:
    """
    This function fetches data from an API using a given URL, parameters, and response model.

    :param url: The URL of the API endpoint that you want to fetch data from
    :param params: The `params` parameter is a dictionary that contains the query
    parameters to be sent with the API request. These parameters are used to filter or modify the data
    that is returned by the API. For example, if the API returns a list of items, the `params` dictionary might contain filters to only
    :param response_model: response_model is a parameter that specifies the type of data model that
    the API response should be converted to. It is usually a Pydantic BaseModel subclass
    that defines the structure of the expected response data. The fetch_api_data function uses this model to
    parse the JSON response from the API and return an instance
    :return: an instance of the `response_model` class, which is created using the JSON data returned by the API call.
    """

    headers = {
        'Authorization': f'Bearer {Config.token}',
        'Accept': 'application/json'
    }
    response = requests.get(url, headers=headers, params=params)
    return response_model(**response.json())


def generate_quote_data(expiration) -> Iterable[Union[OptionChainsResponse, QuoteResponse]]:
    """
    The function generates quote data by fetching data from two API endpoints and yielding the results.

    :param expiration: The expiration parameter is a variable that is passed to the generate_quote_data function. It is used as a parameter in the API call to retrieve option chains
    data for a specific expiration date
    """
    api_dependencies = [
        SimpleNamespace(
            url=Config.chains_endpoint,
            response_model=OptionChainsResponse,
            params=dict(
                symbol=Config.symbol,
                expiration=expiration
            )

        ),
        SimpleNamespace(
            url=Config.quote_endpoint,
            response_model=QuoteResponse,
            params=dict(
                symbols=f"{Config.symbol},{Config.vix_symbol}"
            )
        )
    ]

    for param in api_dependencies:
        yield fetch_api_data(param.url, param.params, param.response_model)


def compare_rows(_new_row: pd.DataFrame, previous_row) -> Optional[pd.DataFrame]:
    """
    This function compares the new row of data with the previous row of data.
    :param _new_row:  The new row of data
    :param previous_row:  The previous row of data
    :return:  The new row of data if the data has changed, otherwise, it returns None
    """

    if previous_row == _new_row or not bool(_new_row.pcr):
        logging.info("The data hasn't changed")
        return None
    _new_row.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return _new_row


def process_expiration(expiration: str) -> RowData:
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
    chains_data, quote_data = generate_quote_data(expiration)

    def _get_options(_options: List[OptionChain], _type: str) -> Iterable[OptionChain]:
        """
        This function returns a generator that yields options of a specific type from a list of option chains.

        :param _options: It is a list of OptionChain objects
        :param _type: The parameter `_type` is a string that represents the type of option chain being
        searched for. It is used to filter the list of option chains passed in as the
        `_options` parameter
        """
        for option in _options:
            if option.type == _type:
                yield option

    def _get_notional(_options: List[OptionChain]) -> Iterable[Union[int, float]]:
        """
        This function calculates the total notional value of a list of option chains by
        multiplying the bid price, volume, and contract size of each option.

        :param _options: The parameter `_options` is a list of `OptionChain` objects
        """
        for option in _options:
            yield option.bid * option.volume * option.contract_size

    puts = list(_get_options(chains_data.options, 'put'))
    calls = list(_get_options(chains_data.options, 'call'))
    put_notional = sum(list(_get_notional(puts)))
    call_notional = sum(list(_get_notional(calls)))

    put_call_ratio = put_notional / call_notional if (call_notional and put_notional) else 0
    spy_current_price, vix_current_price = [quote.last for quote in quote_data.quotes]

    return RowData(
        spy_price=spy_current_price,
        vix=vix_current_price,
        expiration=expiration,
        pcr=put_call_ratio,
        put_notional=put_notional,
        call_notional=call_notional,
    )


def run_threads() -> List[RowData]:
    """
    The function runs multiple threads to fetch expiration data and process it for the given symbol.
    :return: The function `run_threads()` is returning a list of results from the
    `process_expiration()` function for the first 10 expiration dates fetched from an API endpoint. The
    list comprehension at the end filters out any `None` values from the results.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        expiration_params = SimpleNamespace(
            url=Config.expirations_endpoint,
            response_model=ExpirationsResponse,
            params=dict(
                symbol=Config.symbol,
                includeAllRoots='true',
                strikes='false'
            )
        )
        expirations_data = fetch_api_data(
            expiration_params.url,
            expiration_params.params,
            expiration_params.response_model
        )

        expiration_dates = [expiration.date for expiration in expirations_data.expirations[:10]]
        futures = list(executor.map(process_expiration, expiration_dates))
        return [result for result in futures if result]


def check_new_data(new_quote_data: pd.DataFrame, old_quote_data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    The function checks if new quote data is different from old quote data and drops any rows that are not different.

    :param new_quote_data: A pandas DataFrame containing new quote data to be checked against old quote data
    :param old_quote_data: The parameter `old_quote_data` is a pandas DataFrame containing the previous
     quote data that needs to be compared with the new quote data
    :return: either a modified version of the `new_quote_data` DataFrame or `None`. If
    `new_quote_data` is empty, the function returns `None`. Otherwise, it returns the modified
    `new_quote_data` DataFrame.
    """
    for index, row in enumerate(new_quote_data):

        if not compare_rows(new_quote_data.iloc[index], old_quote_data.iloc[index]):
            new_quote_data.drop(index, inplace=True)

    if new_quote_data.empty:
        return

    return new_quote_data


def process_data(row_data: List[RowData]) -> None:
    """
    This function processes row data and writes it to a CSV file if there is new data.

    :param row_data: `row_data` is a list of `RowData` objects. Each `RowData`
     object represents a row of data that needs to be processed and added to a CSV file.
     The `process_data`
    function takes this list of `RowData` objects and adds them to a CSV file if
    :return: nothing (i.e., `None`).
    """
    columns = [
        'timestamp',
        'spy_price',
        'vix',
        'expiration',
        'pcr',
        'put_notional',
        'call_notional'
    ]
    current_date = datetime.now().strftime('%Y-%m-%d')
    csv_file = f"data/{Config.symbol}_{current_date}_time_series.csv"  # Modify the CSV file name here

    try:
        old_quote_data = pd.read_csv(csv_file)
    except FileNotFoundError:
        old_quote_data = pd.DataFrame(columns=columns)  # Create a new DataFrame if file doesn't exist

    new_quote_data = pd.DataFrame([row.dict() for row in row_data], columns=columns)
    if verified_quote_data := check_new_data(new_quote_data, old_quote_data):
        final_row_dump = pd.concat([old_quote_data, new_quote_data], ignore_index=True)
        final_row_dump.to_csv(csv_file, index=False)
        logging.info(f"{len(verified_quote_data)} new rows added to {csv_file}")

    logging.info("No new data to write to CSV")
    return


while True:
    process_data(run_threads())
    time.sleep(10)
