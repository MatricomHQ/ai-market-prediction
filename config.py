"""
Author: Michael Brown
Github: @slapglif
"""
from dotenv import load_dotenv
import os

load_dotenv()


class Config:
    expirations_endpoint = os.getenv('EXPIRATIONS_ENDPOINT')
    chains_endpoint = os.getenv('CHAINS_ENDPOINT')
    quote_endpoint = os.getenv('QUOTE_ENDPOINT')
    spy_symbol = os.getenv('SPY_SYMBOL')
    vix_symbol = os.getenv('VIX_SYMBOL')
    token = os.getenv('TOKEN')
    polygon_key = os.getenv('POLYGON_KEY')
    postgres_connection = os.getenv('POSTGRES_CONNECTION')
