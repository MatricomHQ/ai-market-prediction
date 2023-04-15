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
    symbol = os.getenv('SYMBOL')
    vix_symbol = os.getenv('VIX_SYMBOL')
