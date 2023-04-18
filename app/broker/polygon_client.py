"""
Author: Michael Brown
Github: @slapglif
"""
import logging

from sqlalchemy_mixins.activerecord import ActiveRecordMixin

from app.broker.models import Chain
from config import Config
from polygon import RESTClient
from typing import List

logger = logging.getLogger('app')

client = RESTClient(api_key=Config.polygon_key)


def get_chains(assets: List[str]) -> List[ActiveRecordMixin]:
    for asset in assets:
        logger.info(f"getting options chains for {asset}...")
        options_chains = client.list_snapshot_options_chain(underlying_asset=asset, params={'expiration_date.gte': '2021-01-01'})
        for option in options_chains:
            if chain := Chain.create(
                option_type=option.details.contract_type,
                strike=option.details.strike_price,
                expiration=option.details.expiration_date,
                symbol=option.details.ticker,
                underlying_symbol=option.underlying_asset.ticker,
            ):
                logger.info(f"added {chain.symbol} on {chain.underlying_symbol} to the database")
    return Chain.all()
#
# options_aggs = client.get_aggs(ticker='O:SPY251219C00650000', multiplier=1, timespan='minute', from_='2023-01-01', to='2023-01-31', adjusted=True)
# for agg in options_aggs:
#     close = agg.close
#     volume = agg.volume
#     print(f'Close: {close}, Volume: {volume}')