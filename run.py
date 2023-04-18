import json
import logging
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import List

from app.broker.models import Chain
from app.broker.polygon_client import get_chains
from app.broker.schema import Expiration, Expirations
from config import Config
from py_linq import Enumerable

logger = logging.getLogger('app')


def build_chain_data():
    logger.info("building chain data...")
    return list(
        get_chains(
            [
                Config.spy_symbol,
                Config.vix_symbol
            ]
        ))


def get_expirtations(day_delta: int = 1) -> List[Expirations]:
    chain_data = Enumerable(Chain.expirations)
    delta = datetime.now() - timedelta(days=day_delta)
    chain_result = chain_data.where(
        lambda x: x.expiration < datetime.now() > delta
    ).group_by(
        key_names=['expiration'],
        key=lambda x: x.expiration
    ).order_by(
        lambda x: x.key
    )
    return [Expirations(key=expiration.key, data=expiration.to_list()) for expiration in chain_result]


expirations = get_expirtations(day_delta=30)
for expiration in expirations:
    print(expiration.key.date.strftime("%m/%d/%Y"))
    for data in expiration.data:
        print(data.symbol)

