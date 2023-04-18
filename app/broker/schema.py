import json

from py_linq.py_linq import GroupedEnumerable, Grouping, Enumerable, TEnumerable, TGrouping
from pydantic import Field, validator, BaseModel
from datetime import datetime
from typing import List, Tuple, Type

from app.shared.bases.base_schema import ORMModel


class Expiration(ORMModel):
    """
    The Expiration class is a model that represents the expiration of a token
    """
    symbol: str
    underlying_symbol: str
    date: datetime = Field(..., alias="expiration")

class ExpirationDate(ORMModel):
    date: datetime = Field(..., alias="expiration")

class Expirations(ORMModel):
    key: ExpirationDate
    data: List[Expiration]