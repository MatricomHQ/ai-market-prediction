from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class ExpirationDate(BaseModel):
    """
    `ExpirationDate` is a class that is used to represent an expiration date.
    """
    date: str


class ExpirationsResponse(BaseModel):
    """
    `ExpirationsResponse` is a class that is used to represent an expiration date.
    """
    expirations: List[ExpirationDate]


class OptionChain(BaseModel):
    """
    `OptionChain` is a class that is used to represent an option chain.
    """
    symbol: Optional[str]
    description: Optional[str]
    exch: Optional[str]
    type: Optional[str]
    last: Optional[float]
    change: Optional[float]
    change_percentage: Optional[float]
    volume: Optional[int]
    open_interest: Optional[int]
    bid: Optional[float]
    ask: Optional[float]
    underlying: Optional[str]
    strike: Optional[float]
    expiration_date: Optional[str]
    expiration_type: Optional[str]
    contract_size: Optional[int]


class OptionChainsResponse(BaseModel):
    """
    `OptionChainsResponse` is a class that is used to represent an option chain.
    """
    underlying: str
    expiration_dates: List[str]
    strikes: List[float]
    options: List[OptionChain]


class Quote(BaseModel):
    """
    `Quote` is a class that is used to represent a quote.
    """
    symbol: Optional[str]
    description: Optional[str]
    exch: Optional[str]
    type: Optional[str]
    last: Optional[float]
    change: Optional[float]
    change_percentage: Optional[float]
    volume: Optional[int]
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    bid: Optional[float]
    ask: Optional[float]
    underlying: Optional[str]


class QuoteResponse(BaseModel):
    """
    `QuoteResponse` is a class that is used to represent a quote.
    """
    quotes: List[Quote]


class RowData(BaseModel):
    """
    `RowData` is a class that is used to represent a row of data.
    """
    timestamp: Optional[str]
    spy_price: float
    vix: float
    expiration: str
    pcr: float
    put_notional: float
    call_notional: float
