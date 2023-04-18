from datetime import datetime
import pytz
from sqlalchemy import ARRAY
from sqlalchemy import Column, Text, ForeignKey, DateTime, Integer, Float
from sqlalchemy.ext.hybrid import hybrid_property

from app.broker.schema import Expiration
from app.shared.bases.base_model import ModelMixin, ModelType
from config import Config


class Chain(ModelMixin):
    __tablename__ = "chain"
    id = Column(Integer, autoincrement=True, primary_key=True, index=True)
    symbol = Column(Text, nullable=False, unique=True)
    expiration = Column(DateTime, nullable=False)
    strike = Column(Float, nullable=False)
    option_type = Column(Text, nullable=False)
    underlying_symbol = Column(Text, nullable=False)

    @hybrid_property
    def vix_calls(self):
        return self.where(underlying_symbol=Config.vix_symbol, option_type="call").all()

    @hybrid_property
    def vix_puts(self):
        return self.where(underlying_symbol=Config.vix_symbol, option_type="put").all()

    @hybrid_property
    def spy_calls(self):
        return self.where(underlying_symbol=Config.spy_symbol, option_type="call").all()

    @hybrid_property
    def spy_puts(self):
        return self.where(underlying_symbol=Config.spy_symbol, option_type="put").all()

    @hybrid_property
    def expirations(self) -> ModelType:
        expirations = self.session.query(
            self.symbol,
            self.underlying_symbol,
            self.expiration
        ).distinct().order_by(self.expiration).all()
        return expirations

    def __repr__(self):
        return f"""
        Chain(
        id={self.id}, 
        symbol={self.symbol}, 
        expiration={self.expiration}, 
        strike={self.strike}, 
        option_type={self.option_type}, 
        underlying_symbol={self.underlying_symbol}
        )"""


class ChainAggregates(ModelMixin):
    __tablename__ = "chain_aggregates"
    id = Column(Integer, autoincrement=True, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=lambda: datetime.now(pytz.utc))
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    chain = Column(Integer, ForeignKey("chain.id"), nullable=False, index=True)
    underlying_symbol = Column(Text, nullable=False, index=True)

class CombinedChainDetails(ModelMixin):
    __tablename__ = "combined_chain_details"
    id = Column(Integer, autoincrement=True, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=lambda: datetime.now(pytz.utc))
    spy_price = Column(Float, nullable=False)
    vix_price = Column(Float, nullable=False)
    expiration = Column(DateTime, nullable=False)
    pcr = Column(Float, nullable=False)
    put_notional = Column(Float, nullable=False)
    call_notional = Column(Float, nullable=False)
    chains = Column(ARRAY(Integer), nullable=False)


