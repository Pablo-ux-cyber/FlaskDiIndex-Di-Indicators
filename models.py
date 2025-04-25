from sqlalchemy import Column, Integer, String, Numeric, DateTime, BigInteger, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class CryptoData4h(Base):
    """Model for storing 4-hour cryptocurrency data"""
    __tablename__ = 'crypto_data_4h'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    time_value = Column(BigInteger, nullable=False)  # Original timestamp as integer
    open = Column(Numeric(20, 8), nullable=False)
    high = Column(Numeric(20, 8), nullable=False)
    low = Column(Numeric(20, 8), nullable=False)
    close = Column(Numeric(20, 8), nullable=False)
    volumefrom = Column(Numeric(20, 8), nullable=False)
    volumeto = Column(Numeric(20, 8), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Ensure we don't have duplicate entries for same symbol and time
    __table_args__ = (
        UniqueConstraint('symbol', 'time_value', name='uix_symbol_time'),
    )
    
    def __repr__(self):
        return f"<CryptoData4h(symbol='{self.symbol}', timestamp='{self.timestamp}', close={self.close})>"