from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class OHLCV4h(db.Model):
    """Модель для хранения 4-часовых OHLCV данных"""
    __tablename__ = 'ohlcv_4h'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False, index=True)
    timestamp = db.Column(db.Integer, nullable=False)  # Unix timestamp
    time = db.Column(db.DateTime, nullable=False, index=True)
    open = db.Column(db.Float, nullable=False)
    high = db.Column(db.Float, nullable=False)
    low = db.Column(db.Float, nullable=False)
    close = db.Column(db.Float, nullable=False)
    volumefrom = db.Column(db.Float, nullable=False)
    volumeto = db.Column(db.Float, nullable=False)
    
    # Уникальное ограничение: один timestamp для одного символа
    __table_args__ = (
        db.UniqueConstraint('symbol', 'timestamp', name='uix_symbol_timestamp'),
    )
    
    def __repr__(self):
        return f"<OHLCV4h(symbol='{self.symbol}', time='{self.time}')>"
    
    @classmethod
    def from_api_data(cls, symbol, data_point):
        """Создает объект модели из данных API"""
        return cls(
            symbol=symbol,
            timestamp=data_point['time'],
            time=datetime.fromtimestamp(data_point['time']),
            open=data_point['open'],
            high=data_point['high'],
            low=data_point['low'],
            close=data_point['close'],
            volumefrom=data_point['volumefrom'],
            volumeto=data_point['volumeto']
        )

class DataMetrics(db.Model):
    """Модель для хранения метрик и статистики данных"""
    __tablename__ = 'data_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False, index=True)
    data_type = db.Column(db.String(20), nullable=False)  # '4h', 'daily', etc.
    oldest_record = db.Column(db.DateTime, nullable=True)
    newest_record = db.Column(db.DateTime, nullable=True)
    record_count = db.Column(db.Integer, nullable=False, default=0)
    last_updated = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    # Уникальное ограничение: одна запись для символа и типа данных
    __table_args__ = (
        db.UniqueConstraint('symbol', 'data_type', name='uix_symbol_datatype'),
    )
    
    def __repr__(self):
        return f"<DataMetrics(symbol='{self.symbol}', data_type='{self.data_type}')>"