import os
from flask import Flask
from models import db

def init_db(app=None):
    """
    Инициализирует базу данных
    """
    if app is None:
        app = Flask(__name__)
        app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///crypto.db')
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
        print("База данных успешно инициализирована.")
    
    return app

if __name__ == "__main__":
    init_db()