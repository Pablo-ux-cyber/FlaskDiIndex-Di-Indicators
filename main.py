from flask import Flask
import os
import logging
from utils.logging_config import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

logger.info("Starting application initialization...")

# Create Flask app
try:
    app = Flask(__name__)
    app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")
    logger.info("Flask app created successfully")

    # Import routes and register them
    from server import di_index_blueprint
    app.register_blueprint(di_index_blueprint)
    logger.info("Blueprint registered successfully")

except Exception as e:
    logger.error(f"Error during app initialization: {str(e)}", exc_info=True)
    raise

if __name__ == '__main__':
    logger.info("Starting DI Index API server...")
    app.run(host='0.0.0.0', port=5000, debug=True)