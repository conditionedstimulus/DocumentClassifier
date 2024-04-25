from loguru import logger

# Configure logger
logger.add(
    "logs/app.log", rotation="500 MB", level="INFO"
)  # Adjust log file path and rotation settings as needed
