import logging
import os


class Logger:
    """Logger class"""
    def __init__(self, log_dir: str, log_filename: str):
        self.logger = self._setup_logging(log_dir, log_filename)

    def _setup_logging(self, log_dir: str, log_filename: str) -> logging.Logger:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, log_filename)
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized. Logs are being saved to {log_file}")
        return logger

    def info(self, message: str):
        self.logger.info(message)

    def error(self, message: str):
        self.logger.error(message)

    def warning(self, message: str):
        self.logger.warning(message)