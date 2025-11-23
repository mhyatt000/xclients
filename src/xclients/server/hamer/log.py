import logging

import colorama
from colorama import Fore, Style


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors based on log level."""

    LOG_COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        log_color = self.LOG_COLORS.get(record.levelno, Fore.WHITE)
        message = super().format(record)
        return f"{log_color}{message}{Style.RESET_ALL}"


class Logger:
    def __init__(self, name="MyLogger", level=logging.DEBUG):
        # Create a logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Define formatter with time, name, level, and message
        formatter = ColoredFormatter(
            "%(message)s",  # '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger.propagate = False

        # Attach formatter to the console handler
        console_handler.setFormatter(formatter)

        # Attach handler to the logger
        if not self.logger.hasHandlers():
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger


# Initialize colorama
colorama.init(autoreset=True)

logger = Logger().get_logger()
