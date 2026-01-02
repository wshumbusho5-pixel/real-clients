"""
Aerial Leads - Logging Configuration

Centralized logging setup for the entire application.
Uses both file logging and rich console logging.
"""

import logging
import sys
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console

from config.settings import LOG_LEVEL, LOG_FILE, LOG_FORMAT, LOG_DATE_FORMAT, RICH_LOG_FORMAT

# Create console for rich output
console = Console()


def setup_logging(name: str = 'aerial_leads', log_file: Path = LOG_FILE) -> logging.Logger:
    """
    Set up logging with both file and console handlers

    Args:
        name: Logger name
        log_file: Path to log file

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # File handler - detailed logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Always DEBUG for file
    file_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Rich console handler - pretty terminal output
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )
    console_handler.setLevel(getattr(logging, LOG_LEVEL.upper()))
    console_formatter = logging.Formatter(RICH_LOG_FORMAT, datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Don't propagate to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    # If logger already exists, return it
    if logging.getLogger(name).handlers:
        return logging.getLogger(name)

    # Otherwise set it up
    return setup_logging(name)


# Create main application logger
main_logger = setup_logging('aerial_leads')


# Utility functions for easy logging
def log_info(message: str):
    """Log info message"""
    main_logger.info(message)


def log_error(message: str, exc_info=False):
    """Log error message"""
    main_logger.error(message, exc_info=exc_info)


def log_warning(message: str):
    """Log warning message"""
    main_logger.warning(message)


def log_debug(message: str):
    """Log debug message"""
    main_logger.debug(message)


def log_success(message: str):
    """Log success message with emoji"""
    console.print(f"[bold green]✅ {message}[/bold green]")
    main_logger.info(message)


def log_failure(message: str):
    """Log failure message with emoji"""
    console.print(f"[bold red]❌ {message}[/bold red]")
    main_logger.error(message)


# Example usage
if __name__ == '__main__':
    # Test the logging setup
    logger = get_logger(__name__)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    log_success("This is a success message!")
    log_failure("This is a failure message!")
