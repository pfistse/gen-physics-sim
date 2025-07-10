"""Utility helpers for colored logging."""

import logging
import sys
from typing import Optional, Dict, Any
import os

# Define log levels
VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")

# Color codes for terminal output
COLORS = {
    "RESET": "\033[0m",
    "RED": "\033[31m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "WHITE": "\033[37m",
    "BOLD": "\033[1m",
}

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels."""
    
    LEVEL_COLORS = {
        "DEBUG": COLORS["BLUE"],
        "VERBOSE": COLORS["CYAN"],
        "INFO": COLORS["GREEN"],
        "WARNING": COLORS["YELLOW"],
        "ERROR": COLORS["RED"],
        "CRITICAL": COLORS["RED"] + COLORS["BOLD"],
    }
    
    def format(self, record):
        levelname = record.levelname
        if levelname in self.LEVEL_COLORS:
            record.levelname = f"{self.LEVEL_COLORS[levelname]}{levelname}{COLORS['RESET']}"
        return super().format(record)

def setup_logger(
    name: str = "gen-physics-sim",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
) -> logging.Logger:
    """Configure and return a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create console handler with colored output if enabled
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Create formatter with time, level, and message
        console_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        console_formatter = ColoredFormatter(console_format, datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(console_formatter)

        # Add console handler to logger
        logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create a default logger instance
logger = setup_logger()

def verbose(self, message, *args, **kwargs):
    """Log a message with VERBOSE level."""
    self.log(VERBOSE, message, *args, **kwargs)

# Add verbose method to Logger class
logging.Logger.verbose = verbose

def get_logger(name: str) -> logging.Logger:
    """Return a named child logger."""
    return logging.getLogger(f"gen-physics-sim.{name}")

def set_global_level(level: int) -> None:
    """Set the global logging level."""
    logging.getLogger("gen-physics-sim").setLevel(level)
    
    # Update all handlers
    for handler in logging.getLogger("gen-physics-sim").handlers:
        handler.setLevel(level)

def log_dict(logger: logging.Logger, title: str, data: Dict[str, Any], level: int = logging.INFO) -> None:
    """Log a dictionary line by line."""
    if level >= logger.level:
        logger.log(level, f"{title}:")
        for key, value in data.items():
            logger.log(level, f"  {key}: {value}")
