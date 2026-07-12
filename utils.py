"""
utils.py
--------
Enterprise logging and custom exception hierarchy for the AI-Based Fake News Detection System.
"""

import logging
import sys
from datetime import datetime
from typing import Optional


# =====================================================================
# Custom Exception Hierarchy
# =====================================================================
class FakeNewsDetectionError(Exception):
    """Base exception class for the Fake News Detection Platform."""
    pass


class PreprocessingError(FakeNewsDetectionError):
    """Raised when text cleaning or NLP preprocessing fails."""
    pass


class VectorizationError(FakeNewsDetectionError):
    """Raised when TF-IDF feature extraction or matrix transformation encounters errors."""
    pass


class ModelTrainingError(FakeNewsDetectionError):
    """Raised when classifier fitting or hyperparameter tuning fails."""
    pass


class ModelEvaluationError(FakeNewsDetectionError):
    """Raised during accuracy, precision, recall, or F1 metric calculation failures."""
    pass


class DatabaseError(FakeNewsDetectionError):
    """Raised when SQLite relational queries or audit persistence fails."""
    pass


# =====================================================================
# Enterprise Logger Setup
# =====================================================================
def get_logger(name: str = "AIFakeNewsDetector", level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured enterprise logger with standardized formatting.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
