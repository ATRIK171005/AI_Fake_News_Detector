"""
database.py
-----------
SQLite 3NF Relational Persistence layer for the AI-Based Fake News Detection System.
Tracks historical prediction logs, user audit feedback, and model performance metrics.
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from utils import get_logger, DatabaseError

logger = get_logger("DatabaseManager")

DB_PATH = os.path.join(os.path.dirname(__file__), "database", "fake_news_audit.db")


def get_connection() -> sqlite3.Connection:
    """Returns a resilient SQLite database connection."""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database at {DB_PATH}: {str(e)}")
        raise DatabaseError(f"Database connection error: {str(e)}") from e


def initialize_database() -> None:
    """
    Creates normalized 3NF relational tables:
    1. Models: Metadata of trained classifiers (e.g., Logistic Regression, Naive Bayes).
    2. PredictionAudit: Individual news articles tested by users with model prediction & probability.
    3. UserFeedback: Human-in-the-loop verification (whether user agreed or flagged prediction).
    """
    logger.info("Initializing SQLite 3NF schema for Fake News Audit...")
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS Models (
                ModelID INTEGER PRIMARY KEY AUTOINCREMENT,
                ModelName TEXT UNIQUE NOT NULL,
                Algorithm TEXT NOT NULL,
                Vectorizer TEXT NOT NULL,
                Accuracy REAL NOT NULL,
                PrecisionScore REAL NOT NULL,
                RecallScore REAL NOT NULL,
                F1Score REAL NOT NULL,
                TrainedDate TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS PredictionAudit (
                AuditID INTEGER PRIMARY KEY AUTOINCREMENT,
                ModelUsed TEXT NOT NULL,
                RawText TEXT NOT NULL,
                CleanedText TEXT NOT NULL,
                PredictedLabel TEXT NOT NULL,
                ConfidenceScore REAL NOT NULL,
                Timestamp TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS UserFeedback (
                FeedbackID INTEGER PRIMARY KEY AUTOINCREMENT,
                AuditID INTEGER NOT NULL,
                UserVerdict TEXT NOT NULL, -- 'Agreed' or 'Disagreed'
                UserComment TEXT,
                SubmittedAt TEXT NOT NULL,
                FOREIGN KEY (AuditID) REFERENCES PredictionAudit(AuditID)
            );
        """)
        conn.commit()
        logger.info("Database schema initialized successfully.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Database initialization failed: {str(e)}")
        raise DatabaseError(f"Schema setup failed: {str(e)}") from e
    finally:
        conn.close()


def save_model_metadata(model_name: str, algorithm: str, vectorizer: str, metrics: Dict[str, float]) -> void if False else None:
    """Inserts or updates trained model performance metadata."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO Models 
            (ModelName, Algorithm, Vectorizer, Accuracy, PrecisionScore, RecallScore, F1Score, TrainedDate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_name,
            algorithm,
            vectorizer,
            round(metrics.get("accuracy", 0.0), 4),
            round(metrics.get("precision", 0.0), 4),
            round(metrics.get("recall", 0.0), 4),
            round(metrics.get("f1", 0.0), 4),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving model metadata: {str(e)}")
    finally:
        conn.close()


def log_prediction(model_used: str, raw_text: str, cleaned_text: str, predicted_label: str, confidence: float) -> int:
    """Logs a live prediction test into the PredictionAudit table and returns the AuditID."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO PredictionAudit (ModelUsed, RawText, CleanedText, PredictedLabel, ConfidenceScore, Timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            model_used,
            raw_text[:2000],  # Cap storage length
            cleaned_text[:2000],
            predicted_label,
            round(confidence, 4),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        audit_id = cursor.lastrowid
        return audit_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Error logging prediction: {str(e)}")
        return -1
    finally:
        conn.close()


def log_user_feedback(audit_id: int, verdict: str, comment: str = "") -> bool:
    """Logs human verification feedback on a specific audit log."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO UserFeedback (AuditID, UserVerdict, UserComment, SubmittedAt)
            VALUES (?, ?, ?, ?)
        """, (
            audit_id,
            verdict,
            comment,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        logger.error(f"Error logging feedback: {str(e)}")
        return False
    finally:
        conn.close()


def get_audit_history(limit: int = 50) -> pd.DataFrame:
    """Returns the most recent tested articles with any associated user feedback."""
    conn = get_connection()
    try:
        query = """
            SELECT p.AuditID, p.Timestamp, p.ModelUsed, p.PredictedLabel, p.ConfidenceScore,
                   substr(p.RawText, 1, 120) || '...' AS Snippet,
                   COALESCE(f.UserVerdict, 'Pending Review') AS Verification
            FROM PredictionAudit p
            LEFT JOIN UserFeedback f ON p.AuditID = f.AuditID
            ORDER BY p.AuditID DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        return df
    except Exception as e:
        logger.error(f"Failed to fetch audit history: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()


def get_model_leaderboard() -> pd.DataFrame:
    """Returns historical performance comparison across trained classifiers."""
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT ModelName, Algorithm, Accuracy, PrecisionScore, RecallScore, F1Score, TrainedDate FROM Models ORDER BY F1Score DESC", conn)
        return df
    except Exception as e:
        logger.error(f"Failed to fetch model leaderboard: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()


# Ensure database initializes on import
initialize_database()
