"""
model_trainer.py
----------------
Complete training pipeline for the AI-Based Fake News Detection System.
Loads `real_vs_fake.csv`, performs multi-stage NLP cleaning, extracts n-gram TF-IDF features,
trains multiple distinct classifiers (`Logistic Regression`, `Multinomial Naive Bayes`,
`Random Forest`, and `Passive Aggressive`), benchmarks exact evaluation metrics,
and serializes models via joblib.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from nlp_pipeline import TextPreprocessor
from model_evaluator import evaluate_classifier
import database
from utils import get_logger, ModelTrainingError, VectorizationError

logger = get_logger("ModelTrainer")

DATA_PATH = os.path.join(os.path.dirname(__file__), "sample_data", "real_vs_fake.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


class FakeNewsTrainer:
    """
    Orchestrates the complete NLP Feature Vectorization and Multi-Algorithm Training.
    """

    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.preprocessor = TextPreprocessor(remove_stopwords=True, use_stemming=True)
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=2)
        self.models = {
            "Logistic Regression": LogisticRegression(C=1.5, max_iter=1000, random_state=42),
            "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=25, random_state=42),
            "Passive Aggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=42, C=0.5)
        }
        self.trained_pipelines = {}
        self.evaluation_results = {}
        os.makedirs(MODELS_DIR, exist_ok=True)

    def load_and_preprocess_data(self, csv_path: str = DATA_PATH) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads dataset and applies NLP batch cleaning."""
        if not os.path.exists(csv_path):
            raise ModelTrainingError(f"Benchmark dataset not found at {csv_path}. Run generate_dataset.py first.")

        logger.info(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        if "text" not in df.columns or "label" not in df.columns:
            raise ModelTrainingError("Dataset missing required 'text' or 'label' columns.")

        logger.info("Applying NLP preprocessing across all articles...")
        df["cleaned_text"] = self.preprocessor.batch_preprocess(df["text"].tolist())
        # Filter out completely empty cleaned rows
        df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)
        return df, df["label"]

    def train_and_evaluate(self) -> Dict[str, Any]:
        """
        Executes TF-IDF feature extraction, trains all 4 classifiers, evaluates metrics,
        and saves trained models to disk.
        """
        df, y = self.load_and_preprocess_data()
        X_text = df["cleaned_text"]

        logger.info(f"Splitting {len(X_text)} documents into 80/20 Train-Test split...")
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X_text, y, test_size=0.20, random_state=42, stratify=y
        )

        try:
            logger.info(f"Fitting TF-IDF Vectorizer (max_features={self.max_features}, ngrams={self.ngram_range})...")
            X_train_tfidf = self.vectorizer.fit_transform(X_train_text)
            X_test_tfidf = self.vectorizer.transform(X_test_text)
            logger.info(f"TF-IDF Matrix Shape: {X_train_tfidf.shape}")

            # Save fitted TF-IDF vectorizer
            vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
            joblib.dump(self.vectorizer, vec_path)
            logger.info(f"Fitted TF-IDF Vectorizer saved to: {vec_path}")
        except Exception as e:
            raise VectorizationError(f"TF-IDF vectorization failed: {str(e)}") from e

        # Train and Evaluate each model
        for name, model in self.models.items():
            logger.info(f"Training classifier: {name}...")
            model.fit(X_train_tfidf, y_train)

            # Predict and evaluate
            y_pred = model.predict(X_test_tfidf)
            y_proba = None
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_tfidf)[:, 1]
            elif hasattr(model, "decision_function"):
                # Approximate probability via sigmoid for SVM / PassiveAggressive
                df_scores = model.decision_function(X_test_tfidf)
                y_proba = 1 / (1 + np.exp(-df_scores))

            metrics = evaluate_classifier(y_test.to_numpy(), y_pred, y_proba)
            self.evaluation_results[name] = metrics

            # Save model to disk
            safe_name = name.lower().replace(" ", "_") + ".pkl"
            model_path = os.path.join(MODELS_DIR, safe_name)
            joblib.dump(model, model_path)
            logger.info(f"Model [{name}] saved to {model_path}")

            # Persist evaluation metrics to SQLite relational schema
            database.save_model_metadata(
                model_name=name,
                algorithm=str(type(model).__name__),
                vectorizer=f"TF-IDF ({self.ngram_range[0]}-{self.ngram_range[1]} grams)",
                metrics=metrics
            )

        logger.info("All models trained, evaluated, and serialized successfully!")
        return self.evaluation_results


def load_trained_pipeline(model_name: str = "Logistic Regression") -> Tuple[Any, Any]:
    """Loads the serialized TF-IDF vectorizer and requested classifier from disk."""
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    safe_name = model_name.lower().replace(" ", "_") + ".pkl"
    model_path = os.path.join(MODELS_DIR, safe_name)

    if not os.path.exists(vec_path) or not os.path.exists(model_path):
        logger.warning("Saved model artifacts not found on disk. Executing automated training right now...")
        trainer = FakeNewsTrainer()
        trainer.train_and_evaluate()

    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)
    return vectorizer, model


if __name__ == "__main__":
    trainer = FakeNewsTrainer()
    results = trainer.train_and_evaluate()
    print("\nFINAL MODEL LEADERBOARD:")
    for name, m in results.items():
        print(f"  {name:25s} | Acc: {m['accuracy']*100:.2f}% | F1: {m['f1']*100:.2f}% | AUC: {m.get('roc_auc', {}).get('auc', 0):.3f}")
