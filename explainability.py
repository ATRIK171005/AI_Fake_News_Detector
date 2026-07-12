"""
explainability.py
-----------------
Automated Decision Explainability engine for the AI Fake News Detector.
Extracts top feature contributions (words pushing toward Fake vs. Real classification)
by combining TF-IDF document weights with learned classifier coefficients.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from utils import get_logger

logger = get_logger("DecisionExplainer")


def explain_prediction(
    text: str,
    vectorizer: Any,
    model: Any,
    top_k: int = 8
) -> Dict[str, Any]:
    """
    Computes local explainability for a single text document.
    Returns:
      - fake_contributors: List of (word, score) pushing toward Fake News (Label 1)
      - real_contributors: List of (word, score) pushing toward Real News (Label 0)
    """
    try:
        # Transform single text into TF-IDF sparse matrix
        tfidf_matrix = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()

        # Extract non-zero TF-IDF values for this document
        nonzero_indices = tfidf_matrix.nonzero()[1]
        tfidf_values = tfidf_matrix.data

        if len(nonzero_indices) == 0:
            return {"fake_contributors": [], "real_contributors": [], "explanation_summary": "No recognizable vocabulary tokens found in input text."}

        # Determine feature weights from model
        feature_weights = np.zeros(len(feature_names))
        if hasattr(model, "coef_"):
            # Logistic Regression / Linear SVM / Passive Aggressive
            feature_weights = model.coef_[0]
        elif hasattr(model, "feature_log_prob_"):
            # Multinomial Naive Bayes (log odds: log P(word|Fake) - log P(word|Real))
            feature_weights = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        elif hasattr(model, "feature_importances_"):
            # Random Forest / Decision Trees (feature importance weighted by direction of prediction)
            feature_weights = model.feature_importances_
            # If tree predicted fake, weight positive, else negative
            pred_class = model.predict(tfidf_matrix)[0]
            direction = 1.0 if pred_class == 1 else -1.0
            feature_weights = feature_weights * direction
        else:
            return {"fake_contributors": [], "real_contributors": [], "explanation_summary": "Model does not support coefficient extraction."}

        # Calculate exact contribution for each word present in the text
        word_contributions = []
        for idx, tfidf_val in zip(nonzero_indices, tfidf_values):
            word = feature_names[idx]
            weight = feature_weights[idx]
            contribution = float(tfidf_val * weight)
            word_contributions.append((word, contribution, float(tfidf_val)))

        # Sort by contribution
        word_contributions.sort(key=lambda x: x[1], reverse=True)

        fake_contributors = [(w, round(c, 4)) for w, c, _ in word_contributions if c > 0][:top_k]
        # Real contributors have negative scores, sort by most negative
        real_contributors = [(w, round(abs(c), 4)) for w, c, _ in sorted(word_contributions, key=lambda x: x[1]) if c < 0][:top_k]

        pred_label = model.predict(tfidf_matrix)[0]
        label_name = "⚠️ FAKE NEWS" if pred_label == 1 else "✅ REAL NEWS"

        if pred_label == 1 and fake_contributors:
            top_words = ", ".join([f"'{w}'" for w, _ in fake_contributors[:3]])
            summary = f"Classified as {label_name}. Key vocabulary driving this prediction includes sensational/hoax indicators like: {top_words}."
        elif pred_label == 0 and real_contributors:
            top_words = ", ".join([f"'{w}'" for w, _ in real_contributors[:3]])
            summary = f"Classified as {label_name}. Key vocabulary supporting authenticity includes institutional/factual terms like: {top_words}."
        else:
            summary = f"Classified as {label_name} based on overall syntactic structure and TF-IDF distribution."

        return {
            "fake_contributors": fake_contributors,
            "real_contributors": real_contributors,
            "explanation_summary": summary
        }

    except Exception as e:
        logger.error(f"Error computing explainability: {str(e)}")
        return {"fake_contributors": [], "real_contributors": [], "explanation_summary": f"Could not generate LIME/coefficient rationale: {str(e)}"}
