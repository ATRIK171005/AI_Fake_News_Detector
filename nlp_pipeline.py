"""
nlp_pipeline.py
---------------
Advanced Natural Language Processing (NLP) text preprocessing and cleaning pipeline.
Transforms raw news articles and headlines into clean, tokenized, and normalized strings
ready for TF-IDF feature extraction.
"""

import re
import string
from typing import List, Set
from utils import get_logger, PreprocessingError

logger = get_logger("NLP_Pipeline")

# Comprehensive built-in English stopword set (ensures offline reliability without downloading dependencies)
ENGLISH_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing",
    "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
    "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
    "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves", "said", "says", "also", "would", "could",
    "one", "two", "new", "time", "year", "years", "people"
}


class TextPreprocessor:
    """
    Production NLP Preprocessing class performing multi-stage cleaning:
    1. HTML tag removal
    2. URL & Email address stripping
    3. Punctuation and special character stripping
    4. Lowercasing & whitespace normalization
    5. Stopword filtering & stemming/lemmatization simulation
    """

    def __init__(self, remove_stopwords: bool = True, use_stemming: bool = True):
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        # Simple Porter-like suffix rules for fast clean stemming
        self.suffixes = [("ing", ""), ("ed", ""), ("es", ""), ("s", "")]
        logger.info(f"TextPreprocessor initialized (remove_stopwords={remove_stopwords}, use_stemming={use_stemming})")

    def _clean_stem(self, word: str) -> str:
        """Lightweight heuristic stemmer ensuring zero external dependency crashes."""
        if len(word) <= 3:
            return word
        for suffix, replacement in self.suffixes:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[:-len(suffix)] + replacement
        return word

    def clean_text(self, text: str) -> str:
        """
        Cleans raw article text or headline into a normalized string.
        """
        if not isinstance(text, str):
            if text is None:
                return ""
            text = str(text)

        try:
            # 1. Lowercase
            cleaned = text.lower()

            # 2. Strip HTML tags
            cleaned = re.sub(r'<.*?>', ' ', cleaned)

            # 3. Strip URLs & Emails
            cleaned = re.sub(r'http\S+|www\S+|https\S+', ' ', cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r'\S+@\S+', ' ', cleaned)

            # 4. Strip punctuation, numbers, and special characters
            cleaned = re.sub(r'[^a-zA-Z\s]', ' ', cleaned)

            # 5. Tokenize and filter
            tokens = cleaned.split()

            if self.remove_stopwords:
                tokens = [t for t in tokens if t not in ENGLISH_STOPWORDS and len(t) > 2]

            if self.use_stemming:
                tokens = [self._clean_stem(t) for t in tokens]

            return " ".join(tokens)

        except Exception as e:
            logger.error(f"Error during NLP cleaning: {str(e)}")
            raise PreprocessingError(f"Failed to preprocess text: {str(e)}") from e

    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """
        Preprocesses a list/Series of text documents.
        """
        logger.info(f"Batch preprocessing {len(texts)} documents...")
        cleaned_list = [self.clean_text(doc) for doc in texts]
        logger.info("Batch preprocessing completed.")
        return cleaned_list


def extract_text_statistics(text: str) -> dict:
    """
    Extracts surface-level linguistic features from raw text for exploratory data analysis.
    """
    words = text.split()
    char_count = len(text)
    word_count = len(words)
    avg_word_len = round(sum(len(w) for w in words) / max(1, word_count), 2)
    uppercase_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    exclamation_count = text.count('!')
    question_count = text.count('?')

    return {
        "Character Count": char_count,
        "Word Count": word_count,
        "Avg Word Length": avg_word_len,
        "All-Caps Words": uppercase_words,
        "Exclamation Marks (!)": exclamation_count,
        "Question Marks (?)": question_count
    }
