"""
summarizer.py
Core NLP logic for the Automated Text Summarization System.

Two summarization modes:
1. TF-IDF extractive summarization (fast, lightweight, works everywhere -
   this is what powers the deployed web app)
2. BERT-based extractive summarization (optional, heavier - demoed in the
   Colab notebook using sentence embeddings; see colab/ for that experiment)
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.data.find("tokenizers/punkt")
except (ImportError, LookupError):
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize
    except Exception:
        sent_tokenize = None


def simple_sentence_tokenize(text):
    """Fallback sentence splitter if NLTK punkt isn't available."""
    text = text.replace("\n", " ")
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def tokenize_sentences(text):
    if sent_tokenize:
        try:
            return sent_tokenize(text)
        except Exception:
            return simple_sentence_tokenize(text)
    return simple_sentence_tokenize(text)


def clean_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text


def summarize_tfidf(text, num_sentences=3):
    """
    Extractive summarization using TF-IDF sentence scoring.
    Each sentence is scored by the sum of TF-IDF weights of its words;
    the top-N highest scoring sentences (in original order) form the summary.
    """
    text = clean_text(text)
    sentences = tokenize_sentences(text)

    if len(sentences) <= num_sentences:
        return text, sentences

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Score each sentence = sum of its TF-IDF weights
    scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()

    # Pick top-N sentence indices, then restore original order
    top_indices = scores.argsort()[::-1][:num_sentences]
    top_indices_sorted = sorted(top_indices)

    summary_sentences = [sentences[i] for i in top_indices_sorted]
    summary = " ".join(summary_sentences)
    return summary, sentences


def get_stats(original_text, summary_text):
    orig_words = len(original_text.split())
    summ_words = len(summary_text.split())
    reduction = round((1 - summ_words / orig_words) * 100, 1) if orig_words else 0
    return {
        "original_words": orig_words,
        "summary_words": summ_words,
        "reduction_percent": reduction,
    }
