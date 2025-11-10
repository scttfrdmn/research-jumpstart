"""Text analysis functions for digital humanities research."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter
import re

logger = logging.getLogger(__name__)


def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze sentiment of text (using simple lexicon-based approach).

    Args:
        text: Input text

    Returns:
        Dictionary with sentiment scores
    """
    # Simple sentiment lexicon (positive/negative words)
    positive_words = {
        'good', 'great', 'excellent', 'wonderful', 'beautiful', 'love', 'happy',
        'joy', 'amazing', 'perfect', 'best', 'brilliant', 'fantastic', 'superb'
    }
    negative_words = {
        'bad', 'terrible', 'horrible', 'awful', 'hate', 'sad', 'angry', 'worst',
        'poor', 'disappointing', 'disgusting', 'dreadful', 'miserable', 'pathetic'
    }

    words = text.lower().split()
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)

    total = len(words)
    if total == 0:
        return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}

    positive_score = positive_count / total
    negative_score = negative_count / total
    neutral_score = 1.0 - (positive_score + negative_score)

    # Compound score (-1 to 1)
    compound = (positive_count - negative_count) / max(total, 1)

    return {
        'positive': positive_score,
        'negative': negative_score,
        'neutral': neutral_score,
        'compound': compound
    }


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities from text (simple pattern-based).

    Args:
        text: Input text

    Returns:
        Dictionary with entity types and lists of entities
    """
    entities = {
        'PERSON': [],
        'LOCATION': [],
        'ORGANIZATION': [],
        'DATE': []
    }

    # Simple patterns for demonstration
    # In production, use spaCy or AWS Comprehend

    # Extract capitalized phrases (potential names)
    name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
    potential_names = re.findall(name_pattern, text)
    entities['PERSON'] = list(set(potential_names))

    # Extract dates
    date_pattern = r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
    dates = re.findall(date_pattern, text)
    entities['DATE'] = list(set(dates))

    logger.info(f"Extracted {sum(len(v) for v in entities.values())} entities")
    return entities


def extract_keywords(text: str, top_n: int = 10) -> List[Tuple[str, int]]:
    """
    Extract most frequent keywords from text.

    Args:
        text: Input text
        top_n: Number of top keywords to return

    Returns:
        List of (keyword, frequency) tuples
    """
    # Common English stopwords
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'it', 'its', 'they', 'their', 'them', 'he', 'she'
    }

    # Tokenize and clean
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())

    # Remove stopwords
    filtered_words = [word for word in words if word not in stopwords]

    # Count frequencies
    word_counts = Counter(filtered_words)

    # Get top N
    top_keywords = word_counts.most_common(top_n)

    logger.info(f"Extracted {len(top_keywords)} keywords")
    return top_keywords


def calculate_readability(text: str) -> Dict[str, float]:
    """
    Calculate readability metrics for text.

    Args:
        text: Input text

    Returns:
        Dictionary with readability scores
    """
    # Count sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = len(sentences)

    # Count words
    words = re.findall(r'\b\w+\b', text)
    num_words = len(words)

    # Count syllables (simple approximation)
    def count_syllables(word):
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        # Adjust for silent e
        if word.endswith('e'):
            syllable_count -= 1

        # Ensure at least one syllable
        return max(1, syllable_count)

    num_syllables = sum(count_syllables(word) for word in words)

    if num_sentences == 0 or num_words == 0:
        return {
            'flesch_reading_ease': 0.0,
            'flesch_kincaid_grade': 0.0,
            'avg_words_per_sentence': 0.0,
            'avg_syllables_per_word': 0.0
        }

    # Flesch Reading Ease
    avg_sentence_length = num_words / num_sentences
    avg_syllables_per_word = num_syllables / num_words
    flesch_reading_ease = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word

    # Flesch-Kincaid Grade Level
    flesch_kincaid_grade = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59

    return {
        'flesch_reading_ease': flesch_reading_ease,
        'flesch_kincaid_grade': flesch_kincaid_grade,
        'avg_words_per_sentence': avg_sentence_length,
        'avg_syllables_per_word': avg_syllables_per_word
    }


def detect_language(text: str) -> str:
    """
    Detect language of text (simple heuristic based on common words).

    Args:
        text: Input text

    Returns:
        Detected language code
    """
    # Simple language detection based on common words
    english_words = {'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that'}
    spanish_words = {'el', 'la', 'de', 'que', 'y', 'en', 'un', 'ser', 'se', 'no'}
    french_words = {'le', 'de', 'un', 'être', 'et', 'à', 'il', 'avoir', 'ne', 'je'}
    german_words = {'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich'}

    words = set(text.lower().split()[:100])  # Check first 100 words

    scores = {
        'en': len(words & english_words),
        'es': len(words & spanish_words),
        'fr': len(words & french_words),
        'de': len(words & german_words)
    }

    detected = max(scores, key=scores.get)
    logger.info(f"Detected language: {detected}")
    return detected


def perform_topic_modeling(
    corpus_df: pd.DataFrame,
    text_column: str = 'text',
    n_topics: int = 5,
    n_words: int = 10
) -> Dict:
    """
    Perform simple topic modeling using word frequency analysis.

    Args:
        corpus_df: DataFrame with text documents
        text_column: Column name containing text
        n_topics: Number of topics to extract
        n_words: Number of words per topic

    Returns:
        Dictionary with topic information
    """
    logger.info(f"Performing topic modeling with {n_topics} topics")

    # Simple approach: cluster by most distinctive words
    # In production, use LDA or other advanced methods

    all_words = []
    for text in corpus_df[text_column]:
        keywords = extract_keywords(text, top_n=20)
        all_words.extend([word for word, _ in keywords])

    # Get most common words across corpus
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(n_topics * n_words)

    # Distribute into topics (simple distribution)
    topics = {}
    for i in range(n_topics):
        topic_words = top_words[i*n_words:(i+1)*n_words]
        topics[f'Topic_{i+1}'] = [word for word, _ in topic_words]

    logger.info(f"Generated {len(topics)} topics")
    return topics


def analyze_corpus_statistics(corpus_df: pd.DataFrame, text_column: str = 'text') -> Dict:
    """
    Generate comprehensive statistics for text corpus.

    Args:
        corpus_df: DataFrame with text documents
        text_column: Column name containing text

    Returns:
        Dictionary with corpus statistics
    """
    logger.info("Calculating corpus statistics")

    stats = {
        'total_documents': len(corpus_df),
        'total_words': 0,
        'avg_document_length': 0,
        'vocabulary_size': 0,
        'avg_readability_score': 0
    }

    all_words = []
    total_words = 0
    readability_scores = []

    for text in corpus_df[text_column]:
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
        total_words += len(words)

        # Calculate readability
        readability = calculate_readability(text)
        readability_scores.append(readability['flesch_reading_ease'])

    stats['total_words'] = total_words
    stats['avg_document_length'] = total_words / len(corpus_df) if len(corpus_df) > 0 else 0
    stats['vocabulary_size'] = len(set(all_words))
    stats['avg_readability_score'] = np.mean(readability_scores) if readability_scores else 0

    logger.info(f"Corpus statistics: {stats}")
    return stats


def compare_texts(text1: str, text2: str) -> Dict[str, float]:
    """
    Compare two texts for similarity.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Dictionary with similarity metrics
    """
    # Extract word sets
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))

    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    jaccard = intersection / union if union > 0 else 0

    # Overlap coefficient
    smaller_set_size = min(len(words1), len(words2))
    overlap = intersection / smaller_set_size if smaller_set_size > 0 else 0

    # Dice coefficient
    dice = (2 * intersection) / (len(words1) + len(words2)) if (len(words1) + len(words2)) > 0 else 0

    return {
        'jaccard_similarity': jaccard,
        'overlap_coefficient': overlap,
        'dice_coefficient': dice,
        'unique_to_text1': len(words1 - words2),
        'unique_to_text2': len(words2 - words1),
        'shared_words': intersection
    }


def extract_ngrams(text: str, n: int = 2, top_k: int = 10) -> List[Tuple[str, int]]:
    """
    Extract most frequent n-grams from text.

    Args:
        text: Input text
        n: Size of n-grams (default: 2 for bigrams)
        top_k: Number of top n-grams to return

    Returns:
        List of (ngram, frequency) tuples
    """
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())

    # Generate n-grams
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)

    # Count frequencies
    ngram_counts = Counter(ngrams)
    top_ngrams = ngram_counts.most_common(top_k)

    logger.info(f"Extracted {len(top_ngrams)} {n}-grams")
    return top_ngrams
