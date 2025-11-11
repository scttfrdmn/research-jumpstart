#!/usr/bin/env python3
"""
Sentiment Analysis and Text Processing Utilities

Tools for social media text analysis: sentiment classification,
emotion detection, topic extraction, and text preprocessing.

Usage:
    from sentiment_utils import analyze_sentiment, extract_topics

    sentiment_scores = analyze_sentiment(texts)
    topics = extract_topics(texts, n_topics=5)
"""

import re
import numpy as np
import pandas as pd
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF


# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


def analyze_sentiment(texts, method='vader', return_detailed=False):
    """
    Analyze sentiment of texts.

    Parameters:
    -----------
    texts : list of str
        Input texts
    method : str
        Sentiment analysis method ('vader' for rule-based)
    return_detailed : bool
        If True, return all VADER scores (pos, neg, neu, compound)
        If False, return only compound score

    Returns:
    --------
    sentiments : list or DataFrame
        Sentiment scores
    """
    analyzer = SentimentIntensityAnalyzer()

    results = []
    for text in texts:
        scores = analyzer.polarity_scores(text)
        results.append(scores)

    if return_detailed:
        return pd.DataFrame(results)
    else:
        return [r['compound'] for r in results]


def classify_sentiment(compound_score, threshold_pos=0.05, threshold_neg=-0.05):
    """
    Classify sentiment as positive, negative, or neutral.

    Parameters:
    -----------
    compound_score : float
        VADER compound score (-1 to 1)
    threshold_pos : float
        Threshold for positive sentiment
    threshold_neg : float
        Threshold for negative sentiment

    Returns:
    --------
    label : str
        'positive', 'negative', or 'neutral'
    """
    if compound_score >= threshold_pos:
        return 'positive'
    elif compound_score <= threshold_neg:
        return 'negative'
    else:
        return 'neutral'


def preprocess_text(text, remove_urls=True, remove_mentions=True,
                   remove_hashtags=False, lowercase=True,
                   remove_punctuation=True, remove_stopwords=True,
                   min_word_length=2):
    """
    Preprocess social media text.

    Parameters:
    -----------
    text : str
        Input text
    remove_urls : bool
        Remove URLs
    remove_mentions : bool
        Remove @mentions
    remove_hashtags : bool
        Remove #hashtags (keep text if False)
    lowercase : bool
        Convert to lowercase
    remove_punctuation : bool
        Remove punctuation
    remove_stopwords : bool
        Remove common stopwords
    min_word_length : int
        Minimum word length to keep

    Returns:
    --------
    cleaned_text : str
        Preprocessed text
    """
    # Remove URLs
    if remove_urls:
        text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove mentions
    if remove_mentions:
        text = re.sub(r'@\w+', '', text)

    # Remove or clean hashtags
    if remove_hashtags:
        text = re.sub(r'#\w+', '', text)
    else:
        text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag text

    # Lowercase
    if lowercase:
        text = text.lower()

    # Remove punctuation
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', ' ', text)

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]

    # Filter by length
    words = [w for w in words if len(w) >= min_word_length]

    return ' '.join(words)


def extract_hashtags(text):
    """
    Extract hashtags from text.

    Parameters:
    -----------
    text : str
        Input text

    Returns:
    --------
    hashtags : list of str
        Extracted hashtags (without #)
    """
    return re.findall(r'#(\w+)', text)


def extract_mentions(text):
    """
    Extract @mentions from text.

    Parameters:
    -----------
    text : str
        Input text

    Returns:
    --------
    mentions : list of str
        Extracted mentions (without @)
    """
    return re.findall(r'@(\w+)', text)


def extract_topics(texts, n_topics=5, method='lda', n_words=10,
                  max_features=1000, min_df=2):
    """
    Extract topics from text corpus using topic modeling.

    Parameters:
    -----------
    texts : list of str
        Input texts
    n_topics : int
        Number of topics to extract
    method : str
        Topic modeling method: 'lda' (Latent Dirichlet Allocation)
        or 'nmf' (Non-negative Matrix Factorization)
    n_words : int
        Number of top words per topic
    max_features : int
        Maximum vocabulary size
    min_df : int
        Minimum document frequency for words

    Returns:
    --------
    topics : list of lists
        Top words for each topic
    """
    # Vectorize texts
    if method == 'lda':
        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            stop_words='english'
        )
        doc_term_matrix = vectorizer.fit_transform(texts)

        # LDA
        model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20
        )
        model.fit(doc_term_matrix)

    elif method == 'nmf':
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            stop_words='english'
        )
        doc_term_matrix = vectorizer.fit_transform(texts)

        # NMF
        model = NMF(
            n_components=n_topics,
            random_state=42,
            max_iter=200
        )
        model.fit(doc_term_matrix)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Extract top words per topic
    feature_names = vectorizer.get_feature_names_out()
    topics = []

    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topics.append(top_words)

    return topics


def compute_text_statistics(texts):
    """
    Compute basic text statistics.

    Parameters:
    -----------
    texts : list of str
        Input texts

    Returns:
    --------
    stats : dict
        Text statistics
    """
    word_counts = [len(text.split()) for text in texts]
    char_counts = [len(text) for text in texts]

    stats = {
        'num_texts': len(texts),
        'avg_words': np.mean(word_counts),
        'std_words': np.std(word_counts),
        'min_words': np.min(word_counts),
        'max_words': np.max(word_counts),
        'avg_chars': np.mean(char_counts),
        'total_words': sum(word_counts)
    }

    return stats


def get_word_frequencies(texts, top_n=20, remove_stopwords=True):
    """
    Get most common words in corpus.

    Parameters:
    -----------
    texts : list of str
        Input texts
    top_n : int
        Number of top words to return
    remove_stopwords : bool
        Whether to remove stopwords

    Returns:
    --------
    frequencies : list of tuples
        (word, count) pairs
    """
    # Combine and tokenize
    all_words = []
    for text in texts:
        words = word_tokenize(text.lower())
        all_words.extend(words)

    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        all_words = [w for w in all_words if w.isalnum() and w not in stop_words]

    # Count
    counter = Counter(all_words)
    return counter.most_common(top_n)


def detect_trending_hashtags(texts, time_window='1D', top_n=10):
    """
    Detect trending hashtags over time.

    Parameters:
    -----------
    texts : list of str or DataFrame
        If DataFrame, must have 'text' and 'timestamp' columns
    time_window : str
        Time window for aggregation (e.g., '1H', '1D')
    top_n : int
        Number of top hashtags per window

    Returns:
    --------
    trending : DataFrame
        Hashtag trends over time
    """
    if isinstance(texts, pd.DataFrame):
        df = texts.copy()
    else:
        # Assume texts is list
        df = pd.DataFrame({'text': texts})

    # Extract hashtags
    df['hashtags'] = df['text'].apply(extract_hashtags)

    # Explode hashtags
    df_exploded = df.explode('hashtags')

    if 'timestamp' in df_exploded.columns:
        # Group by time window
        df_exploded['time_window'] = pd.to_datetime(df_exploded['timestamp']).dt.floor(time_window)
        trending = df_exploded.groupby(['time_window', 'hashtags']).size().reset_index(name='count')
        trending = trending.sort_values(['time_window', 'count'], ascending=[True, False])

        # Get top N per window
        trending = trending.groupby('time_window').head(top_n)
    else:
        # No timestamp, just return overall top
        trending = df_exploded['hashtags'].value_counts().head(top_n).reset_index()
        trending.columns = ['hashtags', 'count']

    return trending


def analyze_sentiment_by_topic(texts, sentiments, n_topics=5):
    """
    Analyze sentiment distribution across topics.

    Parameters:
    -----------
    texts : list of str
        Input texts
    sentiments : list of float
        Sentiment scores (compound scores)
    n_topics : int
        Number of topics

    Returns:
    --------
    topic_sentiments : DataFrame
        Average sentiment per topic
    """
    # Extract topics
    topics_words = extract_topics(texts, n_topics=n_topics)

    # Get topic assignments (simplified: assign to topic with highest word overlap)
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
    doc_topics = lda.fit_transform(doc_term_matrix)

    # Assign each document to dominant topic
    topic_assignments = doc_topics.argmax(axis=1)

    # Compute average sentiment per topic
    df = pd.DataFrame({
        'topic': topic_assignments,
        'sentiment': sentiments
    })

    topic_sentiments = df.groupby('topic')['sentiment'].agg(['mean', 'std', 'count']).reset_index()
    topic_sentiments['topic_words'] = [', '.join(topics_words[i][:5]) for i in range(n_topics)]

    return topic_sentiments


def detect_emotional_content(texts):
    """
    Detect emotional content using NRC Emotion Lexicon categories.

    Simplified version using keyword matching.

    Parameters:
    -----------
    texts : list of str
        Input texts

    Returns:
    --------
    emotions : DataFrame
        Emotion scores per text
    """
    # Simplified emotion keywords
    emotion_keywords = {
        'joy': ['happy', 'joy', 'excited', 'love', 'wonderful', 'great', 'amazing'],
        'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'lonely', 'grief'],
        'anger': ['angry', 'mad', 'furious', 'outraged', 'hate', 'annoyed'],
        'fear': ['afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous'],
        'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected'],
        'trust': ['trust', 'confidence', 'faith', 'reliable', 'dependable']
    }

    results = []
    for text in texts:
        text_lower = text.lower()
        emotion_scores = {}

        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = score

        results.append(emotion_scores)

    return pd.DataFrame(results)


if __name__ == '__main__':
    # Example usage
    print("Sentiment Analysis and Text Processing Utilities")
    print("=" * 60)

    # Sample social media texts
    sample_texts = [
        "I love this product! Best purchase ever! #happy #recommended",
        "Terrible experience. Very disappointed. @company please fix this.",
        "Just okay, nothing special. Could be better.",
        "Amazing service! Will definitely come back! 😊",
        "The worst. Complete waste of money. #angry",
        "Not sure how I feel about this... mixed feelings.",
        "Absolutely fantastic! Exceeded all expectations! #love",
        "Meh. Pretty average. Nothing to write home about."
    ]

    # Sentiment analysis
    print("\n1. Analyzing sentiment...")
    sentiments = analyze_sentiment(sample_texts)
    for i, (text, score) in enumerate(zip(sample_texts, sentiments)):
        label = classify_sentiment(score)
        print(f"   Text {i+1} ({label}): {score:.3f}")

    # Text preprocessing
    print("\n2. Preprocessing text...")
    sample = sample_texts[0]
    print(f"   Original: {sample}")
    cleaned = preprocess_text(sample)
    print(f"   Cleaned:  {cleaned}")

    # Extract hashtags and mentions
    print("\n3. Extracting hashtags and mentions...")
    hashtags = extract_hashtags(sample_texts[0])
    mentions = extract_mentions(sample_texts[1])
    print(f"   Hashtags: {hashtags}")
    print(f"   Mentions: {mentions}")

    # Word frequencies
    print("\n4. Computing word frequencies...")
    frequencies = get_word_frequencies(sample_texts, top_n=10)
    print("   Top 10 words:")
    for word, count in frequencies[:5]:
        print(f"     {word}: {count}")

    # Text statistics
    print("\n5. Text statistics...")
    stats = compute_text_statistics(sample_texts)
    print(f"   Avg words per text: {stats['avg_words']:.1f}")
    print(f"   Total words: {stats['total_words']}")

    # Topic extraction (requires more texts for meaningful results)
    print("\n6. Topic extraction...")
    # Duplicate texts to have enough for topic modeling
    expanded_texts = sample_texts * 10
    topics = extract_topics(expanded_texts, n_topics=2, n_words=5)
    for i, topic_words in enumerate(topics):
        print(f"   Topic {i+1}: {', '.join(topic_words)}")

    print("\n✓ Sentiment analysis utilities ready")
