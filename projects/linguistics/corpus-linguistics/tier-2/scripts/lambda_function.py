"""
Lambda Function for Corpus Linguistics Analysis

This function performs automated linguistic analysis on text files uploaded to S3.
It extracts linguistic features including:
- Tokenization and sentence segmentation
- POS (Part-of-Speech) tagging
- Lemmatization
- Word frequency analysis
- N-gram extraction (bigrams, trigrams)
- Collocation detection
- Lexical diversity metrics (TTR, MATTR)
- Syntactic complexity measures

Results are stored in DynamoDB for fast querying.
"""

import json
import os
import re
from collections import Counter
from urllib.parse import unquote_plus

import boto3
import nltk
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.collocations import (
    BigramAssocMeasures,
    BigramCollocationFinder,
    TrigramAssocMeasures,
    TrigramCollocationFinder,
)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize AWS clients
s3_client = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

# Environment variables
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "LinguisticAnalysis")
NLTK_DATA_PATH = os.environ.get("NLTK_DATA", "/var/task/nltk_data")

# Set NLTK data path for Lambda
if NLTK_DATA_PATH:
    nltk.data.path.append(NLTK_DATA_PATH)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def lambda_handler(event, context):
    """
    Main Lambda handler function.
    Triggered by S3 upload event.

    Args:
        event: S3 event notification
        context: Lambda context

    Returns:
        dict: Response with status code and message
    """
    try:
        # Get S3 bucket and object key from event
        record = event["Records"][0]
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])

        print(f"Processing file: s3://{bucket}/{key}")

        # Download text file from S3
        response = s3_client.get_object(Bucket=bucket, Key=key)
        text = response["Body"].read().decode("utf-8")

        # Extract metadata from S3 key
        metadata = extract_metadata_from_key(key)
        text_id = metadata["text_id"]

        print(f"Text ID: {text_id}")
        print(f"Text length: {len(text)} characters")

        # Perform linguistic analysis
        analysis_results = analyze_text(text, metadata)

        # Store results in DynamoDB
        store_results(text_id, analysis_results)

        print(f"Successfully processed {text_id}")

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Successfully processed text",
                    "text_id": text_id,
                    "word_count": analysis_results["word_count"],
                }
            ),
        }

    except Exception as e:
        print(f"Error processing text: {e!s}")
        import traceback

        traceback.print_exc()

        return {
            "statusCode": 500,
            "body": json.dumps({"message": "Error processing text", "error": str(e)}),
        }


def extract_metadata_from_key(key):
    """
    Extract metadata from S3 object key.
    Expected format: raw/{language}/{genre}/{filename}.txt

    Args:
        key: S3 object key

    Returns:
        dict: Metadata (language, genre, filename, text_id)
    """
    parts = key.split("/")

    if len(parts) >= 4 and parts[0] == "raw":
        language = parts[1]
        genre = parts[2]
        filename = parts[-1].replace(".txt", "")
    else:
        # Fallback for different structure
        language = "unknown"
        genre = "unknown"
        filename = parts[-1].replace(".txt", "")

    text_id = f"{language}_{genre}_{filename}".replace(" ", "_").lower()

    return {
        "language": language,
        "genre": genre,
        "filename": filename,
        "text_id": text_id,
        "s3_key": key,
    }


def analyze_text(text, metadata):
    """
    Perform comprehensive linguistic analysis on text.

    Args:
        text: Input text string
        metadata: Text metadata

    Returns:
        dict: Analysis results
    """
    results = metadata.copy()

    # Basic text cleaning
    text = clean_text(text)

    # Tokenization
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    # Filter out punctuation and numbers
    words = [w for w in words if w.isalpha()]

    results["sentence_count"] = len(sentences)
    results["word_count"] = len(words)

    # POS tagging
    pos_tags = pos_tag(words)
    results["pos_distribution"] = get_pos_distribution(pos_tags)

    # Lemmatization
    lemmas = [lemmatizer.lemmatize(word) for word in words]
    results["unique_words"] = len(set(words))
    results["unique_lemmas"] = len(set(lemmas))

    # Word frequencies
    word_freq = Counter(words)
    lemma_freq = Counter(lemmas)
    results["top_words"] = word_freq.most_common(20)
    results["top_lemmas"] = lemma_freq.most_common(20)

    # Lexical diversity metrics
    results["lexical_diversity"] = calculate_lexical_diversity(words, sentences)

    # N-grams and collocations
    results["collocations"] = extract_collocations(words)

    # Syntactic complexity
    results["syntactic_complexity"] = calculate_syntactic_complexity(sentences, words)

    # Average word length
    results["avg_word_length"] = sum(len(w) for w in words) / len(words) if words else 0

    return results


def clean_text(text):
    """
    Basic text cleaning.

    Args:
        text: Input text

    Returns:
        str: Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^\w\s.,!?;:\'-]", "", text)

    return text.strip()


def get_pos_distribution(pos_tags):
    """
    Calculate POS tag distribution.

    Args:
        pos_tags: List of (word, tag) tuples

    Returns:
        dict: POS tag counts
    """
    pos_counts = Counter(tag for word, tag in pos_tags)

    # Group into major categories
    pos_categories = {
        "NOUN": sum(pos_counts.get(tag, 0) for tag in ["NN", "NNS", "NNP", "NNPS"]),
        "VERB": sum(pos_counts.get(tag, 0) for tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]),
        "ADJ": sum(pos_counts.get(tag, 0) for tag in ["JJ", "JJR", "JJS"]),
        "ADV": sum(pos_counts.get(tag, 0) for tag in ["RB", "RBR", "RBS"]),
        "PRON": sum(pos_counts.get(tag, 0) for tag in ["PRP", "PRP$", "WP", "WP$"]),
        "DET": sum(pos_counts.get(tag, 0) for tag in ["DT", "PDT", "WDT"]),
        "PREP": sum(pos_counts.get(tag, 0) for tag in ["IN"]),
        "CONJ": sum(pos_counts.get(tag, 0) for tag in ["CC"]),
    }

    return pos_categories


def calculate_lexical_diversity(words, sentences):
    """
    Calculate lexical diversity metrics.

    Args:
        words: List of words
        sentences: List of sentences

    Returns:
        dict: Lexical diversity metrics
    """
    if not words:
        return {}

    # Type-Token Ratio (TTR)
    ttr = len(set(words)) / len(words)

    # Moving Average TTR (MATTR)
    # Calculate TTR for sliding windows of 100 words
    window_size = min(100, len(words))
    if len(words) >= window_size:
        ttrs = []
        for i in range(len(words) - window_size + 1):
            window = words[i : i + window_size]
            window_ttr = len(set(window)) / len(window)
            ttrs.append(window_ttr)
        mattr = sum(ttrs) / len(ttrs) if ttrs else ttr
    else:
        mattr = ttr

    # Root TTR (RTTR)
    rttr = len(set(words)) / (len(words) ** 0.5) if len(words) > 0 else 0

    return {
        "ttr": round(ttr, 4),
        "mattr": round(mattr, 4),
        "rttr": round(rttr, 4),
        "types": len(set(words)),
        "tokens": len(words),
    }


def extract_collocations(words):
    """
    Extract bigram and trigram collocations using PMI.

    Args:
        words: List of words

    Returns:
        dict: Top collocations
    """
    if len(words) < 2:
        return {"bigrams": [], "trigrams": []}

    # Remove stopwords for collocation detection
    try:
        stop_words = set(stopwords.words("english"))
    except:
        stop_words = set()

    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]

    # Bigrams
    bigram_measures = BigramAssocMeasures()
    bigram_finder = BigramCollocationFinder.from_words(filtered_words)
    bigram_finder.apply_freq_filter(2)  # Appear at least 2 times

    top_bigrams = []
    try:
        scored_bigrams = bigram_finder.score_ngrams(bigram_measures.pmi)
        for (word1, word2), score in scored_bigrams[:10]:
            freq = bigram_finder.ngram_fd[(word1, word2)]
            top_bigrams.append({"bigram": f"{word1} {word2}", "pmi": round(score, 2), "freq": freq})
    except:
        pass

    # Trigrams
    trigram_measures = TrigramAssocMeasures()
    trigram_finder = TrigramCollocationFinder.from_words(filtered_words)
    trigram_finder.apply_freq_filter(2)

    top_trigrams = []
    try:
        scored_trigrams = trigram_finder.score_ngrams(trigram_measures.pmi)
        for (word1, word2, word3), score in scored_trigrams[:10]:
            freq = trigram_finder.ngram_fd[(word1, word2, word3)]
            top_trigrams.append(
                {"trigram": f"{word1} {word2} {word3}", "pmi": round(score, 2), "freq": freq}
            )
    except:
        pass

    return {"bigrams": top_bigrams, "trigrams": top_trigrams}


def calculate_syntactic_complexity(sentences, words):
    """
    Calculate syntactic complexity metrics.

    Args:
        sentences: List of sentences
        words: List of words

    Returns:
        dict: Syntactic complexity metrics
    """
    if not sentences:
        return {}

    # Average sentence length
    avg_sentence_length = len(words) / len(sentences)

    # Sentence length variation (standard deviation)
    sentence_lengths = [len(word_tokenize(sent)) for sent in sentences]
    avg_length = sum(sentence_lengths) / len(sentence_lengths)
    variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
    std_dev = variance**0.5

    return {
        "avg_sentence_length": round(avg_sentence_length, 2),
        "sentence_length_std": round(std_dev, 2),
        "min_sentence_length": min(sentence_lengths) if sentence_lengths else 0,
        "max_sentence_length": max(sentence_lengths) if sentence_lengths else 0,
    }


def store_results(text_id, results):
    """
    Store analysis results in DynamoDB.

    Args:
        text_id: Unique text identifier
        results: Analysis results dictionary
    """
    table = dynamodb.Table(DYNAMODB_TABLE)

    # Convert results to DynamoDB-compatible format
    item = {
        "text_id": text_id,
        "language": results.get("language", "unknown"),
        "genre": results.get("genre", "unknown"),
        "filename": results.get("filename", "unknown"),
        "s3_key": results.get("s3_key", ""),
        "word_count": results.get("word_count", 0),
        "sentence_count": results.get("sentence_count", 0),
        "unique_words": results.get("unique_words", 0),
        "unique_lemmas": results.get("unique_lemmas", 0),
        "avg_word_length": results.get("avg_word_length", 0),
        "pos_distribution": results.get("pos_distribution", {}),
        "lexical_diversity": results.get("lexical_diversity", {}),
        "syntactic_complexity": results.get("syntactic_complexity", {}),
        "top_words": [{"word": w, "freq": f} for w, f in results.get("top_words", [])[:10]],
        "top_lemmas": [{"lemma": l, "freq": f} for l, f in results.get("top_lemmas", [])[:10]],
        "collocations": results.get("collocations", {}),
    }

    # Store in DynamoDB
    table.put_item(Item=item)

    print(f"Stored results for {text_id} in DynamoDB")
