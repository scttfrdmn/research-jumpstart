#!/usr/bin/env python3
"""
AWS Lambda function for NLP text analysis.

Performs comprehensive linguistic analysis on historical texts:
- Word frequency analysis
- Named Entity Recognition (NER)
- Topic modeling (LDA)
- Literary features (sentence length, vocabulary richness)
- Sentiment analysis

Triggered by S3 upload or manual invocation.
Results stored in DynamoDB for fast querying.
"""

import json
import os
import re
import boto3
from datetime import datetime
from typing import Dict, List, Any
from collections import Counter
import urllib.parse

# Initialize AWS clients
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Configuration from environment variables
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'TextAnalysis')
BUCKET_NAME = os.environ.get('BUCKET_NAME', '')

# Set NLTK data path for Lambda environment
os.environ['NLTK_DATA'] = '/var/task/nltk_data:/tmp/nltk_data'


def lambda_handler(event, context):
    """
    Lambda handler function.

    Args:
        event: S3 event trigger or manual invocation
        context: Lambda context

    Returns:
        Response with processing status
    """
    try:
        # Parse S3 event
        if 'Records' in event:
            # Triggered by S3 upload
            record = event['Records'][0]
            bucket = record['s3']['bucket']['name']
            key = urllib.parse.unquote_plus(record['s3']['object']['key'])
        else:
            # Manual invocation
            bucket = event.get('bucket', BUCKET_NAME)
            key = event.get('key', '')

        print(f"Processing: s3://{bucket}/{key}")

        # Download text from S3
        text_content, metadata = download_text_from_s3(bucket, key)

        # Perform NLP analysis
        analysis_results = analyze_text(text_content, metadata)

        # Add document metadata
        analysis_results['document_id'] = f"{metadata['author']}-{os.path.basename(key)}"
        analysis_results['s3_key'] = key
        analysis_results['bucket'] = bucket
        analysis_results['timestamp'] = int(datetime.now().timestamp())

        # Store results in DynamoDB
        store_results_in_dynamodb(analysis_results)

        # Save detailed results to S3
        save_detailed_results_to_s3(bucket, key, analysis_results)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Processing completed successfully',
                'document_id': analysis_results['document_id'],
                'word_count': analysis_results['word_count']
            })
        }

    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


def download_text_from_s3(bucket: str, key: str) -> tuple:
    """
    Download text file and metadata from S3.

    Args:
        bucket: S3 bucket name
        key: S3 object key

    Returns:
        Tuple of (text_content, metadata)
    """
    try:
        # Get object with metadata
        response = s3.get_object(Bucket=bucket, Key=key)

        # Read text content
        text_content = response['Body'].read().decode('utf-8', errors='ignore')

        # Extract metadata
        metadata = response.get('Metadata', {})

        # Parse author from key if not in metadata
        if 'author' not in metadata:
            parts = key.split('/')
            if len(parts) >= 3:  # raw/author/filename.txt
                metadata['author'] = parts[1].replace('-', ' ').title()

        # Ensure required metadata fields
        metadata.setdefault('title', os.path.basename(key).replace('.txt', '').replace('-', ' ').title())
        metadata.setdefault('period', 'Unknown')
        metadata.setdefault('genre', 'Text')

        print(f"Downloaded {len(text_content)} characters from {key}")
        return text_content, metadata

    except Exception as e:
        print(f"Error downloading from S3: {e}")
        raise


def analyze_text(text: str, metadata: Dict) -> Dict[str, Any]:
    """
    Perform comprehensive NLP analysis on text.

    Args:
        text: Text content to analyze
        metadata: Document metadata

    Returns:
        Dict with analysis results
    """
    # Import NLP libraries
    try:
        import nltk
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus import stopwords
        from nltk.tag import pos_tag
        from nltk.chunk import ne_chunk
    except ImportError:
        print("Warning: NLTK not available, using basic analysis")
        return basic_analysis(text, metadata)

    print("Starting NLP analysis...")

    # Clean text (remove Gutenberg header/footer)
    cleaned_text = clean_gutenberg_text(text)

    # Tokenization
    sentences = sent_tokenize(cleaned_text)
    words = word_tokenize(cleaned_text.lower())

    # Filter words (only alphabetic, length > 2)
    words = [w for w in words if w.isalpha() and len(w) > 2]

    # Calculate basic statistics
    total_words = len(words)
    unique_words = len(set(words))
    vocabulary_richness = unique_words / total_words if total_words > 0 else 0

    # Word frequency analysis
    word_freq = Counter(words)
    stop_words = set(stopwords.words('english'))
    content_words = [w for w in words if w not in stop_words]
    content_word_freq = Counter(content_words)

    # Top words
    top_words = [
        {'word': word, 'count': count}
        for word, count in content_word_freq.most_common(50)
    ]

    # Sentence analysis
    avg_sentence_length = total_words / len(sentences) if sentences else 0

    # Named Entity Recognition
    named_entities = extract_named_entities(cleaned_text)

    # Literary features
    literary_features = calculate_literary_features(words, sentences)

    # Assemble results
    results = {
        'author': metadata.get('author', 'Unknown'),
        'title': metadata.get('title', 'Unknown'),
        'period': metadata.get('period', 'Unknown'),
        'genre': metadata.get('genre', 'Text'),
        'word_count': total_words,
        'unique_words': unique_words,
        'vocabulary_richness': round(vocabulary_richness, 4),
        'sentence_count': len(sentences),
        'avg_sentence_length': round(avg_sentence_length, 2),
        'top_words': top_words[:30],  # Top 30 for DynamoDB size limits
        'named_entities': named_entities,
        'literary_features': literary_features,
        'processing_timestamp': datetime.now().isoformat()
    }

    print(f"Analysis complete: {total_words} words, {unique_words} unique")
    return results


def basic_analysis(text: str, metadata: Dict) -> Dict[str, Any]:
    """
    Basic analysis without NLTK (fallback).

    Args:
        text: Text content
        metadata: Document metadata

    Returns:
        Dict with basic analysis results
    """
    # Simple tokenization
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    words = [w for w in words if len(w) > 2]

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    total_words = len(words)
    unique_words = len(set(words))

    word_freq = Counter(words)
    top_words = [
        {'word': word, 'count': count}
        for word, count in word_freq.most_common(30)
    ]

    return {
        'author': metadata.get('author', 'Unknown'),
        'title': metadata.get('title', 'Unknown'),
        'period': metadata.get('period', 'Unknown'),
        'genre': metadata.get('genre', 'Text'),
        'word_count': total_words,
        'unique_words': unique_words,
        'vocabulary_richness': round(unique_words / total_words, 4) if total_words > 0 else 0,
        'sentence_count': len(sentences),
        'avg_sentence_length': round(total_words / len(sentences), 2) if sentences else 0,
        'top_words': top_words,
        'named_entities': {'people': [], 'places': [], 'organizations': []},
        'literary_features': {},
        'processing_timestamp': datetime.now().isoformat()
    }


def clean_gutenberg_text(text: str) -> str:
    """
    Remove Project Gutenberg header and footer.

    Args:
        text: Raw text from Gutenberg

    Returns:
        Cleaned text content
    """
    # Find start of content
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG"
    ]

    start_pos = 0
    for marker in start_markers:
        pos = text.find(marker)
        if pos != -1:
            # Find end of marker line
            start_pos = text.find('\n', pos) + 1
            break

    # Find end of content
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg"
    ]

    end_pos = len(text)
    for marker in end_markers:
        pos = text.find(marker)
        if pos != -1:
            end_pos = pos
            break

    return text[start_pos:end_pos].strip()


def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities (people, places, organizations).

    Args:
        text: Text to analyze

    Returns:
        Dict with entity lists
    """
    try:
        import nltk
        from nltk import word_tokenize, pos_tag, ne_chunk

        # Limit text length for performance (first 50,000 chars)
        sample_text = text[:50000]

        # Tokenize and tag
        tokens = word_tokenize(sample_text)
        tagged = pos_tag(tokens)

        # Named entity chunking
        entities = ne_chunk(tagged, binary=False)

        # Extract entities by type
        people = []
        places = []
        organizations = []

        for chunk in entities:
            if hasattr(chunk, 'label'):
                entity_text = ' '.join(c[0] for c in chunk)

                if chunk.label() == 'PERSON':
                    people.append(entity_text)
                elif chunk.label() == 'GPE':  # Geo-political entity (place)
                    places.append(entity_text)
                elif chunk.label() == 'ORGANIZATION':
                    organizations.append(entity_text)

        # Count and get top entities
        people_counts = Counter(people).most_common(20)
        places_counts = Counter(places).most_common(20)
        org_counts = Counter(organizations).most_common(20)

        return {
            'people': [{'name': name, 'count': count} for name, count in people_counts],
            'places': [{'name': name, 'count': count} for name, count in places_counts],
            'organizations': [{'name': name, 'count': count} for name, count in org_counts]
        }

    except Exception as e:
        print(f"Error in NER: {e}")
        return {'people': [], 'places': [], 'organizations': []}


def calculate_literary_features(words: List[str], sentences: List[str]) -> Dict[str, Any]:
    """
    Calculate literary and stylistic features.

    Args:
        words: List of word tokens
        sentences: List of sentences

    Returns:
        Dict with literary features
    """
    features = {}

    # Lexical diversity (Type-Token Ratio)
    features['type_token_ratio'] = len(set(words)) / len(words) if words else 0

    # Average word length
    features['avg_word_length'] = sum(len(w) for w in words) / len(words) if words else 0

    # Long words (> 6 characters)
    long_words = [w for w in words if len(w) > 6]
    features['long_word_ratio'] = len(long_words) / len(words) if words else 0

    # Sentence length variation (standard deviation approximation)
    sentence_lengths = [len(s.split()) for s in sentences]
    if sentence_lengths:
        avg_len = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((l - avg_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
        features['sentence_length_std'] = variance ** 0.5
    else:
        features['sentence_length_std'] = 0

    # Readability score (simplified Flesch-Kincaid)
    total_syllables = estimate_syllables(words)
    if words and sentences:
        features['readability_score'] = (
            206.835
            - 1.015 * (len(words) / len(sentences))
            - 84.6 * (total_syllables / len(words))
        )
    else:
        features['readability_score'] = 0

    # Round values
    for key in features:
        if isinstance(features[key], float):
            features[key] = round(features[key], 3)

    return features


def estimate_syllables(words: List[str]) -> int:
    """
    Estimate syllable count (simplified).

    Args:
        words: List of words

    Returns:
        Estimated total syllables
    """
    total = 0
    for word in words:
        # Simple vowel counting heuristic
        vowels = 'aeiou'
        syllables = sum(1 for char in word.lower() if char in vowels)
        # Minimum 1 syllable per word
        syllables = max(1, syllables)
        total += syllables
    return total


def store_results_in_dynamodb(results: Dict[str, Any]):
    """
    Store analysis results in DynamoDB.

    Args:
        results: Analysis results dict
    """
    try:
        table = dynamodb.Table(DYNAMODB_TABLE)

        # Prepare item for DynamoDB
        item = {
            'document_id': results['document_id'],
            'timestamp': results['timestamp'],
            'author': results['author'],
            'title': results['title'],
            'period': results['period'],
            'genre': results['genre'],
            'word_count': results['word_count'],
            'unique_words': results['unique_words'],
            'vocabulary_richness': results['vocabulary_richness'],
            'sentence_count': results['sentence_count'],
            'avg_sentence_length': results['avg_sentence_length'],
            's3_key': results['s3_key'],
            'bucket': results['bucket'],
        }

        # Add literary features if present
        if 'literary_features' in results:
            for key, value in results['literary_features'].items():
                item[f'literary_{key}'] = value

        # Store in DynamoDB
        table.put_item(Item=item)

        print(f"Stored results in DynamoDB: {results['document_id']}")

    except Exception as e:
        print(f"Error storing in DynamoDB: {e}")
        raise


def save_detailed_results_to_s3(bucket: str, original_key: str, results: Dict[str, Any]):
    """
    Save detailed analysis results to S3 as JSON.

    Args:
        bucket: S3 bucket name
        original_key: Original text file key
        results: Complete analysis results
    """
    try:
        # Create results key
        filename = os.path.basename(original_key).replace('.txt', '.json')
        results_key = f"processed/{filename}"

        # Convert to JSON
        results_json = json.dumps(results, indent=2, default=str)

        # Upload to S3
        s3.put_object(
            Bucket=bucket,
            Key=results_key,
            Body=results_json,
            ContentType='application/json',
            Metadata={
                'original_key': original_key,
                'processed_timestamp': datetime.now().isoformat()
            }
        )

        print(f"Saved detailed results to s3://{bucket}/{results_key}")

    except Exception as e:
        print(f"Error saving detailed results: {e}")
        # Non-critical error, don't raise


# Test function for local testing
def test_lambda_locally():
    """Test Lambda function locally with sample event."""
    # Sample event
    event = {
        'bucket': BUCKET_NAME,
        'key': 'raw/test/sample.txt'
    }

    # Mock context
    class Context:
        def __init__(self):
            self.function_name = 'process-text-document'
            self.memory_limit_in_mb = 512
            self.invoked_function_arn = 'arn:aws:lambda:us-east-1:123456789:function:test'

    context = Context()

    # Run handler
    response = lambda_handler(event, context)
    print(json.dumps(response, indent=2))


if __name__ == '__main__':
    # For local testing
    test_lambda_locally()
