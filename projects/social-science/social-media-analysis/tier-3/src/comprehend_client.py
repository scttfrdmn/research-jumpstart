"""AWS Comprehend integration for NLP analysis."""

import boto3
import logging

logger = logging.getLogger(__name__)

class ComprehendAnalyzer:
    """Client for AWS Comprehend NLP services."""
    
    def __init__(self, region: str = 'us-east-1'):
        self.client = boto3.client('comprehend', region_name=region)
        logger.info("Initialized Comprehend client")
    
    def analyze_sentiment(self, text: str, language: str = 'en') -> dict:
        """Analyze sentiment using Comprehend."""
        response = self.client.detect_sentiment(
            Text=text[:5000],  # Comprehend limit
            LanguageCode=language
        )
        return response
    
    def detect_entities(self, text: str, language: str = 'en') -> dict:
        """Detect named entities."""
        response = self.client.detect_entities(
            Text=text[:5000],
            LanguageCode=language
        )
        return response
    
    def detect_key_phrases(self, text: str, language: str = 'en') -> dict:
        """Extract key phrases."""
        response = self.client.detect_key_phrases(
            Text=text[:5000],
            LanguageCode=language
        )
        return response
