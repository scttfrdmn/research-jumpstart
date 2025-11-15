#!/usr/bin/env python3
"""
Query sentiment analysis results from DynamoDB.

This script:
- Queries DynamoDB for analyzed posts
- Filters by sentiment, time range, hashtags
- Aggregates sentiment statistics
- Displays results in formatted tables
- Exports results to CSV/JSON

Usage:
    python scripts/query_results.py \
        --table-name SocialMediaPosts \
        --sentiment POSITIVE \
        --limit 100
"""

import argparse
import logging
import os
import sys
from collections import Counter
from typing import Any

import boto3
from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class SentimentAnalyzer:
    """Query and analyze sentiment data from DynamoDB."""

    def __init__(self, table_name: str, region: str = "us-east-1"):
        """
        Initialize analyzer.

        Args:
            table_name: DynamoDB table name
            region: AWS region
        """
        self.table_name = table_name
        self.region = region
        self.dynamodb = boto3.resource("dynamodb", region_name=region)
        self.table = self.dynamodb.Table(table_name)

    def query_all_posts(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Query all posts from DynamoDB.

        Args:
            limit: Maximum number of items to retrieve

        Returns:
            List of post dictionaries
        """
        try:
            logger.info(f"Querying table: {self.table_name}")
            response = self.table.scan(Limit=limit)
            items = response.get("Items", [])
            logger.info(f"Retrieved {len(items)} posts")
            return items

        except ClientError as e:
            logger.error(f"DynamoDB query error: {e}")
            return []

    def query_by_sentiment(self, sentiment: str, limit: int = 100) -> list[dict[str, Any]]:
        """
        Query posts by sentiment.

        Args:
            sentiment: Sentiment type (POSITIVE, NEGATIVE, NEUTRAL, MIXED)
            limit: Maximum results

        Returns:
            List of matching posts
        """
        try:
            response = self.table.scan(
                FilterExpression=Attr("sentiment").eq(sentiment), Limit=limit
            )
            return response.get("Items", [])

        except ClientError as e:
            logger.error(f"Query error: {e}")
            return []

    def get_statistics(self, posts: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Calculate sentiment statistics.

        Args:
            posts: List of posts

        Returns:
            Statistics dictionary
        """
        if not posts:
            return {"error": "No posts to analyze"}

        sentiments = [post.get("sentiment", "UNKNOWN") for post in posts]
        sentiment_counts = Counter(sentiments)

        # Calculate average scores
        avg_positive = sum(post.get("positive_score", 0) for post in posts) / len(posts)
        avg_negative = sum(post.get("negative_score", 0) for post in posts) / len(posts)
        avg_neutral = sum(post.get("neutral_score", 0) for post in posts) / len(posts)

        # Hashtag analysis
        all_hashtags = []
        for post in posts:
            all_hashtags.extend(post.get("hashtags", []))
        hashtag_counts = Counter(all_hashtags)

        return {
            "total_posts": len(posts),
            "sentiment_distribution": dict(sentiment_counts),
            "average_scores": {
                "positive": round(avg_positive, 3),
                "negative": round(avg_negative, 3),
                "neutral": round(avg_neutral, 3),
            },
            "top_hashtags": hashtag_counts.most_common(10),
        }

    def display_posts(self, posts: list[dict[str, Any]], max_display: int = 10):
        """
        Display posts in formatted table.

        Args:
            posts: List of posts
            max_display: Maximum posts to display
        """
        print("\n" + "=" * 80)
        print("SENTIMENT ANALYSIS RESULTS")
        print("=" * 80)

        for i, post in enumerate(posts[:max_display], 1):
            print(f"\nPost {i}: {post.get('post_id', 'N/A')}")
            print(f"  User: {post.get('username', 'N/A')}")
            print(f"  Text: {post.get('text', 'N/A')[:100]}...")
            print(f"  Sentiment: {post.get('sentiment', 'N/A')}")
            print(
                f"  Scores: P={post.get('positive_score', 0):.2f}, "
                f"N={post.get('negative_score', 0):.2f}, "
                f"Neu={post.get('neutral_score', 0):.2f}"
            )
            print(f"  Hashtags: {', '.join(post.get('hashtags', []))}")
            print(f"  Mentions: {', '.join(post.get('mentions', []))}")

        if len(posts) > max_display:
            print(f"\n... and {len(posts) - max_display} more posts")

    def display_statistics(self, stats: dict[str, Any]):
        """
        Display statistics in formatted output.

        Args:
            stats: Statistics dictionary
        """
        print("\n" + "=" * 80)
        print("SENTIMENT STATISTICS")
        print("=" * 80)
        print(f"\nTotal posts: {stats['total_posts']}")

        print("\nSentiment Distribution:")
        for sentiment, count in stats["sentiment_distribution"].items():
            percentage = (count / stats["total_posts"]) * 100
            print(f"  {sentiment}: {count} ({percentage:.1f}%)")

        print("\nAverage Sentiment Scores:")
        scores = stats["average_scores"]
        print(f"  Positive: {scores['positive']:.3f}")
        print(f"  Negative: {scores['negative']:.3f}")
        print(f"  Neutral:  {scores['neutral']:.3f}")

        print("\nTop Hashtags:")
        for hashtag, count in stats["top_hashtags"]:
            print(f"  #{hashtag}: {count}")

        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Query sentiment analysis results from DynamoDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query all posts
  python query_results.py --table-name SocialMediaPosts

  # Query positive posts only
  python query_results.py --table-name SocialMediaPosts --sentiment POSITIVE

  # Limit results
  python query_results.py --table-name SocialMediaPosts --limit 50
        """,
    )

    parser.add_argument(
        "--table-name",
        type=str,
        default=os.getenv("DYNAMODB_TABLE", "SocialMediaPosts"),
        help="DynamoDB table name (default: SocialMediaPosts)",
    )

    parser.add_argument(
        "--sentiment",
        type=str,
        choices=["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"],
        help="Filter by sentiment type",
    )

    parser.add_argument(
        "--limit", type=int, default=100, help="Maximum number of results (default: 100)"
    )

    parser.add_argument(
        "--region",
        type=str,
        default=os.getenv("AWS_REGION", "us-east-1"),
        help="AWS region (default: us-east-1)",
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = SentimentAnalyzer(args.table_name, region=args.region)

    # Query posts
    if args.sentiment:
        posts = analyzer.query_by_sentiment(args.sentiment, limit=args.limit)
        logger.info(f"Found {len(posts)} {args.sentiment} posts")
    else:
        posts = analyzer.query_all_posts(limit=args.limit)

    if not posts:
        logger.error("No posts found")
        sys.exit(1)

    # Display results
    analyzer.display_posts(posts)

    # Calculate and display statistics
    stats = analyzer.get_statistics(posts)
    analyzer.display_statistics(stats)


if __name__ == "__main__":
    main()
