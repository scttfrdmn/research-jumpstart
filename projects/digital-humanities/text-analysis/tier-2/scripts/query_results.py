#!/usr/bin/env python3
"""
Query and analyze text analysis results from DynamoDB.

This script provides convenient methods to query DynamoDB for analyzed texts
and export results for visualization and further analysis.

Usage:
    # Query by author
    python query_results.py --author "Jane Austen"

    # Query by period
    python query_results.py --period "Victorian"

    # Query by vocabulary richness threshold
    python query_results.py --min-richness 0.7

    # Export all results to CSV
    python query_results.py --export results.csv

    # Get summary statistics
    python query_results.py --summary
"""

import argparse
import json

import boto3
import pandas as pd
from boto3.dynamodb.conditions import Attr


class TextAnalysisQuerier:
    """Query text analysis results from DynamoDB."""

    def __init__(self, table_name: str = "TextAnalysis"):
        """
        Initialize querier.

        Args:
            table_name: DynamoDB table name
        """
        self.dynamodb = boto3.resource("dynamodb")
        self.table = self.dynamodb.Table(table_name)
        self.table_name = table_name

    def query_by_author(self, author: str) -> list[dict]:
        """
        Query all documents by author.

        Args:
            author: Author name

        Returns:
            List of document analysis results
        """
        try:
            response = self.table.scan(FilterExpression=Attr("author").eq(author))

            items = response.get("Items", [])

            # Handle pagination
            while "LastEvaluatedKey" in response:
                response = self.table.scan(
                    FilterExpression=Attr("author").eq(author),
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                )
                items.extend(response.get("Items", []))

            return items

        except Exception as e:
            print(f"Error querying by author: {e}")
            return []

    def query_by_period(self, period: str) -> list[dict]:
        """
        Query all documents by literary period.

        Args:
            period: Literary period (e.g., "Victorian", "Romantic")

        Returns:
            List of document analysis results
        """
        try:
            response = self.table.scan(FilterExpression=Attr("period").eq(period))

            items = response.get("Items", [])

            # Handle pagination
            while "LastEvaluatedKey" in response:
                response = self.table.scan(
                    FilterExpression=Attr("period").eq(period),
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                )
                items.extend(response.get("Items", []))

            return items

        except Exception as e:
            print(f"Error querying by period: {e}")
            return []

    def query_by_vocabulary_richness(
        self, min_richness: float = 0.0, max_richness: float = 1.0
    ) -> list[dict]:
        """
        Query documents by vocabulary richness range.

        Args:
            min_richness: Minimum type-token ratio
            max_richness: Maximum type-token ratio

        Returns:
            List of document analysis results
        """
        try:
            response = self.table.scan(
                FilterExpression=Attr("vocabulary_richness").between(min_richness, max_richness)
            )

            items = response.get("Items", [])

            # Handle pagination
            while "LastEvaluatedKey" in response:
                response = self.table.scan(
                    FilterExpression=Attr("vocabulary_richness").between(
                        min_richness, max_richness
                    ),
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                )
                items.extend(response.get("Items", []))

            return items

        except Exception as e:
            print(f"Error querying by vocabulary richness: {e}")
            return []

    def get_all_documents(self) -> list[dict]:
        """
        Get all documents from table.

        Returns:
            List of all document analysis results
        """
        try:
            response = self.table.scan()
            items = response.get("Items", [])

            # Handle pagination
            while "LastEvaluatedKey" in response:
                response = self.table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
                items.extend(response.get("Items", []))

            return items

        except Exception as e:
            print(f"Error getting all documents: {e}")
            return []

    def search_by_title(self, title_substring: str) -> list[dict]:
        """
        Search documents by title substring.

        Args:
            title_substring: Substring to search for in titles

        Returns:
            List of matching documents
        """
        try:
            response = self.table.scan(FilterExpression=Attr("title").contains(title_substring))

            items = response.get("Items", [])

            # Handle pagination
            while "LastEvaluatedKey" in response:
                response = self.table.scan(
                    FilterExpression=Attr("title").contains(title_substring),
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                )
                items.extend(response.get("Items", []))

            return items

        except Exception as e:
            print(f"Error searching by title: {e}")
            return []

    def get_corpus_statistics(self) -> dict:
        """
        Calculate corpus-wide statistics.

        Returns:
            Dict with corpus statistics
        """
        all_docs = self.get_all_documents()

        if not all_docs:
            return {}

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(all_docs)

        stats = {
            "total_documents": len(all_docs),
            "total_words": int(df["word_count"].sum()) if "word_count" in df else 0,
            "avg_vocabulary_richness": float(df["vocabulary_richness"].mean())
            if "vocabulary_richness" in df
            else 0,
            "avg_sentence_length": float(df["avg_sentence_length"].mean())
            if "avg_sentence_length" in df
            else 0,
            "authors": df["author"].nunique() if "author" in df else 0,
            "periods": df["period"].nunique() if "period" in df else 0,
            "genres": df["genre"].nunique() if "genre" in df else 0,
        }

        # Author statistics
        if "author" in df:
            author_stats = (
                df.groupby("author")
                .agg({"word_count": "sum", "vocabulary_richness": "mean", "document_id": "count"})
                .to_dict("index")
            )
            stats["by_author"] = author_stats

        # Period statistics
        if "period" in df:
            period_stats = (
                df.groupby("period")
                .agg({"word_count": "sum", "vocabulary_richness": "mean", "document_id": "count"})
                .to_dict("index")
            )
            stats["by_period"] = period_stats

        return stats


def display_documents(documents: list[dict], sort_by: str = "author"):
    """
    Display documents in formatted table.

    Args:
        documents: List of document dicts
        sort_by: Field to sort by
    """
    if not documents:
        print("No documents found.")
        return

    # Convert to DataFrame for display
    df = pd.DataFrame(documents)

    # Select display columns
    display_cols = [
        "author",
        "title",
        "period",
        "genre",
        "word_count",
        "unique_words",
        "vocabulary_richness",
        "avg_sentence_length",
    ]

    # Filter to existing columns
    display_cols = [col for col in display_cols if col in df.columns]

    # Sort
    if sort_by in df.columns:
        df = df.sort_values(sort_by)

    print(f"\n{'=' * 100}")
    print(f"Found {len(documents)} documents")
    print(f"{'=' * 100}\n")

    # Display table
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 40)

    print(df[display_cols].to_string(index=False))
    print(f"\n{'=' * 100}\n")


def display_statistics(stats: dict):
    """
    Display corpus statistics.

    Args:
        stats: Statistics dict
    """
    print(f"\n{'=' * 80}")
    print("CORPUS STATISTICS")
    print(f"{'=' * 80}\n")

    print(f"Total Documents:              {stats.get('total_documents', 0):,}")
    print(f"Total Words:                  {stats.get('total_words', 0):,}")
    print(f"Average Vocabulary Richness:  {stats.get('avg_vocabulary_richness', 0):.4f}")
    print(f"Average Sentence Length:      {stats.get('avg_sentence_length', 0):.2f} words")
    print(f"Unique Authors:               {stats.get('authors', 0)}")
    print(f"Literary Periods:             {stats.get('periods', 0)}")
    print(f"Genres:                       {stats.get('genres', 0)}")

    # By author
    if "by_author" in stats:
        print(f"\n{'-' * 80}")
        print("BY AUTHOR")
        print(f"{'-' * 80}")

        author_df = pd.DataFrame.from_dict(stats["by_author"], orient="index")
        author_df = author_df.sort_values("vocabulary_richness", ascending=False)

        print(f"\n{'Author':<30} {'Documents':>10} {'Total Words':>15} {'Avg Richness':>15}")
        print("-" * 80)

        for author, row in author_df.iterrows():
            doc_count = int(row["document_id"])
            total_words = int(row["word_count"])
            avg_richness = float(row["vocabulary_richness"])
            print(f"{author:<30} {doc_count:>10} {total_words:>15,} {avg_richness:>15.4f}")

    # By period
    if "by_period" in stats:
        print(f"\n{'-' * 80}")
        print("BY PERIOD")
        print(f"{'-' * 80}")

        period_df = pd.DataFrame.from_dict(stats["by_period"], orient="index")
        period_df = period_df.sort_values("vocabulary_richness", ascending=False)

        print(f"\n{'Period':<30} {'Documents':>10} {'Total Words':>15} {'Avg Richness':>15}")
        print("-" * 80)

        for period, row in period_df.iterrows():
            doc_count = int(row["document_id"])
            total_words = int(row["word_count"])
            avg_richness = float(row["vocabulary_richness"])
            print(f"{period:<30} {doc_count:>10} {total_words:>15,} {avg_richness:>15.4f}")

    print(f"\n{'=' * 80}\n")


def export_to_csv(documents: list[dict], output_file: str):
    """
    Export documents to CSV file.

    Args:
        documents: List of document dicts
        output_file: Output CSV file path
    """
    if not documents:
        print("No documents to export.")
        return

    df = pd.DataFrame(documents)

    # Export
    df.to_csv(output_file, index=False)
    print(f"✓ Exported {len(documents)} documents to {output_file}")


def export_to_json(documents: list[dict], output_file: str):
    """
    Export documents to JSON file.

    Args:
        documents: List of document dicts
        output_file: Output JSON file path
    """
    if not documents:
        print("No documents to export.")
        return

    # Convert Decimal types to float for JSON serialization
    def decimal_to_float(obj):
        if isinstance(obj, dict):
            return {k: decimal_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [decimal_to_float(item) for item in obj]
        elif hasattr(obj, "__float__"):
            return float(obj)
        return obj

    documents = decimal_to_float(documents)

    with open(output_file, "w") as f:
        json.dump(documents, f, indent=2, default=str)

    print(f"✓ Exported {len(documents)} documents to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Query text analysis results from DynamoDB")

    # Query options
    parser.add_argument("--author", help="Query by author name")
    parser.add_argument("--period", help="Query by literary period")
    parser.add_argument("--title", help="Search by title substring")
    parser.add_argument("--min-richness", type=float, help="Minimum vocabulary richness")
    parser.add_argument("--max-richness", type=float, help="Maximum vocabulary richness")

    # Output options
    parser.add_argument("--export", help="Export to CSV file")
    parser.add_argument("--export-json", help="Export to JSON file")
    parser.add_argument("--summary", action="store_true", help="Show corpus statistics")
    parser.add_argument("--sort", default="author", help="Sort by field (default: author)")

    # Configuration
    parser.add_argument("--table", default="TextAnalysis", help="DynamoDB table name")

    args = parser.parse_args()

    # Initialize querier
    querier = TextAnalysisQuerier(table_name=args.table)

    # Execute query
    documents = []

    if args.author:
        print(f"Querying documents by author: {args.author}")
        documents = querier.query_by_author(args.author)

    elif args.period:
        print(f"Querying documents by period: {args.period}")
        documents = querier.query_by_period(args.period)

    elif args.title:
        print(f"Searching documents by title: {args.title}")
        documents = querier.search_by_title(args.title)

    elif args.min_richness is not None or args.max_richness is not None:
        min_r = args.min_richness if args.min_richness is not None else 0.0
        max_r = args.max_richness if args.max_richness is not None else 1.0
        print(f"Querying documents by vocabulary richness: {min_r:.2f} - {max_r:.2f}")
        documents = querier.query_by_vocabulary_richness(min_r, max_r)

    elif args.summary:
        print("Calculating corpus statistics...")
        stats = querier.get_corpus_statistics()
        display_statistics(stats)
        return

    else:
        print("Getting all documents...")
        documents = querier.get_all_documents()

    # Display results
    if documents:
        display_documents(documents, sort_by=args.sort)

        # Export if requested
        if args.export:
            export_to_csv(documents, args.export)

        if args.export_json:
            export_to_json(documents, args.export_json)
    else:
        print("No documents found matching query.")

    # Show summary statistics
    if documents and not args.summary:
        print("\nTo see detailed corpus statistics, run with --summary flag")


if __name__ == "__main__":
    main()
