"""
Query Linguistic Analysis Results from DynamoDB

This script queries and analyzes linguistic data stored in DynamoDB.
It supports:
- Querying by language, genre, register
- Searching for linguistic patterns
- Exporting results to CSV/JSON
- Displaying formatted tables
"""

import argparse
import json
import sys
from decimal import Decimal
from typing import Optional

import boto3
import pandas as pd
from boto3.dynamodb.conditions import Attr
from tabulate import tabulate

# Initialize DynamoDB resource
dynamodb = boto3.resource("dynamodb")


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder for DynamoDB Decimal types."""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Query linguistic analysis results from DynamoDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get all texts
  python query_results.py --table LinguisticAnalysis --all

  # Filter by language
  python query_results.py --table LinguisticAnalysis --language english

  # Filter by genre
  python query_results.py --table LinguisticAnalysis --genre academic

  # Get specific text by ID
  python query_results.py --table LinguisticAnalysis --text-id english_academic_article1

  # Export to CSV
  python query_results.py --table LinguisticAnalysis --all --export results.csv

  # Show detailed analysis
  python query_results.py --table LinguisticAnalysis --text-id english_academic_article1 --detailed
        """,
    )

    parser.add_argument(
        "--table",
        default="LinguisticAnalysis",
        help="DynamoDB table name (default: LinguisticAnalysis)",
    )

    parser.add_argument("--text-id", help="Get specific text by ID")

    parser.add_argument("--language", help="Filter by language")

    parser.add_argument("--genre", help="Filter by genre")

    parser.add_argument("--all", action="store_true", help="Get all texts")

    parser.add_argument("--min-words", type=int, help="Minimum word count")

    parser.add_argument("--min-ttr", type=float, help="Minimum type-token ratio")

    parser.add_argument("--export", help="Export results to file (CSV or JSON)")

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed analysis (collocations, POS distribution, etc.)",
    )

    parser.add_argument(
        "--limit", type=int, default=100, help="Maximum number of results to return (default: 100)"
    )

    return parser.parse_args()


def get_text_by_id(table_name: str, text_id: str) -> Optional[dict]:
    """
    Get single text by ID.

    Args:
        table_name: DynamoDB table name
        text_id: Text identifier

    Returns:
        dict: Text data or None if not found
    """
    table = dynamodb.Table(table_name)

    try:
        response = table.get_item(Key={"text_id": text_id})
        return response.get("Item")
    except Exception as e:
        print(f"Error querying DynamoDB: {e}")
        return None


def scan_all_texts(table_name: str, filters: Optional[dict] = None, limit: int = 100) -> list[dict]:
    """
    Scan table for all texts with optional filters.

    Args:
        table_name: DynamoDB table name
        filters: Optional filter conditions
        limit: Maximum number of items to return

    Returns:
        List of text data dictionaries
    """
    table = dynamodb.Table(table_name)
    items = []

    try:
        # Build scan parameters
        scan_kwargs = {"Limit": limit}

        # Add filter expressions
        if filters:
            filter_expressions = []

            if "language" in filters:
                filter_expressions.append(Attr("language").eq(filters["language"]))

            if "genre" in filters:
                filter_expressions.append(Attr("genre").eq(filters["genre"]))

            if "min_words" in filters:
                filter_expressions.append(Attr("word_count").gte(filters["min_words"]))

            if "min_ttr" in filters:
                filter_expressions.append(Attr("lexical_diversity.ttr").gte(filters["min_ttr"]))

            # Combine filter expressions
            if filter_expressions:
                combined_filter = filter_expressions[0]
                for expr in filter_expressions[1:]:
                    combined_filter = combined_filter & expr
                scan_kwargs["FilterExpression"] = combined_filter

        # Perform scan
        response = table.scan(**scan_kwargs)
        items.extend(response["Items"])

        # Handle pagination
        while "LastEvaluatedKey" in response and len(items) < limit:
            scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
            response = table.scan(**scan_kwargs)
            items.extend(response["Items"])

        return items[:limit]

    except Exception as e:
        print(f"Error scanning DynamoDB: {e}")
        return []


def format_basic_summary(items: list[dict]) -> str:
    """
    Format basic summary table.

    Args:
        items: List of text data

    Returns:
        str: Formatted table
    """
    if not items:
        return "No results found."

    # Extract key fields
    rows = []
    for item in items:
        lexical_div = item.get("lexical_diversity", {})
        rows.append(
            {
                "Text ID": item.get("text_id", "N/A"),
                "Language": item.get("language", "N/A"),
                "Genre": item.get("genre", "N/A"),
                "Words": item.get("word_count", 0),
                "Sentences": item.get("sentence_count", 0),
                "Unique Words": item.get("unique_words", 0),
                "TTR": f"{lexical_div.get('ttr', 0):.3f}"
                if isinstance(lexical_div, dict)
                else "N/A",
            }
        )

    return tabulate(rows, headers="keys", tablefmt="grid")


def format_detailed_analysis(item: dict) -> str:
    """
    Format detailed analysis for single text.

    Args:
        item: Text data

    Returns:
        str: Formatted detailed analysis
    """
    output = []
    output.append(f"\n{'=' * 80}")
    output.append(f"Detailed Analysis: {item.get('text_id', 'N/A')}")
    output.append(f"{'=' * 80}\n")

    # Basic Information
    output.append("BASIC INFORMATION")
    output.append("-" * 80)
    output.append(f"Language: {item.get('language', 'N/A')}")
    output.append(f"Genre: {item.get('genre', 'N/A')}")
    output.append(f"S3 Key: {item.get('s3_key', 'N/A')}")
    output.append(f"Word Count: {item.get('word_count', 0):,}")
    output.append(f"Sentence Count: {item.get('sentence_count', 0):,}")
    output.append(f"Unique Words: {item.get('unique_words', 0):,}")
    output.append(f"Unique Lemmas: {item.get('unique_lemmas', 0):,}")
    output.append(f"Average Word Length: {item.get('avg_word_length', 0):.2f} characters\n")

    # Lexical Diversity
    output.append("LEXICAL DIVERSITY")
    output.append("-" * 80)
    lexical_div = item.get("lexical_diversity", {})
    if isinstance(lexical_div, dict):
        output.append(f"Type-Token Ratio (TTR): {lexical_div.get('ttr', 0):.4f}")
        output.append(f"Moving Average TTR (MATTR): {lexical_div.get('mattr', 0):.4f}")
        output.append(f"Root TTR: {lexical_div.get('rttr', 0):.4f}")
        output.append(f"Types: {lexical_div.get('types', 0):,}")
        output.append(f"Tokens: {lexical_div.get('tokens', 0):,}\n")
    else:
        output.append("No lexical diversity data available\n")

    # Syntactic Complexity
    output.append("SYNTACTIC COMPLEXITY")
    output.append("-" * 80)
    syntax = item.get("syntactic_complexity", {})
    if isinstance(syntax, dict):
        output.append(f"Average Sentence Length: {syntax.get('avg_sentence_length', 0):.2f} words")
        output.append(f"Sentence Length Std Dev: {syntax.get('sentence_length_std', 0):.2f}")
        output.append(f"Min Sentence Length: {syntax.get('min_sentence_length', 0)} words")
        output.append(f"Max Sentence Length: {syntax.get('max_sentence_length', 0)} words\n")
    else:
        output.append("No syntactic complexity data available\n")

    # POS Distribution
    output.append("PART-OF-SPEECH DISTRIBUTION")
    output.append("-" * 80)
    pos_dist = item.get("pos_distribution", {})
    if isinstance(pos_dist, dict) and pos_dist:
        pos_rows = []
        total = sum(pos_dist.values())
        for pos, count in sorted(pos_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            pos_rows.append({"POS": pos, "Count": count, "Percentage": f"{percentage:.1f}%"})
        output.append(tabulate(pos_rows, headers="keys", tablefmt="simple"))
    else:
        output.append("No POS distribution data available")
    output.append("")

    # Top Words
    output.append("TOP 10 WORDS")
    output.append("-" * 80)
    top_words = item.get("top_words", [])
    if top_words:
        word_rows = []
        for i, word_data in enumerate(top_words[:10], 1):
            if isinstance(word_data, dict):
                word_rows.append(
                    {
                        "Rank": i,
                        "Word": word_data.get("word", "N/A"),
                        "Frequency": word_data.get("freq", 0),
                    }
                )
        output.append(tabulate(word_rows, headers="keys", tablefmt="simple"))
    else:
        output.append("No word frequency data available")
    output.append("")

    # Top Collocations
    output.append("TOP COLLOCATIONS (Bigrams)")
    output.append("-" * 80)
    collocations = item.get("collocations", {})
    if isinstance(collocations, dict):
        bigrams = collocations.get("bigrams", [])
        if bigrams:
            bigram_rows = []
            for i, bigram_data in enumerate(bigrams[:10], 1):
                bigram_rows.append(
                    {
                        "Rank": i,
                        "Bigram": bigram_data.get("bigram", "N/A"),
                        "PMI": f"{bigram_data.get('pmi', 0):.2f}",
                        "Frequency": bigram_data.get("freq", 0),
                    }
                )
            output.append(tabulate(bigram_rows, headers="keys", tablefmt="simple"))
        else:
            output.append("No bigram data available")
    else:
        output.append("No collocation data available")
    output.append("")

    output.append("=" * 80)

    return "\n".join(output)


def export_results(items: list[dict], filename: str):
    """
    Export results to CSV or JSON file.

    Args:
        items: List of text data
        filename: Output filename
    """
    # Convert Decimals to float
    items_json = json.loads(json.dumps(items, cls=DecimalEncoder))

    if filename.endswith(".csv"):
        # Flatten nested structures for CSV
        flattened = []
        for item in items_json:
            flat_item = {
                "text_id": item.get("text_id"),
                "language": item.get("language"),
                "genre": item.get("genre"),
                "word_count": item.get("word_count"),
                "sentence_count": item.get("sentence_count"),
                "unique_words": item.get("unique_words"),
                "avg_word_length": item.get("avg_word_length"),
            }

            # Add lexical diversity
            lexical_div = item.get("lexical_diversity", {})
            if isinstance(lexical_div, dict):
                flat_item["ttr"] = lexical_div.get("ttr")
                flat_item["mattr"] = lexical_div.get("mattr")
                flat_item["rttr"] = lexical_div.get("rttr")

            # Add syntactic complexity
            syntax = item.get("syntactic_complexity", {})
            if isinstance(syntax, dict):
                flat_item["avg_sentence_length"] = syntax.get("avg_sentence_length")
                flat_item["sentence_length_std"] = syntax.get("sentence_length_std")

            flattened.append(flat_item)

        df = pd.DataFrame(flattened)
        df.to_csv(filename, index=False)
        print(f"Exported {len(flattened)} results to {filename}")

    elif filename.endswith(".json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(items_json, f, indent=2, ensure_ascii=False)
        print(f"Exported {len(items_json)} results to {filename}")

    else:
        print(f"Unknown file format: {filename}")
        print("Supported formats: .csv, .json")


def main():
    """Main function."""
    args = parse_arguments()

    print(f"{'=' * 80}")
    print(f"Querying DynamoDB Table: {args.table}")
    print(f"{'=' * 80}\n")

    # Query based on arguments
    if args.text_id:
        # Get specific text
        item = get_text_by_id(args.table, args.text_id)
        if item:
            if args.detailed:
                print(format_detailed_analysis(item))
            else:
                print(format_basic_summary([item]))

            if args.export:
                export_results([item], args.export)
        else:
            print(f"Text not found: {args.text_id}")
            sys.exit(1)

    elif args.all or args.language or args.genre or args.min_words or args.min_ttr:
        # Scan with filters
        filters = {}
        if args.language:
            filters["language"] = args.language
        if args.genre:
            filters["genre"] = args.genre
        if args.min_words:
            filters["min_words"] = args.min_words
        if args.min_ttr:
            filters["min_ttr"] = args.min_ttr

        items = scan_all_texts(args.table, filters, args.limit)

        if items:
            print(f"Found {len(items)} results\n")
            print(format_basic_summary(items))

            if args.export:
                export_results(items, args.export)
        else:
            print("No results found matching the criteria")

    else:
        print("No query parameters specified. Use --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
