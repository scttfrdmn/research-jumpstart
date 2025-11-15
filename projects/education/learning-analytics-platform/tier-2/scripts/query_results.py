#!/usr/bin/env python3
"""
Query and analyze student analytics results from DynamoDB.

This script provides:
- Query student metrics by risk level, course, grade range
- Aggregate class-level statistics
- Generate formatted reports
- Export results to CSV

Usage:
    # Query high-risk students
    python scripts/query_results.py --risk-level high

    # Query by course
    python scripts/query_results.py --course-id COURSE_001

    # Get class statistics
    python scripts/query_results.py --class-stats

    # Export to CSV
    python scripts/query_results.py --export results.csv
"""

import argparse
import logging
import os
import sys
from typing import Optional

import boto3
import pandas as pd
from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tabulate import tabulate

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# AWS clients
dynamodb = boto3.resource("dynamodb", region_name=os.getenv("AWS_REGION", "us-east-1"))


class StudentAnalyticsQuery:
    """Query and analyze student analytics from DynamoDB."""

    def __init__(self, table_name: str):
        """
        Initialize query interface.

        Args:
            table_name: DynamoDB table name
        """
        self.table_name = table_name
        self.table = dynamodb.Table(table_name)
        logger.info(f"Connected to DynamoDB table: {table_name}")

    def query_by_risk_level(self, risk_level: str, limit: int = 100) -> list[dict]:
        """
        Query students by risk level.

        Args:
            risk_level: Risk level (high, medium, low, none)
            limit: Maximum number of results

        Returns:
            List of student records
        """
        try:
            response = self.table.scan(
                FilterExpression=Attr("risk_level").eq(risk_level), Limit=limit
            )

            items = response.get("Items", [])
            logger.info(f"Found {len(items)} students with risk level: {risk_level}")
            return items

        except ClientError as e:
            logger.error(f"Error querying DynamoDB: {e}")
            return []

    def query_by_course(self, course_id: str) -> list[dict]:
        """
        Query all students in a course.

        Args:
            course_id: Course identifier

        Returns:
            List of student records
        """
        try:
            response = self.table.scan(FilterExpression=Attr("course_id").eq(course_id))

            items = response.get("Items", [])
            logger.info(f"Found {len(items)} students in course: {course_id}")
            return items

        except ClientError as e:
            logger.error(f"Error querying DynamoDB: {e}")
            return []

    def query_by_grade_range(self, min_grade: float, max_grade: float) -> list[dict]:
        """
        Query students by grade range.

        Args:
            min_grade: Minimum average grade
            max_grade: Maximum average grade

        Returns:
            List of student records
        """
        try:
            response = self.table.scan(
                FilterExpression=Attr("avg_grade").between(min_grade, max_grade)
            )

            items = response.get("Items", [])
            logger.info(f"Found {len(items)} students with grades {min_grade}-{max_grade}")
            return items

        except ClientError as e:
            logger.error(f"Error querying DynamoDB: {e}")
            return []

    def get_all_students(self, limit: Optional[int] = None) -> list[dict]:
        """
        Get all student records.

        Args:
            limit: Maximum number of results (None for all)

        Returns:
            List of student records
        """
        try:
            response = self.table.scan(Limit=limit) if limit else self.table.scan()

            items = response.get("Items", [])

            # Handle pagination if no limit
            while "LastEvaluatedKey" in response and not limit:
                response = self.table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
                items.extend(response.get("Items", []))

            logger.info(f"Retrieved {len(items)} student records")
            return items

        except ClientError as e:
            logger.error(f"Error scanning DynamoDB: {e}")
            return []

    def calculate_class_statistics(self, students: list[dict]) -> dict:
        """
        Calculate aggregate class-level statistics.

        Args:
            students: List of student records

        Returns:
            Dictionary with class statistics
        """
        if not students:
            return {}

        df = pd.DataFrame(students)

        stats = {
            "total_students": len(df),
            "avg_grade_mean": float(df["avg_grade"].mean()),
            "avg_grade_median": float(df["avg_grade"].median()),
            "avg_grade_std": float(df["avg_grade"].std()),
            "completion_rate_mean": float(df["completion_rate"].mean()),
            "engagement_score_mean": float(df["engagement_score"].mean()),
            "risk_distribution": df["risk_level"].value_counts().to_dict(),
            "courses": df["course_id"].nunique(),
            "at_risk_count": len(df[df["risk_level"].isin(["high", "medium"])]),
            "mastery_level_mean": float(df["mastery_level"].mean())
            if "mastery_level" in df
            else 0.0,
        }

        return stats

    def format_results_table(
        self, students: list[dict], columns: Optional[list[str]] = None
    ) -> str:
        """
        Format results as a table.

        Args:
            students: List of student records
            columns: Columns to display (None for default)

        Returns:
            Formatted table string
        """
        if not students:
            return "No results found."

        df = pd.DataFrame(students)

        # Default columns
        if columns is None:
            columns = [
                "student_id",
                "course_id",
                "avg_grade",
                "completion_rate",
                "engagement_score",
                "risk_level",
            ]

        # Filter to available columns
        columns = [col for col in columns if col in df.columns]

        # Truncate student_id for display
        if "student_id" in df.columns:
            df["student_id"] = df["student_id"].str[:8] + "..."

        return tabulate(df[columns].head(20), headers="keys", tablefmt="grid", showindex=False)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Query student analytics from DynamoDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--table-name",
        type=str,
        default=os.getenv("DYNAMODB_TABLE", "StudentAnalytics"),
        help="DynamoDB table name (default: StudentAnalytics)",
    )

    parser.add_argument(
        "--risk-level",
        type=str,
        choices=["high", "medium", "low", "none"],
        help="Filter by risk level",
    )

    parser.add_argument("--course-id", type=str, help="Filter by course ID")

    parser.add_argument("--min-grade", type=float, help="Minimum average grade")

    parser.add_argument("--max-grade", type=float, help="Maximum average grade")

    parser.add_argument(
        "--limit", type=int, default=100, help="Maximum number of results (default: 100)"
    )

    parser.add_argument(
        "--class-stats", action="store_true", help="Calculate and display class statistics"
    )

    parser.add_argument("--export", type=str, help="Export results to CSV file")

    args = parser.parse_args()

    # Initialize query interface
    query = StudentAnalyticsQuery(args.table_name)

    # Execute query based on arguments
    if args.risk_level:
        students = query.query_by_risk_level(args.risk_level, limit=args.limit)
    elif args.course_id:
        students = query.query_by_course(args.course_id)
    elif args.min_grade is not None and args.max_grade is not None:
        students = query.query_by_grade_range(args.min_grade, args.max_grade)
    else:
        students = query.get_all_students(limit=args.limit)

    # Display results
    if students:
        print("\n" + "=" * 80)
        print("STUDENT ANALYTICS RESULTS")
        print("=" * 80)
        print(query.format_results_table(students))
        print(f"\nTotal records: {len(students)}")

        # Calculate and display class statistics
        if args.class_stats or len(students) > 0:
            print("\n" + "=" * 80)
            print("CLASS STATISTICS")
            print("=" * 80)
            stats = query.calculate_class_statistics(students)

            print(f"Total Students:      {stats['total_students']}")
            print(
                f"Average Grade:       {stats['avg_grade_mean']:.2f} ± {stats['avg_grade_std']:.2f}"
            )
            print(f"Median Grade:        {stats['avg_grade_median']:.2f}")
            print(f"Completion Rate:     {stats['completion_rate_mean']:.2f}%")
            print(f"Engagement Score:    {stats['engagement_score_mean']:.2f}")
            print(f"Mastery Level:       {stats['mastery_level_mean']:.2f}%")
            print(
                f"At-Risk Students:    {stats['at_risk_count']} ({stats['at_risk_count'] / stats['total_students'] * 100:.1f}%)"
            )
            print("\nRisk Distribution:")
            for level, count in sorted(stats["risk_distribution"].items()):
                print(f"  {level:10s}: {count:3d} ({count / stats['total_students'] * 100:.1f}%)")

        # Export to CSV if requested
        if args.export:
            df = pd.DataFrame(students)
            df.to_csv(args.export, index=False)
            print(f"\nResults exported to: {args.export}")

    else:
        print("No results found.")

    sys.exit(0)


if __name__ == "__main__":
    main()
