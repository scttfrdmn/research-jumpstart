#!/usr/bin/env python3
"""
Query Behavioral Analysis Results from DynamoDB

Retrieves analysis results from DynamoDB and displays or exports them.

Usage:
    python query_results.py --table BehavioralAnalysis
    python query_results.py --table BehavioralAnalysis --participant sub001
    python query_results.py --table BehavioralAnalysis --export results.csv
"""

import boto3
import pandas as pd
import argparse
import json
from decimal import Decimal
from boto3.dynamodb.conditions import Key, Attr


def decimal_to_float(obj):
    """
    Convert DynamoDB Decimal types to float for pandas compatibility.
    """
    if isinstance(obj, list):
        return [decimal_to_float(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, Decimal):
        return float(obj)
    else:
        return obj


def query_all_results(table_name):
    """
    Scan entire DynamoDB table and return all results.
    """
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)

    print(f"Querying all results from {table_name}...")

    # Scan table (use carefully - not efficient for large tables)
    response = table.scan()
    items = response['Items']

    # Handle pagination
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response['Items'])

    print(f"Found {len(items)} participants")

    # Convert to DataFrame
    items = [decimal_to_float(item) for item in items]
    df = pd.DataFrame(items)

    return df


def query_participant(table_name, participant_id):
    """
    Query results for a specific participant.
    """
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)

    print(f"Querying results for {participant_id}...")

    response = table.query(
        KeyConditionExpression=Key('participant_id').eq(participant_id)
    )

    items = response['Items']
    print(f"Found {len(items)} tasks for {participant_id}")

    items = [decimal_to_float(item) for item in items]
    return items


def query_by_task(table_name, task_type):
    """
    Query all participants who completed a specific task.
    """
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)

    print(f"Querying all participants for {task_type} task...")

    response = table.scan(
        FilterExpression=Attr('task_type').eq(task_type)
    )

    items = response['Items']
    print(f"Found {len(items)} participants")

    items = [decimal_to_float(item) for item in items]
    df = pd.DataFrame(items)

    return df


def query_by_performance(table_name, min_accuracy=None, max_rt=None):
    """
    Query participants by performance criteria.
    """
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)

    print(f"Querying participants by performance...")

    # Build filter expression
    filter_expressions = []
    if min_accuracy is not None:
        filter_expressions.append(Attr('accuracy').gte(min_accuracy))
    if max_rt is not None:
        filter_expressions.append(Attr('mean_rt').lte(max_rt))

    if filter_expressions:
        filter_expr = filter_expressions[0]
        for expr in filter_expressions[1:]:
            filter_expr = filter_expr & expr

        response = table.scan(FilterExpression=filter_expr)
    else:
        response = table.scan()

    items = response['Items']
    print(f"Found {len(items)} participants matching criteria")

    items = [decimal_to_float(item) for item in items]
    df = pd.DataFrame(items)

    return df


def display_summary_statistics(df):
    """
    Display summary statistics for the dataset.
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    print(f"\nTotal participants: {len(df)}")

    if 'task_type' in df.columns:
        print(f"\nTasks:")
        print(df['task_type'].value_counts().to_string())

    print(f"\n{'-'*60}")
    print("REACTION TIME")
    print(f"{'-'*60}")

    if 'mean_rt' in df.columns:
        print(f"Mean RT (ms):")
        print(f"  Mean:   {df['mean_rt'].mean():.2f}")
        print(f"  Median: {df['mean_rt'].median():.2f}")
        print(f"  SD:     {df['mean_rt'].std():.2f}")
        print(f"  Range:  [{df['mean_rt'].min():.2f}, {df['mean_rt'].max():.2f}]")

    print(f"\n{'-'*60}")
    print("ACCURACY")
    print(f"{'-'*60}")

    if 'accuracy' in df.columns:
        print(f"Accuracy:")
        print(f"  Mean:   {df['accuracy'].mean():.3f}")
        print(f"  Median: {df['accuracy'].median():.3f}")
        print(f"  SD:     {df['accuracy'].std():.3f}")
        print(f"  Range:  [{df['accuracy'].min():.3f}, {df['accuracy'].max():.3f}]")

    print(f"\n{'-'*60}")
    print("SIGNAL DETECTION")
    print(f"{'-'*60}")

    if 'dprime' in df.columns:
        dprime_data = df['dprime'].dropna()
        if len(dprime_data) > 0:
            print(f"d-prime:")
            print(f"  Mean:   {dprime_data.mean():.3f}")
            print(f"  Median: {dprime_data.median():.3f}")
            print(f"  SD:     {dprime_data.std():.3f}")

    if 'criterion' in df.columns:
        criterion_data = df['criterion'].dropna()
        if len(criterion_data) > 0:
            print(f"\nResponse Criterion:")
            print(f"  Mean:   {criterion_data.mean():.3f}")
            print(f"  Median: {criterion_data.median():.3f}")
            print(f"  SD:     {criterion_data.std():.3f}")

    # Task-specific statistics
    if 'task_type' in df.columns:
        print(f"\n{'-'*60}")
        print("BY TASK TYPE")
        print(f"{'-'*60}")

        for task in df['task_type'].unique():
            task_df = df[df['task_type'] == task]
            print(f"\n{task.upper()}:")
            print(f"  N = {len(task_df)}")
            if 'mean_rt' in task_df.columns:
                print(f"  Mean RT: {task_df['mean_rt'].mean():.2f} ms")
            if 'accuracy' in task_df.columns:
                print(f"  Accuracy: {task_df['accuracy'].mean():.3f}")


def display_participant_details(items):
    """
    Display detailed results for a specific participant.
    """
    if not items:
        print("No results found")
        return

    print("\n" + "="*60)
    print(f"PARTICIPANT: {items[0]['participant_id']}")
    print("="*60)

    for item in items:
        print(f"\n{'-'*60}")
        print(f"Task: {item['task_type']}")
        print(f"{'-'*60}")

        print(f"\nBasic Statistics:")
        print(f"  Trials:      {item.get('n_trials', 'N/A')}")
        print(f"  Accuracy:    {item.get('accuracy', 'N/A'):.3f}")
        print(f"  Mean RT:     {item.get('mean_rt', 'N/A'):.2f} ms")
        print(f"  Median RT:   {item.get('median_rt', 'N/A'):.2f} ms")

        if 'mean_rt_correct' in item:
            print(f"  Mean RT (correct): {item['mean_rt_correct']:.2f} ms")

        # Signal detection
        if 'dprime' in item and item['dprime'] is not None:
            print(f"\nSignal Detection:")
            print(f"  d-prime:      {item['dprime']:.3f}")
            print(f"  Criterion:    {item.get('criterion', 'N/A'):.3f}")
            print(f"  Hit rate:     {item.get('hit_rate', 'N/A'):.3f}")
            print(f"  FA rate:      {item.get('false_alarm_rate', 'N/A'):.3f}")

        # Task-specific results
        if item['task_type'] == 'stroop':
            if 'stroop_effect_rt' in item:
                print(f"\nStroop Effects:")
                print(f"  RT effect:   {item['stroop_effect_rt']:.2f} ms")
                print(f"  Acc effect:  {item.get('stroop_effect_accuracy', 'N/A'):.3f}")

        elif item['task_type'] == 'decision':
            if 'difficulty_effect_rt' in item:
                print(f"\nDifficulty Effects:")
                print(f"  RT effect:   {item['difficulty_effect_rt']:.2f} ms")
                print(f"  Acc effect:  {item.get('difficulty_effect_accuracy', 'N/A'):.3f}")

        elif item['task_type'] == 'learning':
            if 'learning_rate' in item:
                print(f"\nLearning:")
                print(f"  Learning rate: {item['learning_rate']:.4f}")
                print(f"  Initial acc:   {item.get('initial_accuracy', 'N/A'):.3f}")
                print(f"  Final acc:     {item.get('final_accuracy', 'N/A'):.3f}")

        # Computational models
        if 'model_ddm' in item and item['model_ddm']:
            print(f"\nDrift Diffusion Model:")
            ddm = item['model_ddm']
            print(f"  Drift rate:    {ddm.get('drift_rate', 'N/A'):.3f}")
            print(f"  Boundary:      {ddm.get('boundary_separation', 'N/A'):.3f}")
            print(f"  Non-decision:  {ddm.get('non_decision_time', 'N/A'):.3f} s")

        if 'model_q_learning' in item and item['model_q_learning']:
            print(f"\nQ-Learning Model:")
            q = item['model_q_learning']
            print(f"  Learning rate: {q.get('learning_rate', 'N/A'):.3f}")
            print(f"  Inv. temp:     {q.get('inverse_temperature', 'N/A'):.3f}")

        print(f"\nMetadata:")
        print(f"  Timestamp:   {item.get('timestamp', 'N/A')}")
        print(f"  S3 location: s3://{item.get('s3_bucket', 'N/A')}/{item.get('s3_key', 'N/A')}")


def export_to_csv(df, filename):
    """
    Export results to CSV file.
    """
    # Flatten nested structures for CSV export
    flattened = []
    for _, row in df.iterrows():
        flat_row = {}
        for key, value in row.items():
            if isinstance(value, dict):
                # Flatten nested dict
                for nested_key, nested_value in value.items():
                    flat_row[f"{key}_{nested_key}"] = nested_value
            elif isinstance(value, list):
                # Convert list to JSON string
                flat_row[key] = json.dumps(value)
            else:
                flat_row[key] = value
        flattened.append(flat_row)

    flat_df = pd.DataFrame(flattened)
    flat_df.to_csv(filename, index=False)
    print(f"\nExported {len(flat_df)} rows to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Query behavioral analysis results from DynamoDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query all results
  python query_results.py --table BehavioralAnalysis

  # Query specific participant
  python query_results.py --table BehavioralAnalysis --participant sub001

  # Query by task type
  python query_results.py --table BehavioralAnalysis --task stroop

  # Query by performance
  python query_results.py --table BehavioralAnalysis --min-accuracy 0.8 --max-rt 600

  # Export to CSV
  python query_results.py --table BehavioralAnalysis --export results.csv
        """
    )

    parser.add_argument('--table', default='BehavioralAnalysis',
                        help='DynamoDB table name (default: BehavioralAnalysis)')
    parser.add_argument('--participant',
                        help='Query specific participant ID')
    parser.add_argument('--task',
                        help='Query specific task type (stroop, decision, learning)')
    parser.add_argument('--min-accuracy', type=float,
                        help='Minimum accuracy filter')
    parser.add_argument('--max-rt', type=float,
                        help='Maximum mean RT filter (ms)')
    parser.add_argument('--export',
                        help='Export results to CSV file')
    parser.add_argument('--json', action='store_true',
                        help='Display results as JSON')

    args = parser.parse_args()

    # Query data based on arguments
    if args.participant:
        items = query_participant(args.table, args.participant)
        if args.json:
            print(json.dumps(items, indent=2))
        else:
            display_participant_details(items)

    elif args.task:
        df = query_by_task(args.table, args.task)
        if args.json:
            print(df.to_json(orient='records', indent=2))
        else:
            display_summary_statistics(df)

        if args.export:
            export_to_csv(df, args.export)

    elif args.min_accuracy or args.max_rt:
        df = query_by_performance(args.table, args.min_accuracy, args.max_rt)
        if args.json:
            print(df.to_json(orient='records', indent=2))
        else:
            display_summary_statistics(df)

        if args.export:
            export_to_csv(df, args.export)

    else:
        # Query all
        df = query_all_results(args.table)
        if args.json:
            print(df.to_json(orient='records', indent=2))
        else:
            display_summary_statistics(df)

        if args.export:
            export_to_csv(df, args.export)

    print("\n" + "="*60)
    print("Query complete!")
    print("="*60)


if __name__ == '__main__':
    main()
