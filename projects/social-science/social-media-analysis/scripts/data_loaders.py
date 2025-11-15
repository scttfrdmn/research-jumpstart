#!/usr/bin/env python3
"""
Social Media Data Loaders

Utilities for loading and parsing social media data from various sources
and formats (JSON, CSV, Twitter API, synthetic data generation).

Usage:
    from data_loaders import load_twitter_json, generate_synthetic_posts

    tweets = load_twitter_json('tweets.json')
    synthetic_data = generate_synthetic_posts(n_posts=1000)
"""

import json
from datetime import timedelta

import networkx as nx
import numpy as np
import pandas as pd


def load_twitter_json(filepath, extract_fields=None):
    """
    Load Twitter data from JSON or JSONL format.

    Parameters:
    -----------
    filepath : str
        Path to JSON/JSONL file
    extract_fields : list of str, optional
        Fields to extract (default: all)

    Returns:
    --------
    data : DataFrame
        Parsed Twitter data
    """
    # Try to detect format
    with open(filepath, encoding="utf-8") as f:
        first_line = f.readline()

    # Check if JSONL (one JSON object per line)
    try:
        json.loads(first_line)
        is_jsonl = True
    except Exception:
        is_jsonl = False

    # Load data
    if is_jsonl:
        tweets = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                try:
                    tweets.append(json.loads(line))
                except Exception:
                    continue
    else:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                tweets = data
            elif isinstance(data, dict) and "data" in data:
                tweets = data["data"]
            else:
                tweets = [data]

    # Convert to DataFrame
    df = pd.DataFrame(tweets)

    # Extract specific fields if requested
    if extract_fields:
        available_fields = [f for f in extract_fields if f in df.columns]
        df = df[available_fields]

    # Parse timestamp if present
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])

    return df


def load_csv_posts(
    filepath,
    text_column="text",
    timestamp_column="timestamp",
    user_column="user_id",
    parse_dates=True,
):
    """
    Load social media posts from CSV.

    Parameters:
    -----------
    filepath : str
        Path to CSV file
    text_column : str
        Name of text content column
    timestamp_column : str
        Name of timestamp column
    user_column : str
        Name of user ID column
    parse_dates : bool
        Whether to parse timestamp column

    Returns:
    --------
    data : DataFrame
        Loaded posts
    """
    df = pd.read_csv(filepath)

    # Rename columns if needed
    column_mapping = {}
    if text_column != "text" and text_column in df.columns:
        column_mapping[text_column] = "text"
    if timestamp_column != "timestamp" and timestamp_column in df.columns:
        column_mapping[timestamp_column] = "timestamp"
    if user_column != "user_id" and user_column in df.columns:
        column_mapping[user_column] = "user_id"

    if column_mapping:
        df = df.rename(columns=column_mapping)

    # Parse dates
    if parse_dates and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def generate_synthetic_posts(
    n_users=100,
    n_posts=1000,
    start_date="2024-01-01",
    end_date="2024-12-31",
    sentiment_distribution=None,
    include_network=True,
    seed=42,
):
    """
    Generate synthetic social media posts and network.

    Parameters:
    -----------
    n_users : int
        Number of users
    n_posts : int
        Number of posts
    start_date : str
        Start date for posts
    end_date : str
        End date for posts
    sentiment_distribution : dict, optional
        {'positive': 0.4, 'negative': 0.3, 'neutral': 0.3}
    include_network : bool
        Whether to generate social network
    seed : int
        Random seed

    Returns:
    --------
    data : dict
        Dictionary containing 'posts' DataFrame and optionally 'network' Graph
    """
    np.random.seed(seed)

    # Default sentiment distribution
    if sentiment_distribution is None:
        sentiment_distribution = {"positive": 0.4, "negative": 0.3, "neutral": 0.3}

    # Generate users
    user_ids = [f"user_{i:04d}" for i in range(n_users)]

    # Generate timestamps
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    date_range = (end - start).days
    timestamps = [
        start
        + timedelta(
            days=np.random.randint(0, date_range),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60),
        )
        for _ in range(n_posts)
    ]

    # Sample templates for different sentiments
    positive_templates = [
        "Just had an amazing experience with {topic}! Highly recommend! #happy",
        "Love {topic} so much! Best thing ever! 😊",
        "{topic} is absolutely fantastic! Thank you!",
        "Can't get enough of {topic}! Simply wonderful!",
        "Thrilled with {topic}! Exceeded all expectations!",
    ]

    negative_templates = [
        "Really disappointed with {topic}. Not worth it.",
        "Terrible experience with {topic}. Would not recommend.",
        "{topic} is a complete waste. Very unhappy. 😞",
        "Frustrated with {topic}. Expected much better.",
        "Avoid {topic} at all costs. Worst decision ever.",
    ]

    neutral_templates = [
        "Just tried {topic}. Pretty average, nothing special.",
        "{topic} is okay, I guess. Could be better.",
        "Meh. {topic} didn't really impress me one way or another.",
        "{topic} - it's fine. Not great, not terrible.",
        "Mixed feelings about {topic}. Has pros and cons.",
    ]

    topics = [
        "product",
        "service",
        "app",
        "restaurant",
        "movie",
        "book",
        "event",
        "technology",
        "update",
        "feature",
    ]

    # Generate posts
    posts = []
    for i in range(n_posts):
        # Select sentiment based on distribution
        sentiment = np.random.choice(
            list(sentiment_distribution.keys()), p=list(sentiment_distribution.values())
        )

        # Select template
        if sentiment == "positive":
            template = np.random.choice(positive_templates)
        elif sentiment == "negative":
            template = np.random.choice(negative_templates)
        else:
            template = np.random.choice(neutral_templates)

        # Generate text
        topic = np.random.choice(topics)
        text = template.format(topic=topic)

        # Create post
        post = {
            "post_id": f"post_{i:05d}",
            "user_id": np.random.choice(user_ids),
            "text": text,
            "timestamp": timestamps[i],
            "likes": np.random.poisson(10),
            "retweets": np.random.poisson(3),
            "sentiment": sentiment,
            "topic": topic,
        }

        posts.append(post)

    df = pd.DataFrame(posts)
    df = df.sort_values("timestamp").reset_index(drop=True)

    result = {"posts": df}

    # Generate social network
    if include_network:
        G = nx.barabasi_albert_graph(n_users, m=3, seed=seed)
        # Map node indices to user IDs
        mapping = {i: user_ids[i] for i in range(n_users)}
        G = nx.relabel_nodes(G, mapping)

        result["network"] = G

    return result


def build_interaction_network(
    posts_df, interaction_type="mention", user_column="user_id", text_column="text"
):
    """
    Build social network from posts based on interactions.

    Parameters:
    -----------
    posts_df : DataFrame
        Posts data
    interaction_type : str
        Type of interaction: 'mention', 'reply', 'retweet'
    user_column : str
        Name of user ID column
    text_column : str
        Name of text column

    Returns:
    --------
    G : networkx.DiGraph
        Directed interaction network
    """
    import re

    G = nx.DiGraph()

    # Add all users as nodes
    users = posts_df[user_column].unique()
    G.add_nodes_from(users)

    # Build edges based on interaction type
    for _, row in posts_df.iterrows():
        user = row[user_column]
        text = row[text_column]

        if interaction_type == "mention":
            # Extract @mentions
            mentions = re.findall(r"@(\w+)", text)
            for mentioned in mentions:
                # Add edge from user to mentioned
                if G.has_edge(user, mentioned):
                    G[user][mentioned]["weight"] += 1
                else:
                    G.add_edge(user, mentioned, weight=1)

        elif interaction_type == "reply":
            # Requires 'reply_to_user' column
            if "reply_to_user" in row and pd.notna(row["reply_to_user"]):
                reply_to = row["reply_to_user"]
                if G.has_edge(user, reply_to):
                    G[user][reply_to]["weight"] += 1
                else:
                    G.add_edge(user, reply_to, weight=1)

        elif interaction_type == "retweet":
            # Requires 'retweeted_user' column
            if "retweeted_user" in row and pd.notna(row["retweeted_user"]):
                retweeted = row["retweeted_user"]
                if G.has_edge(user, retweeted):
                    G[user][retweeted]["weight"] += 1
                else:
                    G.add_edge(user, retweeted, weight=1)

    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    return G


def aggregate_by_time(posts_df, time_column="timestamp", freq="1D", agg_columns=None):
    """
    Aggregate posts by time period.

    Parameters:
    -----------
    posts_df : DataFrame
        Posts data
    time_column : str
        Name of timestamp column
    freq : str
        Aggregation frequency ('1H', '1D', '1W', etc.)
    agg_columns : dict, optional
        Columns and aggregation functions, e.g.,
        {'likes': 'sum', 'sentiment': 'mean'}

    Returns:
    --------
    aggregated : DataFrame
        Time-aggregated data
    """
    if time_column not in posts_df.columns:
        raise ValueError(f"Column '{time_column}' not found")

    # Ensure timestamp is datetime
    posts_df = posts_df.copy()
    posts_df[time_column] = pd.to_datetime(posts_df[time_column])

    # Set index
    posts_df = posts_df.set_index(time_column)

    # Default aggregation
    if agg_columns is None:
        agg_columns = {"post_id": "count"}  # Count posts

    # Aggregate
    aggregated = posts_df.resample(freq).agg(agg_columns).reset_index()

    return aggregated


def filter_by_keywords(
    posts_df, keywords, text_column="text", case_sensitive=False, match_all=False
):
    """
    Filter posts by keywords.

    Parameters:
    -----------
    posts_df : DataFrame
        Posts data
    keywords : list of str
        Keywords to search for
    text_column : str
        Name of text column
    case_sensitive : bool
        Whether search is case-sensitive
    match_all : bool
        If True, match all keywords (AND)
        If False, match any keyword (OR)

    Returns:
    --------
    filtered : DataFrame
        Filtered posts
    """
    df = posts_df.copy()

    if not case_sensitive:
        df[text_column] = df[text_column].str.lower()
        keywords = [k.lower() for k in keywords]

    if match_all:
        # Match all keywords (AND)
        mask = pd.Series([True] * len(df))
        for keyword in keywords:
            mask &= df[text_column].str.contains(keyword, na=False)
    else:
        # Match any keyword (OR)
        mask = pd.Series([False] * len(df))
        for keyword in keywords:
            mask |= df[text_column].str.contains(keyword, na=False)

    return posts_df[mask]


def split_train_test(
    posts_df, test_size=0.2, time_based=True, time_column="timestamp", random_state=42
):
    """
    Split data into train and test sets.

    Parameters:
    -----------
    posts_df : DataFrame
        Posts data
    test_size : float
        Fraction of data for testing
    time_based : bool
        If True, split by time (test = most recent)
        If False, random split
    time_column : str
        Name of timestamp column (for time-based split)
    random_state : int
        Random seed

    Returns:
    --------
    train, test : DataFrames
        Train and test sets
    """
    if time_based and time_column in posts_df.columns:
        # Sort by time
        df = posts_df.sort_values(time_column)
        split_idx = int(len(df) * (1 - test_size))
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
    else:
        # Random split
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(posts_df, test_size=test_size, random_state=random_state)

    return train, test


if __name__ == "__main__":
    # Example usage
    print("Social Media Data Loaders")
    print("=" * 60)

    # Generate synthetic data
    print("\n1. Generating synthetic social media data...")
    data = generate_synthetic_posts(n_users=50, n_posts=200, include_network=True, seed=42)

    posts_df = data["posts"]
    network = data["network"]

    print(f"   Posts generated: {len(posts_df)}")
    print(f"   Users: {posts_df['user_id'].nunique()}")
    print(f"   Date range: {posts_df['timestamp'].min()} to {posts_df['timestamp'].max()}")

    # Network statistics
    print("\n2. Social network statistics...")
    print(f"   Nodes: {network.number_of_nodes()}")
    print(f"   Edges: {network.number_of_edges()}")
    print(f"   Avg degree: {sum(dict(network.degree()).values()) / network.number_of_nodes():.2f}")

    # Filter by keywords
    print("\n3. Filtering by keywords...")
    filtered = filter_by_keywords(posts_df, ["amazing", "fantastic"], match_all=False)
    print(f"   Original posts: {len(posts_df)}")
    print(f"   Filtered posts: {len(filtered)}")

    # Time aggregation
    print("\n4. Aggregating by time...")
    daily_stats = aggregate_by_time(
        posts_df, freq="1D", agg_columns={"post_id": "count", "likes": "sum"}
    )
    print(f"   Days with posts: {len(daily_stats)}")
    print(f"   Avg posts per day: {daily_stats['post_id'].mean():.1f}")

    # Train-test split
    print("\n5. Splitting data...")
    train, test = split_train_test(posts_df, test_size=0.2, time_based=True)
    print(f"   Train size: {len(train)}")
    print(f"   Test size: {len(test)}")

    print("\n✓ Data loaders ready")
