"""Visualization utilities for social media analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx

def plot_sentiment_distribution(df: pd.DataFrame, save_path: str = None):
    """Plot sentiment distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df['sentiment'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Sentiment Distribution')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_topic_evolution(topics_over_time: pd.DataFrame, save_path: str = None):
    """Plot topic trends over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    topics_over_time.plot(ax=ax)
    ax.set_title('Topic Evolution Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Topic Prevalence')
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_network_graph(G: nx.Graph, save_path: str = None):
    """Plot network graph."""
    fig, ax = plt.subplots(figsize=(12, 12))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax=ax, node_size=50, with_labels=False, alpha=0.6)
    ax.set_title('Social Network Graph')
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_engagement_analysis(df: pd.DataFrame, save_path: str = None):
    """Plot engagement metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df['total_engagement'] = df[['likes', 'retweets', 'replies']].sum(axis=1)
    df['total_engagement'].hist(bins=50, ax=ax)
    ax.set_title('Engagement Distribution')
    ax.set_xlabel('Total Engagement')
    ax.set_ylabel('Frequency')
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
