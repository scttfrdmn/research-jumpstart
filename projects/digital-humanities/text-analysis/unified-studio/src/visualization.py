"""Visualization functions for text analysis results."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Optional, List, Tuple, Dict
from collections import Counter

logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100


def plot_word_cloud(
    text: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    max_words: int = 100
) -> None:
    """
    Create word cloud visualization (simple frequency-based).

    Args:
        text: Input text
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
        max_words: Maximum number of words to display
    """
    logger.info("Creating word cloud")

    from .text_analysis import extract_keywords

    # Get keywords
    keywords = extract_keywords(text, top_n=max_words)

    if not keywords:
        logger.warning("No keywords found for word cloud")
        return

    # Create simple visualization using bar chart
    words, counts = zip(*keywords[:30])  # Top 30 for readability

    fig, ax = plt.subplots(figsize=figsize)

    # Horizontal bar chart
    y_pos = np.arange(len(words))
    ax.barh(y_pos, counts, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_title('Top Keywords by Frequency', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Word cloud saved to {save_path}")

    plt.show()


def plot_sentiment_timeline(
    df: pd.DataFrame,
    text_column: str = 'text',
    date_column: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot sentiment over time or documents.

    Args:
        df: DataFrame with text and optional date
        text_column: Column containing text
        date_column: Column containing dates (optional)
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    logger.info("Creating sentiment timeline")

    from .text_analysis import analyze_sentiment

    # Calculate sentiment for each document
    sentiments = []
    for text in df[text_column]:
        sentiment = analyze_sentiment(text)
        sentiments.append(sentiment['compound'])

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    if date_column and date_column in df.columns:
        x = pd.to_datetime(df[date_column])
        ax.plot(x, sentiments, marker='o', linestyle='-', alpha=0.7)
        ax.set_xlabel('Date', fontsize=12)
        plt.xticks(rotation=45)
    else:
        x = range(len(sentiments))
        ax.plot(x, sentiments, marker='o', linestyle='-', alpha=0.7)
        ax.set_xlabel('Document Index', fontsize=12)

    ax.set_ylabel('Sentiment (Compound Score)', fontsize=12)
    ax.set_title('Sentiment Timeline', fontsize=14, fontweight='bold')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='Neutral')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Color positive/negative
    colors = ['green' if s > 0 else 'red' for s in sentiments]
    ax.scatter(x, sentiments, c=colors, alpha=0.5, s=50)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Sentiment timeline saved to {save_path}")

    plt.show()


def plot_entity_network(
    entities_dict: Dict[str, List[str]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Visualize entity distribution by type.

    Args:
        entities_dict: Dictionary with entity types and lists
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    logger.info("Creating entity visualization")

    # Count entities by type
    entity_counts = {etype: len(entities) for etype, entities in entities_dict.items()}

    if not entity_counts or sum(entity_counts.values()) == 0:
        logger.warning("No entities to visualize")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Bar chart of entity counts
    types = list(entity_counts.keys())
    counts = list(entity_counts.values())

    ax1.bar(types, counts, alpha=0.7, edgecolor='black')
    ax1.set_title('Entity Count by Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Pie chart of entity distribution
    ax2.pie(counts, labels=types, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Entity Distribution', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Entity network saved to {save_path}")

    plt.show()


def plot_topic_distribution(
    topics_dict: Dict[str, List[str]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """
    Visualize topic modeling results.

    Args:
        topics_dict: Dictionary with topic names and word lists
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    logger.info("Creating topic distribution plot")

    if not topics_dict:
        logger.warning("No topics to visualize")
        return

    n_topics = len(topics_dict)
    fig, axes = plt.subplots(1, min(n_topics, 5), figsize=figsize)

    if n_topics == 1:
        axes = [axes]

    for idx, (topic_name, words) in enumerate(list(topics_dict.items())[:5]):
        if idx >= len(axes):
            break

        # Display top words
        y_pos = np.arange(len(words[:10]))
        axes[idx].barh(y_pos, range(len(words[:10]), 0, -1), alpha=0.7)
        axes[idx].set_yticks(y_pos)
        axes[idx].set_yticklabels(words[:10])
        axes[idx].invert_yaxis()
        axes[idx].set_title(topic_name, fontweight='bold')
        axes[idx].set_xlabel('Importance')
        axes[idx].grid(True, alpha=0.3, axis='x')

    plt.suptitle('Topic Modeling Results', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Topic distribution saved to {save_path}")

    plt.show()


def plot_ngram_frequency(
    ngrams: List[Tuple[str, int]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    top_n: int = 20
) -> None:
    """
    Plot n-gram frequency distribution.

    Args:
        ngrams: List of (ngram, frequency) tuples
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
        top_n: Number of top n-grams to display
    """
    logger.info("Creating n-gram frequency plot")

    if not ngrams:
        logger.warning("No n-grams to visualize")
        return

    # Get top N
    top_ngrams = ngrams[:top_n]
    ngram_texts, frequencies = zip(*top_ngrams)

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(ngram_texts))
    ax.barh(y_pos, frequencies, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ngram_texts)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_title('Top N-grams by Frequency', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"N-gram frequency plot saved to {save_path}")

    plt.show()


def plot_readability_comparison(
    df: pd.DataFrame,
    text_column: str = 'text',
    group_column: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Compare readability scores across documents or groups.

    Args:
        df: DataFrame with text
        text_column: Column containing text
        group_column: Optional column for grouping
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    logger.info("Creating readability comparison")

    from .text_analysis import calculate_readability

    # Calculate readability for each document
    readability_scores = []
    for text in df[text_column]:
        scores = calculate_readability(text)
        readability_scores.append(scores['flesch_reading_ease'])

    df_plot = df.copy()
    df_plot['readability'] = readability_scores

    fig, ax = plt.subplots(figsize=figsize)

    if group_column and group_column in df.columns:
        # Box plot by group
        df_plot.boxplot(column='readability', by=group_column, ax=ax)
        ax.set_title('Readability by Group', fontsize=14, fontweight='bold')
        ax.set_xlabel(group_column, fontsize=12)
        plt.suptitle('')  # Remove automatic title
    else:
        # Histogram
        ax.hist(readability_scores, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(readability_scores), color='red', linestyle='--',
                  label=f'Mean: {np.mean(readability_scores):.1f}')
        ax.set_title('Readability Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Flesch Reading Ease', fontsize=12)
        ax.legend()

    ax.set_ylabel('Flesch Reading Ease Score', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Readability comparison saved to {save_path}")

    plt.show()


def plot_corpus_statistics(
    stats: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Visualize corpus statistics.

    Args:
        stats: Dictionary with corpus statistics
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    logger.info("Creating corpus statistics visualization")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Total documents
    axes[0].bar(['Documents'], [stats.get('total_documents', 0)],
               alpha=0.7, edgecolor='black')
    axes[0].set_title('Total Documents', fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Total words
    axes[1].bar(['Words'], [stats.get('total_words', 0)],
               alpha=0.7, edgecolor='black', color='orange')
    axes[1].set_title('Total Words', fontweight='bold')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Average document length
    axes[2].bar(['Avg Length'], [stats.get('avg_document_length', 0)],
               alpha=0.7, edgecolor='black', color='green')
    axes[2].set_title('Average Document Length', fontweight='bold')
    axes[2].set_ylabel('Words per Document')
    axes[2].grid(True, alpha=0.3, axis='y')

    # Vocabulary size
    axes[3].bar(['Vocabulary'], [stats.get('vocabulary_size', 0)],
               alpha=0.7, edgecolor='black', color='red')
    axes[3].set_title('Vocabulary Size', fontweight='bold')
    axes[3].set_ylabel('Unique Words')
    axes[3].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Corpus Statistics Overview', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Corpus statistics saved to {save_path}")

    plt.show()


def plot_text_similarity_matrix(
    df: pd.DataFrame,
    text_column: str = 'text',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Create similarity matrix heatmap for texts.

    Args:
        df: DataFrame with texts
        text_column: Column containing text
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    logger.info("Creating text similarity matrix")

    from .text_analysis import compare_texts

    n = len(df)
    if n > 50:
        logger.warning(f"Too many documents ({n}), limiting to first 50")
        df = df.head(50)
        n = 50

    # Calculate similarity matrix
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                comparison = compare_texts(
                    df.iloc[i][text_column],
                    df.iloc[j][text_column]
                )
                similarity = comparison['jaccard_similarity']
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(similarity_matrix, annot=False, fmt='.2f',
               cmap='YlOrRd', square=True, ax=ax,
               cbar_kws={'label': 'Jaccard Similarity'})

    ax.set_title('Text Similarity Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Document Index')
    ax.set_ylabel('Document Index')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Similarity matrix saved to {save_path}")

    plt.show()
