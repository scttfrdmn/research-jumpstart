"""
Visualization utilities for sky surveys and classification results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
import warnings


def plot_sky_distribution(ra, dec, values=None, title='Sky Distribution',
                          projection='mollweide', cmap='viridis',
                          figsize=(12, 6)):
    """
    Plot distribution of sources on the sky.

    Parameters
    ----------
    ra : array-like
        Right ascension (degrees)
    dec : array-like
        Declination (degrees)
    values : array-like, optional
        Color values for each source
    title : str
        Plot title
    projection : str
        'mollweide' or 'hammer'
    cmap : str
        Colormap name
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax
        Matplotlib figure and axis
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=projection)

    # Convert RA to [-180, 180] for projection
    ra_plot = np.where(ra > 180, ra - 360, ra)
    ra_plot_rad = np.radians(ra_plot)
    dec_rad = np.radians(dec)

    # Plot points
    if values is not None:
        scatter = ax.scatter(ra_plot_rad, dec_rad, c=values, cmap=cmap,
                           s=1, alpha=0.5)
        plt.colorbar(scatter, ax=ax, label='Value')
    else:
        ax.scatter(ra_plot_rad, dec_rad, s=1, alpha=0.5, color='blue')

    # Grid and labels
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    ax.set_title(title)

    plt.tight_layout()
    return fig, ax


def plot_color_color(catalog, x_color='g_r', y_color='r_i',
                    hue=None, title='Color-Color Diagram',
                    figsize=(10, 8)):
    """
    Plot color-color diagram for object classification.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Catalog with color information
    x_color : str
        Column name for x-axis color
    y_color : str
        Column name for y-axis color
    hue : str, optional
        Column name for color coding (e.g., object type)
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax
        Matplotlib figure and axis
    """
    fig, ax = plt.subplots(figsize=figsize)

    if hue is not None and hue in catalog.columns:
        # Plot with different colors for each class
        for class_name in catalog[hue].unique():
            mask = catalog[hue] == class_name
            ax.scatter(
                catalog.loc[mask, x_color],
                catalog.loc[mask, y_color],
                label=class_name,
                alpha=0.5,
                s=10
            )
        ax.legend()
    else:
        # Simple scatter plot
        ax.scatter(catalog[x_color], catalog[y_color], alpha=0.5, s=10)

    ax.set_xlabel(x_color.replace('_', '-'))
    ax.set_ylabel(y_color.replace('_', '-'))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_confusion_matrix(y_true, y_pred, class_names=None,
                         title='Confusion Matrix', figsize=(10, 8)):
    """
    Plot confusion matrix for classification results.

    Parameters
    ----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    class_names : list, optional
        Names of classes
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax
        Matplotlib figure and axis
    """
    from sklearn.metrics import confusion_matrix

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Set ticks
    if class_names is not None:
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
    else:
        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_yticks(np.arange(cm.shape[0]))

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Add text annotations
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2f})',
                   ha="center", va="center",
                   color="white" if cm_norm[i, j] > thresh else "black",
                   fontsize=9)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)

    plt.tight_layout()
    return fig, ax


def plot_feature_importance(feature_names, importances, top_n=20,
                           title='Feature Importance', figsize=(10, 8)):
    """
    Plot feature importance from tree-based model.

    Parameters
    ----------
    feature_names : list
        Names of features
    importances : array-like
        Feature importance values
    top_n : int
        Number of top features to show
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax
        Matplotlib figure and axis
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(range(top_n), importances[indices])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.invert_yaxis()

    plt.tight_layout()
    return fig, ax


def plot_classification_report(y_true, y_pred, class_names=None,
                               title='Classification Report', figsize=(10, 6)):
    """
    Visualize classification metrics per class.

    Parameters
    ----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    class_names : list, optional
        Names of classes
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax
        Matplotlib figure and axis
    """
    from sklearn.metrics import precision_recall_fscore_support

    # Calculate metrics per class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )

    # Create dataframe
    import pandas as pd
    metrics_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    }, index=class_names if class_names else range(len(precision)))

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(metrics_df))
    width = 0.25

    ax.bar(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
    ax.bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df.index, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1.0])
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig, ax


def plot_match_statistics(matched_catalog, title='Survey Match Statistics',
                          figsize=(10, 6)):
    """
    Visualize cross-match statistics across surveys.

    Parameters
    ----------
    matched_catalog : pandas.DataFrame
        Multi-survey matched catalog
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax
        Matplotlib figure and axis
    """
    # Count sources with data from each survey
    # (Simplified - adjust based on actual column names)

    fig, ax = plt.subplots(figsize=figsize)

    # Placeholder for actual implementation
    ax.text(0.5, 0.5, 'Match statistics visualization\n(to be implemented)',
            ha='center', va='center', fontsize=14)

    ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')

    plt.tight_layout()
    return fig, ax
