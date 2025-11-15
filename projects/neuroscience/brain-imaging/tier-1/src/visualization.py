"""
Brain and network visualization utilities.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_connectivity_matrix(
    conn_matrix,
    labels=None,
    title="Connectivity Matrix",
    vmin=-1,
    vmax=1,
    cmap="RdBu_r",
    figsize=(10, 9),
):
    """
    Plot connectivity matrix as heatmap.

    Parameters
    ----------
    conn_matrix : ndarray
        Connectivity matrix
    labels : list or None
        ROI labels
    title : str
        Plot title
    vmin, vmax : float
        Color scale limits
    cmap : str
        Colormap
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        conn_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest"
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation", rotation=270, labelpad=20)

    # Labels
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("ROI", fontsize=12)
    ax.set_ylabel("ROI", fontsize=12)

    # Tick labels if provided
    if labels is not None:
        n_rois = len(labels)
        tick_interval = max(n_rois // 10, 1)
        tick_positions = range(0, n_rois, tick_interval)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels([labels[i] for i in tick_positions], rotation=45, ha="right")
        ax.set_yticklabels([labels[i] for i in tick_positions])

    plt.tight_layout()
    return fig, ax


def plot_connectome(conn_matrix, coords, threshold=0.3, node_size=50, title="Brain Connectome"):
    """
    Plot brain connectome (requires Nilearn).

    Parameters
    ----------
    conn_matrix : ndarray
        Connectivity matrix
    coords : ndarray
        ROI coordinates (n_rois, 3)
    threshold : float
        Edge threshold
    node_size : float
        Node size
    title : str
        Plot title

    Returns
    -------
    fig : matplotlib figure
    """
    try:
        from nilearn import plotting
    except ImportError:
        print("Nilearn not installed. Install with: pip install nilearn")
        return None

    # Threshold matrix
    conn_thresholded = conn_matrix.copy()
    conn_thresholded[np.abs(conn_thresholded) < threshold] = 0

    # Plot
    fig = plotting.plot_connectome(
        conn_thresholded,
        coords,
        edge_threshold=threshold,
        node_size=node_size,
        title=title,
        colorbar=True,
    )

    return fig


def plot_brain_network(conn_matrix, network_labels, title="Brain Networks"):
    """
    Plot connectivity matrix organized by networks.

    Parameters
    ----------
    conn_matrix : ndarray
        Connectivity matrix
    network_labels : list or ndarray
        Network assignment for each ROI
    title : str
        Plot title

    Returns
    -------
    fig, ax : matplotlib objects
    """
    network_labels = np.array(network_labels)
    unique_networks = np.unique(network_labels)

    # Sort ROIs by network
    sort_idx = np.argsort(network_labels)
    conn_sorted = conn_matrix[sort_idx, :][:, sort_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 9))

    im = ax.imshow(
        conn_sorted, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", interpolation="nearest"
    )

    # Add network boundaries
    network_boundaries = []
    for network in unique_networks:
        idx = np.where(network_labels[sort_idx] == network)[0]
        if len(idx) > 0:
            network_boundaries.append((idx[0], idx[-1]))

    for _start, end in network_boundaries:
        ax.axhline(y=end + 0.5, color="white", linewidth=2)
        ax.axvline(x=end + 0.5, color="white", linewidth=2)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation", rotation=270, labelpad=20)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("ROI (sorted by network)", fontsize=12)
    ax.set_ylabel("ROI (sorted by network)", fontsize=12)

    plt.tight_layout()
    return fig, ax


def plot_degree_distribution(conn_matrix, threshold=0.3, title="Degree Distribution"):
    """
    Plot node degree distribution.

    Parameters
    ----------
    conn_matrix : ndarray
        Connectivity matrix
    threshold : float
        Edge threshold
    title : str
        Plot title

    Returns
    -------
    fig, ax : matplotlib objects
    """
    # Binarize
    adj_matrix = (np.abs(conn_matrix) > threshold).astype(int)
    np.fill_diagonal(adj_matrix, 0)

    # Compute degrees
    degrees = np.sum(adj_matrix, axis=1)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(degrees, bins=30, color="steelblue", edgecolor="black", alpha=0.7)

    ax.set_xlabel("Degree", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # Add statistics
    ax.axvline(
        np.mean(degrees),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(degrees):.1f}",
    )
    ax.axvline(
        np.median(degrees),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(degrees):.1f}",
    )
    ax.legend()

    plt.tight_layout()
    return fig, ax


def plot_training_history(history, title="Training History"):
    """
    Plot model training history.

    Parameters
    ----------
    history : keras History or dict
        Training history
    title : str
        Plot title

    Returns
    -------
    fig, axes : matplotlib objects
    """
    if hasattr(history, "history"):
        history = history.history

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history["accuracy"], label="Train", linewidth=2)
    if "val_accuracy" in history:
        axes[0].plot(history["val_accuracy"], label="Validation", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    axes[0].set_title("Model Accuracy", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history["loss"], label="Train", linewidth=2)
    if "val_loss" in history:
        axes[1].plot(history["val_loss"], label="Validation", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Loss", fontsize=12, fontweight="bold")
    axes[1].set_title("Model Loss", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    return fig, axes


def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=True, title="Confusion Matrix"):
    """
    Plot confusion matrix.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list or None
        Class labels
    normalize : bool
        If True, normalize by row (true label)
    title : str
        Plot title

    Returns
    -------
    fig, ax : matplotlib objects
    """
    from sklearn.metrics import confusion_matrix

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    # Labels
    if labels is not None:
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

    # Annotate
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("True label", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted label", fontsize=12, fontweight="bold")

    plt.tight_layout()
    return fig, ax


def plot_brain_activation(stat_map, bg_img=None, threshold=2.0, title="Brain Activation"):
    """
    Plot statistical brain map (requires Nilearn).

    Parameters
    ----------
    stat_map : Nifti1Image or str
        Statistical map
    bg_img : Nifti1Image or str or None
        Background anatomical image
    threshold : float
        Display threshold
    title : str
        Plot title

    Returns
    -------
    display : Nilearn display object
    """
    try:
        from nilearn import plotting
    except ImportError:
        print("Nilearn not installed. Install with: pip install nilearn")
        return None

    display = plotting.plot_stat_map(
        stat_map,
        bg_img=bg_img,
        threshold=threshold,
        title=title,
        display_mode="ortho",
        cut_coords=(0, 0, 0),
        colorbar=True,
    )

    return display


def plot_roc_curves(y_true, y_probs, class_names=None, title="ROC Curves"):
    """
    Plot ROC curves for multi-class classification.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_probs : ndarray
        Predicted probabilities (n_samples, n_classes)
    class_names : list or None
        Class names
    title : str
        Plot title

    Returns
    -------
    fig, ax : matplotlib objects
    """
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    n_classes = y_probs.shape[1]

    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    # Compute ROC curve for each class
    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)

        label = f"Class {i}" if class_names is None else class_names[i]
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC = {roc_auc:.2f})")

    # Diagonal line
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Chance")

    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")

    # Generate synthetic connectivity matrix
    n_rois = 100
    conn_matrix = np.random.randn(n_rois, n_rois)
    conn_matrix = (conn_matrix + conn_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(conn_matrix, 1.0)

    # Test connectivity matrix plot
    print("\n1. Connectivity Matrix:")
    fig, ax = plot_connectivity_matrix(conn_matrix, title="Test Connectivity")
    plt.close()
    print("   ✓ Plot generated")

    # Test degree distribution
    print("\n2. Degree Distribution:")
    fig, ax = plot_degree_distribution(conn_matrix, threshold=0.3)
    plt.close()
    print("   ✓ Plot generated")

    print("\n✓ All visualization tests passed!")
