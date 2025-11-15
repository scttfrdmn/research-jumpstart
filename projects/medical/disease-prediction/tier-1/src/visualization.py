"""
Visualization utilities for medical imaging.

Includes GradCAM, ROC curves, and medical image plotting.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import auc, roc_curve


def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, title: str = "ROC Curve"):
    """
    Plot ROC curve.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        title: Plot title
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: list):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.show()


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for model interpretability.

    Shows which regions of the image the model focuses on.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: PyTorch model
            target_layer: Layer to visualize (e.g., last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Save forward pass activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Save backward pass gradients."""
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_image: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generate GradCAM heatmap.

        Args:
            input_image: Input image tensor
            target_class: Class to visualize

        Returns:
            Heatmap as numpy array
        """
        # Forward pass
        output = self.model(input_image)

        # Backward pass
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()

        # Generate heatmap
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1, keepdim=True)
        heatmap = torch.relu(heatmap)

        # Normalize
        heatmap = heatmap / torch.max(heatmap)

        return heatmap.squeeze().cpu().numpy()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay GradCAM heatmap on original image.

    Args:
        image: Original image
        heatmap: GradCAM heatmap
        alpha: Transparency of overlay

    Returns:
        Overlaid image
    """
    import cv2

    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Convert heatmap to colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlay


if __name__ == "__main__":
    print("Medical Imaging Visualization Utilities")
