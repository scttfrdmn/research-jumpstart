#!/usr/bin/env python3
"""
Model Utilities for Medical Image Classification

Common CNN architectures, transfer learning, and training utilities.

Usage:
    from model_utils import create_model, train_epoch

    model = create_model('resnet50', num_classes=2, pretrained=True)
    loss = train_epoch(model, train_loader, optimizer, criterion, device)
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm.auto import tqdm


def create_model(architecture="resnet50", num_classes=2, pretrained=True, freeze_backbone=False):
    """
    Create a classification model with transfer learning.

    Parameters:
    -----------
    architecture : str
        Model architecture ('resnet50', 'efficientnet_b0', 'densenet121', etc.)
    num_classes : int
        Number of output classes
    pretrained : bool
        Whether to use pretrained ImageNet weights
    freeze_backbone : bool
        Whether to freeze backbone layers (only train classifier head)

    Returns:
    --------
    model : torch.nn.Module
        Classification model
    """
    # Use timm for wide model selection
    if architecture in timm.list_models():
        model = timm.create_model(architecture, pretrained=pretrained, num_classes=num_classes)
    else:
        # Fallback to torchvision
        if architecture == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif architecture == "densenet121":
            model = models.densenet121(pretrained=pretrained)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif architecture == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    # Optionally freeze backbone
    if freeze_backbone:
        for name, param in model.named_parameters():
            # Keep classifier/fc/head trainable
            if not any(x in name for x in ["fc", "classifier", "head"]):
                param.requires_grad = False

    return model


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for improved predictions.

    Averages predictions from multiple architectures.
    """

    def __init__(self, models_list):
        """
        Parameters:
        -----------
        models_list : list of torch.nn.Module
            List of models to ensemble
        """
        super().__init__()
        self.models = nn.ModuleList(models_list)

    def forward(self, x):
        """Average predictions from all models."""
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        return torch.stack(outputs).mean(dim=0)


def train_epoch(model, dataloader, optimizer, criterion, device, use_amp=False):
    """
    Train model for one epoch.

    Parameters:
    -----------
    model : torch.nn.Module
        Model to train
    dataloader : DataLoader
        Training data loader
    optimizer : torch.optim.Optimizer
        Optimizer
    criterion : loss function
        Loss function (e.g., nn.CrossEntropyLoss())
    device : torch.device
        Device to train on
    use_amp : bool
        Whether to use automatic mixed precision

    Returns:
    --------
    avg_loss : float
        Average loss for the epoch
    accuracy : float
        Training accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Setup AMP if requested
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({"loss": total_loss / (pbar.n + 1), "acc": 100.0 * correct / total})

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation/test set.

    Parameters:
    -----------
    model : torch.nn.Module
        Model to evaluate
    dataloader : DataLoader
        Validation/test data loader
    criterion : loss function
        Loss function
    device : torch.device
        Device to evaluate on

    Returns:
    --------
    avg_loss : float
        Average loss
    accuracy : float
        Accuracy
    all_preds : list
        All predictions
    all_labels : list
        All ground truth labels
    all_probs : list
        All prediction probabilities
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(dataloader, desc="Evaluating")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        # Get predictions
        probs = F.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        # Track metrics
        total_loss += loss.item()
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Store for detailed analysis
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({"loss": total_loss / (pbar.n + 1), "acc": 100.0 * correct / total})

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, all_preds, all_labels, all_probs


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """
    Save model checkpoint.

    Parameters:
    -----------
    model : torch.nn.Module
        Model to save
    optimizer : torch.optim.Optimizer
        Optimizer state
    epoch : int
        Current epoch
    loss : float
        Training loss
    accuracy : float
        Validation accuracy
    filepath : str
        Output checkpoint path
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "accuracy": accuracy,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint.

    Parameters:
    -----------
    filepath : str
        Checkpoint file path
    model : torch.nn.Module
        Model to load weights into
    optimizer : torch.optim.Optimizer, optional
        Optimizer to load state into

    Returns:
    --------
    epoch : int
        Epoch of checkpoint
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"]


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focuses training on hard examples by down-weighting easy ones.

    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    """

    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        """
        Parameters:
        -----------
        alpha : float or list
            Weighting factor for class balance
        gamma : float
            Focusing parameter (higher = more focus on hard examples)
        reduction : str
            'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Parameters:
        -----------
        inputs : torch.Tensor
            Model predictions (logits), shape [batch_size, num_classes]
        targets : torch.Tensor
            Ground truth labels, shape [batch_size]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def get_optimizer(model, optimizer_name="adam", learning_rate=1e-3, weight_decay=1e-4):
    """
    Create optimizer.

    Parameters:
    -----------
    model : torch.nn.Module
        Model to optimize
    optimizer_name : str
        'adam', 'adamw', or 'sgd'
    learning_rate : float
        Initial learning rate
    weight_decay : float
        L2 regularization strength

    Returns:
    --------
    optimizer : torch.optim.Optimizer
    """
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name="cosine", num_epochs=100):
    """
    Create learning rate scheduler.

    Parameters:
    -----------
    optimizer : torch.optim.Optimizer
        Optimizer to schedule
    scheduler_name : str
        'cosine', 'step', or 'plateau'
    num_epochs : int
        Total number of training epochs

    Returns:
    --------
    scheduler : torch.optim.lr_scheduler
    """
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


if __name__ == "__main__":
    # Example usage
    print("Medical Image Classification Model Utilities")
    print("=" * 50)

    # Check available models
    print("\n1. Available architectures:")
    popular_models = ["resnet50", "efficientnet_b0", "densenet121", "vit_base_patch16_224"]
    for arch in popular_models:
        available = arch in timm.list_models()
        print(f"   {arch}: {'✓' if available else '✗'}")

    # Create model
    print("\n2. Creating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model("resnet50", num_classes=2, pretrained=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Test forward pass
    print("\n3. Testing forward pass...")
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")

    # Create optimizer and scheduler
    print("\n4. Creating optimizer and scheduler...")
    optimizer = get_optimizer(model, "adam", learning_rate=1e-3)
    scheduler = get_scheduler(optimizer, "cosine", num_epochs=50)
    print(f"   Optimizer: {type(optimizer).__name__}")
    print(f"   Scheduler: {type(scheduler).__name__}")

    print("\n✓ Model utilities ready")
