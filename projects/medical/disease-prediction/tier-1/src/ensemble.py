"""
Ensemble learning methods for multi-modal medical imaging.

Combines predictions from X-ray, CT, and MRI models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional


class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple models.

    Supports different fusion strategies:
    - Averaging: Simple or weighted average
    - Voting: Majority voting for classification
    - Stacking: Meta-learner on top of base models
    """

    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        method: str = 'weighted_average'
    ):
        """
        Args:
            models: List of trained PyTorch models
            weights: Optional weights for each model (sum to 1.0)
            method: Ensemble method ('average', 'weighted_average', 'voting', 'stacking')
        """
        self.models = models
        self.method = method

        if weights is None:
            # Equal weights
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
            self.weights = weights

        # Set all models to eval mode
        for model in self.models:
            model.eval()

    def predict(
        self,
        inputs: List[torch.Tensor],
        return_individual: bool = False
    ) -> torch.Tensor:
        """
        Make ensemble prediction.

        Args:
            inputs: List of inputs for each model
            return_individual: Whether to return individual predictions

        Returns:
            Ensemble prediction (and optionally individual predictions)
        """
        if len(inputs) != len(self.models):
            raise ValueError("Number of inputs must match number of models")

        # Get predictions from each model
        individual_preds = []
        with torch.no_grad():
            for model, input_tensor in zip(self.models, inputs):
                pred = model(input_tensor)
                individual_preds.append(pred)

        # Combine predictions based on method
        if self.method in ['average', 'weighted_average']:
            ensemble_pred = self._weighted_average(individual_preds)
        elif self.method == 'voting':
            ensemble_pred = self._majority_voting(individual_preds)
        elif self.method == 'stacking':
            ensemble_pred = self._stacking(individual_preds)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

        if return_individual:
            return ensemble_pred, individual_preds
        return ensemble_pred

    def _weighted_average(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Weighted average of predictions."""
        weighted_preds = [
            pred * weight for pred, weight in zip(predictions, self.weights)
        ]
        return torch.stack(weighted_preds).sum(dim=0)

    def _majority_voting(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Majority voting for classification."""
        # Convert logits to class predictions
        class_preds = [torch.argmax(pred, dim=-1) for pred in predictions]
        stacked = torch.stack(class_preds)

        # Majority vote
        voted, _ = torch.mode(stacked, dim=0)
        return voted

    def _stacking(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Stacking ensemble (requires trained meta-learner).
        For now, falls back to weighted average.
        """
        # TODO: Implement meta-learner
        return self._weighted_average(predictions)

    def update_weights(self, new_weights: List[float]):
        """Update ensemble weights."""
        assert len(new_weights) == len(self.models)
        assert abs(sum(new_weights) - 1.0) < 1e-6
        self.weights = new_weights


class MetaLearner(nn.Module):
    """
    Meta-learner for stacking ensemble.

    Learns optimal combination of base model predictions.
    """

    def __init__(self, num_models: int, num_classes: int, hidden_dim: int = 128):
        """
        Args:
            num_models: Number of base models
            num_classes: Number of output classes
            hidden_dim: Hidden layer dimension
        """
        super(MetaLearner, self).__init__()

        input_dim = num_models * num_classes

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            predictions: List of predictions from base models

        Returns:
            Meta-learner output
        """
        # Concatenate all predictions
        x = torch.cat(predictions, dim=-1)

        # Forward pass
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class UncertaintyEnsemble:
    """
    Ensemble with uncertainty quantification.

    Uses Monte Carlo dropout or ensemble variance to estimate prediction confidence.
    """

    def __init__(self, models: List[nn.Module], num_samples: int = 10):
        """
        Args:
            models: List of trained models
            num_samples: Number of forward passes for MC dropout
        """
        self.models = models
        self.num_samples = num_samples

    def predict_with_uncertainty(
        self,
        inputs: List[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimates.

        Args:
            inputs: List of inputs for each model

        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        all_predictions = []

        # Multiple forward passes with dropout
        for _ in range(self.num_samples):
            predictions = []
            for model, input_tensor in zip(self.models, inputs):
                # Enable dropout during inference
                model.train()
                pred = model(input_tensor)
                predictions.append(pred)
                model.eval()

            # Average predictions across models
            avg_pred = torch.stack(predictions).mean(dim=0)
            all_predictions.append(avg_pred)

        # Calculate mean and variance
        all_predictions = torch.stack(all_predictions)
        mean_pred = all_predictions.mean(dim=0)
        uncertainty = all_predictions.var(dim=0)

        return mean_pred, uncertainty


def optimize_weights(
    models: List[nn.Module],
    val_loader,
    criterion,
    device: str = 'cuda'
) -> List[float]:
    """
    Find optimal ensemble weights using validation set.

    Args:
        models: List of trained models
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run on

    Returns:
        Optimal weights for each model
    """
    from scipy.optimize import minimize

    def objective(weights):
        """Objective function: validation loss."""
        ensemble = EnsemblePredictor(models, weights=list(weights))
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                # Assume inputs is a list for multi-modal
                predictions = ensemble.predict(inputs)
                loss = criterion(predictions, targets)
                total_loss += loss.item()

        return total_loss

    # Initial guess: equal weights
    initial_weights = np.ones(len(models)) / len(models)

    # Constraints: weights sum to 1 and are non-negative
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0) for _ in range(len(models))]

    # Optimize
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x.tolist()
    return optimal_weights


def calculate_contribution_scores(
    models: List[nn.Module],
    inputs: List[torch.Tensor],
    ensemble_pred: torch.Tensor
) -> List[float]:
    """
    Calculate how much each model contributes to the ensemble prediction.

    Args:
        models: List of models
        inputs: List of inputs
        ensemble_pred: Ensemble prediction

    Returns:
        List of contribution scores (0-1) for each model
    """
    contributions = []

    with torch.no_grad():
        for model, input_tensor in zip(models, inputs):
            pred = model(input_tensor)

            # Calculate similarity to ensemble prediction
            # Using cosine similarity
            similarity = torch.cosine_similarity(
                pred.flatten(),
                ensemble_pred.flatten(),
                dim=0
            )
            contributions.append(similarity.item())

    # Normalize to sum to 1
    total = sum(contributions)
    if total > 0:
        contributions = [c / total for c in contributions]

    return contributions


if __name__ == '__main__':
    print("Multi-Modal Medical Imaging Ensemble")
    print("=" * 50)
    print("\\nEnsemble methods available:")
    print("  - EnsemblePredictor: Combine multiple models")
    print("  - MetaLearner: Learned ensemble with neural network")
    print("  - UncertaintyEnsemble: Predictions with confidence")
    print("  - optimize_weights: Find optimal model weights")
    print("  - calculate_contribution_scores: Model importance")
    print("\\nFusion strategies:")
    print("  - Weighted average: Best for regression/probabilities")
    print("  - Majority voting: Best for classification")
    print("  - Stacking: Most powerful, requires meta-learner training")
    print("=" * 50)
