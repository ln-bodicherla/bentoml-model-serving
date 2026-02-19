"""
Custom BentoML runners for specialized model serving.

Implements runners for XGBoost with batching configuration,
PyTorch models with GPU support, and an ensemble runner that
combines predictions from multiple models.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import bentoml
import numpy as np

logger = logging.getLogger(__name__)


class XGBoostRunner:
    """Custom runner for XGBoost models with batching configuration.

    Provides optimized inference with configurable batch sizes and
    thread allocation for XGBoost's internal parallelism.
    """

    def __init__(
        self,
        model_tag: str = "fraud_classifier_xgb:latest",
        n_threads: int = 4,
        max_batch_size: int = 64,
    ):
        self.model_tag = model_tag
        self.n_threads = n_threads
        self.max_batch_size = max_batch_size
        self._model = None
        self._is_setup = False

    def setup(self) -> None:
        """Load the XGBoost model from BentoML store and configure threads."""
        try:
            model_ref = bentoml.models.get(self.model_tag)
            self._model = bentoml.xgboost.load_model(model_ref.tag)
            if hasattr(self._model, "set_params"):
                self._model.set_params(n_jobs=self.n_threads)
            self._is_setup = True
            logger.info(
                f"XGBoostRunner setup complete: {self.model_tag}, "
                f"threads={self.n_threads}"
            )
        except bentoml.exceptions.NotFound:
            logger.warning(f"XGBoost model not found: {self.model_tag}")
            self._model = None
            self._is_setup = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Run prediction on a single sample or small batch."""
        if not self._is_setup:
            self.setup()

        if self._model is None:
            return np.array([0.5] * features.shape[0])

        if features.ndim == 1:
            features = features.reshape(1, -1)

        try:
            probas = self._model.predict_proba(features)[:, 1]
        except AttributeError:
            probas = self._model.predict(features).astype(float)

        return probas

    def batch_predict(
        self, features: np.ndarray
    ) -> np.ndarray:
        """Run prediction on a large batch, splitting if necessary."""
        if not self._is_setup:
            self.setup()

        n_samples = features.shape[0]
        if n_samples <= self.max_batch_size:
            return self.predict(features)

        results = []
        for start in range(0, n_samples, self.max_batch_size):
            end = min(start + self.max_batch_size, n_samples)
            batch = features[start:end]
            results.append(self.predict(batch))

        return np.concatenate(results)


class PyTorchRunner:
    """Custom runner for PyTorch models with GPU support.

    Handles device placement, inference mode, and optional half-precision
    for GPU acceleration.
    """

    def __init__(
        self,
        model_tag: str = "fraud_classifier_torch:latest",
        device: str = "auto",
        use_half_precision: bool = False,
        max_batch_size: int = 128,
    ):
        self.model_tag = model_tag
        self._requested_device = device
        self.use_half_precision = use_half_precision
        self.max_batch_size = max_batch_size
        self._model = None
        self._device = None
        self._is_setup = False

    def setup(self) -> None:
        """Load the PyTorch model and move to appropriate device."""
        import torch

        if self._requested_device == "auto":
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = torch.device(self._requested_device)

        try:
            model_ref = bentoml.models.get(self.model_tag)
            self._model = bentoml.pytorch.load_model(model_ref.tag)
            self._model.to(self._device)
            self._model.eval()

            if self.use_half_precision and self._device.type == "cuda":
                self._model.half()

            self._is_setup = True
            logger.info(
                f"PyTorchRunner setup: {self.model_tag} on {self._device}"
            )
        except bentoml.exceptions.NotFound:
            logger.warning(f"PyTorch model not found: {self.model_tag}")
            self._model = None
            self._is_setup = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Run inference on a feature array."""
        import torch

        if not self._is_setup:
            self.setup()

        if self._model is None:
            return np.full(features.shape[0], 0.5)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        tensor = torch.FloatTensor(features).to(self._device)
        if self.use_half_precision and self._device.type == "cuda":
            tensor = tensor.half()

        with torch.inference_mode():
            output = self._model(tensor)
            probas = output.squeeze(-1).cpu().numpy()

        return probas

    def batch_predict(self, features: np.ndarray) -> np.ndarray:
        """Run inference on a large batch with automatic chunking."""
        if not self._is_setup:
            self.setup()

        n_samples = features.shape[0]
        if n_samples <= self.max_batch_size:
            return self.predict(features)

        results = []
        for start in range(0, n_samples, self.max_batch_size):
            end = min(start + self.max_batch_size, n_samples)
            results.append(self.predict(features[start:end]))

        return np.concatenate(results)


class EnsembleRunner:
    """Runner that combines predictions from multiple models.

    Supports weighted averaging, voting, and stacking strategies
    for combining model outputs.
    """

    def __init__(
        self,
        runners: Optional[List[Any]] = None,
        weights: Optional[List[float]] = None,
        strategy: str = "weighted_average",
    ):
        self.runners = runners or []
        self.weights = weights
        self.strategy = strategy
        self._is_setup = False

        if self.weights is not None:
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]

    def add_runner(self, runner: Any, weight: float = 1.0) -> None:
        """Add a runner to the ensemble."""
        self.runners.append(runner)
        if self.weights is None:
            self.weights = [1.0] * len(self.runners)
        else:
            self.weights.append(weight)
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]

    def setup(self) -> None:
        """Initialize all child runners."""
        for runner in self.runners:
            if hasattr(runner, "setup"):
                runner.setup()
        self._is_setup = True
        logger.info(
            f"EnsembleRunner setup: {len(self.runners)} models, "
            f"strategy={self.strategy}"
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Run prediction across all models and combine results."""
        if not self._is_setup:
            self.setup()

        if not self.runners:
            return np.full(features.shape[0], 0.5)

        all_predictions = []
        for runner in self.runners:
            preds = runner.predict(features)
            all_predictions.append(preds)

        predictions_matrix = np.stack(all_predictions, axis=0)

        if self.strategy == "weighted_average":
            return self._weighted_average(predictions_matrix)
        elif self.strategy == "majority_vote":
            return self._majority_vote(predictions_matrix)
        elif self.strategy == "max_confidence":
            return self._max_confidence(predictions_matrix)
        else:
            return self._weighted_average(predictions_matrix)

    def batch_predict(self, features: np.ndarray) -> np.ndarray:
        """Run batch prediction across all models."""
        if not self._is_setup:
            self.setup()

        if not self.runners:
            return np.full(features.shape[0], 0.5)

        all_predictions = []
        for runner in self.runners:
            if hasattr(runner, "batch_predict"):
                preds = runner.batch_predict(features)
            else:
                preds = runner.predict(features)
            all_predictions.append(preds)

        predictions_matrix = np.stack(all_predictions, axis=0)
        return self._weighted_average(predictions_matrix)

    def _weighted_average(self, predictions: np.ndarray) -> np.ndarray:
        """Combine predictions using weighted averaging."""
        weights = self.weights or [1.0 / len(self.runners)] * len(self.runners)
        weights_arr = np.array(weights).reshape(-1, 1)
        return np.sum(predictions * weights_arr, axis=0)

    def _majority_vote(
        self, predictions: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        """Combine predictions using majority voting."""
        binary = (predictions >= threshold).astype(float)
        vote_fractions = np.mean(binary, axis=0)
        return vote_fractions

    def _max_confidence(self, predictions: np.ndarray) -> np.ndarray:
        """Select the prediction with highest confidence (distance from 0.5)."""
        confidence = np.abs(predictions - 0.5)
        best_model_idx = np.argmax(confidence, axis=0)
        n_samples = predictions.shape[1]
        return predictions[best_model_idx, np.arange(n_samples)]

    def get_individual_predictions(
        self, features: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Get predictions from each model individually for analysis."""
        if not self._is_setup:
            self.setup()

        results = {}
        for i, runner in enumerate(self.runners):
            name = getattr(runner, "model_tag", f"model_{i}")
            results[name] = runner.predict(features)
        return results

    def get_prediction_agreement(
        self, features: np.ndarray, threshold: float = 0.5
    ) -> float:
        """Calculate the fraction of samples where all models agree."""
        individual = self.get_individual_predictions(features)
        if not individual:
            return 1.0

        binary_preds = [
            (preds >= threshold).astype(int)
            for preds in individual.values()
        ]
        stacked = np.stack(binary_preds, axis=0)
        unanimous = np.all(stacked == stacked[0:1], axis=0)
        return float(np.mean(unanimous))
