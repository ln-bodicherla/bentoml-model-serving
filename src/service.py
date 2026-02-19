"""
BentoML Service definition for fraud detection.

Configures the FraudDetectionService with adaptive micro-batching,
input validation, pre/post processing hooks, and Prometheus metrics.
Supports scikit-learn, XGBoost, and PyTorch model backends.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import bentoml
import numpy as np
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load service configuration from YAML."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config not found at {config_path}, using defaults")
        return {
            "service": {"name": "fraud_detection", "workers": 2, "timeout": 30},
            "batching": {
                "enabled": True,
                "max_batch_size": 32,
                "max_latency_ms": 100,
            },
            "model": {
                "name": "fraud_classifier",
                "version": "latest",
                "framework": "sklearn",
                "feature_names": [
                    "amount", "hour_of_day", "day_of_week",
                    "merchant_category", "transaction_type",
                    "distance_from_home", "distance_from_last_transaction",
                    "ratio_to_median_purchase_price", "used_pin", "online_order",
                ],
            },
        }


class TransactionInput(BaseModel):
    """Single transaction input for fraud scoring."""
    amount: float = Field(..., ge=0, description="Transaction amount in USD")
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    merchant_category: int = Field(..., ge=0, le=50)
    transaction_type: int = Field(..., ge=0, le=5)
    distance_from_home: float = Field(..., ge=0)
    distance_from_last_transaction: float = Field(..., ge=0)
    ratio_to_median_purchase_price: float = Field(..., ge=0)
    used_pin: int = Field(..., ge=0, le=1)
    online_order: int = Field(..., ge=0, le=1)


class BatchInput(BaseModel):
    """Batch of transactions for scoring."""
    transactions: List[TransactionInput]


class PredictionOutput(BaseModel):
    """Single prediction result."""
    transaction_id: Optional[str] = None
    is_fraud: bool
    confidence: float
    risk_score: float
    processing_time_ms: float


class BatchOutput(BaseModel):
    """Batch prediction results."""
    predictions: List[PredictionOutput]
    batch_size: int
    total_processing_time_ms: float


class ProbabilityOutput(BaseModel):
    """Probability scores for fraud classification."""
    probability_legitimate: float
    probability_fraud: float
    risk_tier: str


class ModelInfoOutput(BaseModel):
    """Model metadata response."""
    model_name: str
    framework: str
    version: str
    feature_count: int
    feature_names: List[str]


class HealthOutput(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    uptime_seconds: float
    total_predictions: int


CONFIG = _load_config()
BATCH_CONFIG = CONFIG.get("batching", {})


@bentoml.service(
    name=CONFIG.get("service", {}).get("name", "fraud_detection"),
    traffic={
        "timeout": CONFIG.get("service", {}).get("timeout", 30),
    },
)
class FraudDetectionService:
    """BentoML service for real-time fraud detection."""

    def __init__(self) -> None:
        self.config = CONFIG
        self.model_config = self.config.get("model", {})
        self.feature_names = self.model_config.get("feature_names", [])
        self._start_time = time.time()
        self._prediction_count = 0
        self._model = None
        self._model_version = "unknown"
        self._framework = self.model_config.get("framework", "sklearn")

        self._load_model()

    def _load_model(self) -> None:
        """Load the model from BentoML model store."""
        model_name = self.model_config.get("name", "fraud_classifier")
        try:
            model_ref = bentoml.models.get(f"{model_name}:latest")
            self._model = bentoml.load_model(model_ref.tag)
            self._model_version = str(model_ref.tag.version)
            logger.info(f"Loaded model {model_ref.tag}")
        except bentoml.exceptions.NotFound:
            logger.warning(
                f"Model '{model_name}' not found in BentoML store. "
                "Using fallback threshold-based classifier."
            )
            self._model = None
            self._model_version = "fallback"

    def _preprocess(self, transaction: TransactionInput) -> np.ndarray:
        """Convert a transaction input to a feature vector."""
        features = np.array([
            transaction.amount,
            transaction.hour_of_day,
            transaction.day_of_week,
            transaction.merchant_category,
            transaction.transaction_type,
            transaction.distance_from_home,
            transaction.distance_from_last_transaction,
            transaction.ratio_to_median_purchase_price,
            transaction.used_pin,
            transaction.online_order,
        ], dtype=np.float64).reshape(1, -1)
        return features

    def _preprocess_batch(
        self, transactions: List[TransactionInput]
    ) -> np.ndarray:
        """Convert a batch of transaction inputs to a feature matrix."""
        rows = []
        for txn in transactions:
            rows.append([
                txn.amount,
                txn.hour_of_day,
                txn.day_of_week,
                txn.merchant_category,
                txn.transaction_type,
                txn.distance_from_home,
                txn.distance_from_last_transaction,
                txn.ratio_to_median_purchase_price,
                txn.used_pin,
                txn.online_order,
            ])
        return np.array(rows, dtype=np.float64)

    def _fallback_predict(self, features: np.ndarray) -> np.ndarray:
        """Rule-based fallback when no trained model is available."""
        scores = np.zeros(features.shape[0])
        scores += (features[:, 0] > 500) * 0.3
        scores += (features[:, 5] > 100) * 0.2
        scores += (features[:, 6] > 200) * 0.15
        scores += (features[:, 7] > 3.0) * 0.15
        scores += (1 - features[:, 8]) * 0.1
        scores += features[:, 9] * 0.1
        return np.clip(scores, 0.0, 1.0)

    def _postprocess(
        self,
        probability: float,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Convert raw probability to structured prediction output."""
        is_fraud = probability >= threshold
        risk_score = round(float(probability), 4)
        confidence = round(abs(probability - 0.5) * 2, 4)

        return {
            "is_fraud": is_fraud,
            "confidence": confidence,
            "risk_score": risk_score,
        }

    @bentoml.api()
    def predict(self, transaction: TransactionInput) -> PredictionOutput:
        """Score a single transaction for fraud."""
        start = time.time()
        self._prediction_count += 1

        features = self._preprocess(transaction)

        if self._model is not None:
            try:
                proba = self._model.predict_proba(features)[0, 1]
            except AttributeError:
                pred = self._model.predict(features)
                proba = float(pred[0])
        else:
            proba = float(self._fallback_predict(features)[0])

        result = self._postprocess(proba)
        elapsed_ms = (time.time() - start) * 1000

        return PredictionOutput(
            is_fraud=result["is_fraud"],
            confidence=result["confidence"],
            risk_score=result["risk_score"],
            processing_time_ms=round(elapsed_ms, 3),
        )

    @bentoml.api(
        batchable=BATCH_CONFIG.get("enabled", True),
    )
    def predict_batch(self, batch: BatchInput) -> BatchOutput:
        """Score a batch of transactions with adaptive batching.

        BentoML automatically collects individual requests into batches
        when adaptive batching is enabled, optimizing throughput.
        """
        start = time.time()
        self._prediction_count += len(batch.transactions)

        features = self._preprocess_batch(batch.transactions)

        if self._model is not None:
            try:
                probas = self._model.predict_proba(features)[:, 1]
            except AttributeError:
                probas = self._model.predict(features).astype(float)
        else:
            probas = self._fallback_predict(features)

        predictions = []
        for i, proba in enumerate(probas):
            result = self._postprocess(float(proba))
            predictions.append(
                PredictionOutput(
                    transaction_id=f"txn_{i}",
                    is_fraud=result["is_fraud"],
                    confidence=result["confidence"],
                    risk_score=result["risk_score"],
                    processing_time_ms=0.0,
                )
            )

        elapsed_ms = (time.time() - start) * 1000
        return BatchOutput(
            predictions=predictions,
            batch_size=len(batch.transactions),
            total_processing_time_ms=round(elapsed_ms, 3),
        )

    @bentoml.api()
    def predict_proba(
        self, transaction: TransactionInput
    ) -> ProbabilityOutput:
        """Return probability scores for both classes."""
        features = self._preprocess(transaction)
        self._prediction_count += 1

        if self._model is not None:
            try:
                probas = self._model.predict_proba(features)[0]
                p_legit = float(probas[0])
                p_fraud = float(probas[1])
            except AttributeError:
                pred = float(self._model.predict(features)[0])
                p_fraud = pred
                p_legit = 1.0 - pred
        else:
            p_fraud = float(self._fallback_predict(features)[0])
            p_legit = 1.0 - p_fraud

        if p_fraud < 0.3:
            risk_tier = "low"
        elif p_fraud < 0.6:
            risk_tier = "medium"
        elif p_fraud < 0.85:
            risk_tier = "high"
        else:
            risk_tier = "critical"

        return ProbabilityOutput(
            probability_legitimate=round(p_legit, 4),
            probability_fraud=round(p_fraud, 4),
            risk_tier=risk_tier,
        )

    @bentoml.api()
    def health(self) -> HealthOutput:
        """Service health check endpoint."""
        uptime = time.time() - self._start_time
        return HealthOutput(
            status="healthy",
            model_loaded=self._model is not None,
            uptime_seconds=round(uptime, 2),
            total_predictions=self._prediction_count,
        )

    @bentoml.api()
    def model_info(self) -> ModelInfoOutput:
        """Return metadata about the loaded model."""
        return ModelInfoOutput(
            model_name=self.model_config.get("name", "fraud_classifier"),
            framework=self._framework,
            version=self._model_version,
            feature_count=len(self.feature_names),
            feature_names=self.feature_names,
        )
