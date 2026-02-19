"""
Model packaging utilities for BentoML.

Handles saving models from multiple frameworks (scikit-learn, XGBoost,
PyTorch, Transformers) to the BentoML model store, building portable
Bento archives, and generating Docker container images.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import bentoml
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class ModelPackager:
    """Manages model lifecycle in the BentoML model store."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = {"model": {"name": "fraud_classifier"}}

        self.model_config = self.config.get("model", {})
        self.model_name = self.model_config.get("name", "fraud_classifier")

    def save_sklearn_model(
        self,
        model: Optional[Any] = None,
        custom_objects: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bentoml.Tag:
        """Save a scikit-learn model to the BentoML model store.

        If no model is provided, trains a demonstration Random Forest
        classifier on synthetic fraud data.
        """
        if model is None:
            model = self._train_demo_sklearn_model()

        labels = {"framework": "sklearn", "task": "fraud_detection"}
        meta = metadata or {}
        meta.update({
            "feature_names": self.model_config.get("feature_names", []),
            "model_type": type(model).__name__,
        })

        saved_model = bentoml.sklearn.save_model(
            self.model_name,
            model,
            custom_objects=custom_objects or {},
            metadata=meta,
            labels=labels,
        )

        logger.info(f"Saved sklearn model: {saved_model.tag}")
        return saved_model.tag

    def save_xgboost_model(
        self,
        model: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bentoml.Tag:
        """Save an XGBoost model to the BentoML model store."""
        if model is None:
            model = self._train_demo_xgboost_model()

        meta = metadata or {}
        meta.update({
            "feature_names": self.model_config.get("feature_names", []),
            "model_type": "XGBClassifier",
        })

        saved_model = bentoml.xgboost.save_model(
            f"{self.model_name}_xgb",
            model,
            metadata=meta,
            labels={"framework": "xgboost", "task": "fraud_detection"},
        )

        logger.info(f"Saved XGBoost model: {saved_model.tag}")
        return saved_model.tag

    def save_pytorch_model(
        self,
        model: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bentoml.Tag:
        """Save a PyTorch model to the BentoML model store with a custom runner."""
        import torch
        import torch.nn as nn

        if model is None:
            model = self._build_demo_pytorch_model()

        meta = metadata or {}
        meta.update({
            "framework": "pytorch",
            "input_size": 10,
            "hidden_size": 64,
        })

        saved_model = bentoml.pytorch.save_model(
            f"{self.model_name}_torch",
            model,
            metadata=meta,
            labels={"framework": "pytorch", "task": "fraud_detection"},
        )

        logger.info(f"Saved PyTorch model: {saved_model.tag}")
        return saved_model.tag

    def save_transformers_model(
        self,
        model_name_or_path: str = "distilbert-base-uncased",
        task: str = "text-classification",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bentoml.Tag:
        """Save a Hugging Face Transformers model to the BentoML store."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=2
        )

        meta = metadata or {}
        meta.update({
            "source_model": model_name_or_path,
            "task": task,
            "framework": "transformers",
        })

        saved_model = bentoml.transformers.save_model(
            f"{self.model_name}_transformer",
            model,
            custom_objects={"tokenizer": tokenizer},
            metadata=meta,
            labels={"framework": "transformers", "task": task},
        )

        logger.info(f"Saved transformers model: {saved_model.tag}")
        return saved_model.tag

    def build_bento(
        self,
        bentofile_path: str = "docker/bentofile.yaml",
    ) -> str:
        """Package the service and model into a portable Bento archive.

        The Bento includes the service code, model artifacts, Python
        dependencies, and Docker configuration.
        """
        if not Path(bentofile_path).exists():
            raise FileNotFoundError(f"Bentofile not found: {bentofile_path}")

        bento = bentoml.bentos.build(
            service="src.service:FraudDetectionService",
            build_ctx=".",
            bentofile=bentofile_path,
        )

        logger.info(f"Built Bento: {bento.tag}")
        logger.info(f"Bento path: {bento.path}")
        return str(bento.tag)

    def containerize(
        self,
        bento_tag: Optional[str] = None,
        image_tag: Optional[str] = None,
    ) -> str:
        """Build a Docker image from a Bento archive.

        Uses BentoML's built-in containerization which generates an
        optimized Dockerfile and builds the image.
        """
        if bento_tag is None:
            bentos = bentoml.bentos.list()
            if not bentos:
                raise RuntimeError(
                    "No Bentos found. Run build_bento() first."
                )
            bento_tag = str(bentos[0].tag)

        docker_tag = image_tag or bento_tag.replace(":", "_")

        bentoml.container.build(
            bento_tag,
            image_tag=[docker_tag],
            progress="plain",
        )

        logger.info(f"Built Docker image: {docker_tag}")
        return docker_tag

    def list_models(self) -> List[Dict[str, Any]]:
        """List all models in the BentoML store with metadata."""
        models = bentoml.models.list()
        result = []
        for model_info in models:
            result.append({
                "tag": str(model_info.tag),
                "module": model_info.module,
                "creation_time": str(model_info.creation_time),
                "labels": dict(model_info.labels) if model_info.labels else {},
            })
        logger.info(f"Found {len(result)} models in store")
        return result

    def get_model_info(self, model_tag: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve detailed model metadata and signatures."""
        if model_tag is None:
            model_tag = f"{self.model_name}:latest"

        try:
            model_info = bentoml.models.get(model_tag)
        except bentoml.exceptions.NotFound:
            logger.error(f"Model not found: {model_tag}")
            return {"error": f"Model not found: {model_tag}"}

        info = {
            "tag": str(model_info.tag),
            "module": model_info.module,
            "creation_time": str(model_info.creation_time),
            "labels": dict(model_info.labels) if model_info.labels else {},
            "metadata": dict(model_info.metadata) if model_info.metadata else {},
            "path": str(model_info.path),
        }
        return info

    def _train_demo_sklearn_model(self) -> Any:
        """Train a demonstration scikit-learn model on synthetic data."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(42)
        n_samples = 5000
        n_features = 10

        X = rng.randn(n_samples, n_features)
        fraud_score = (
            0.3 * X[:, 0] + 0.25 * X[:, 5] + 0.2 * X[:, 6]
            + 0.15 * X[:, 7] + 0.1 * rng.randn(n_samples)
        )
        y = (fraud_score > np.percentile(fraud_score, 95)).astype(int)

        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ])
        pipeline.fit(X, y)

        logger.info(
            f"Trained demo sklearn model: {n_samples} samples, "
            f"{y.sum()} fraud cases"
        )
        return pipeline

    def _train_demo_xgboost_model(self) -> Any:
        """Train a demonstration XGBoost model on synthetic data."""
        import xgboost as xgb

        rng = np.random.RandomState(42)
        n_samples = 5000
        n_features = 10

        X = rng.randn(n_samples, n_features)
        fraud_score = (
            0.3 * X[:, 0] + 0.25 * X[:, 5] + 0.2 * X[:, 6]
            + 0.15 * X[:, 7] + 0.1 * rng.randn(n_samples)
        )
        y = (fraud_score > np.percentile(fraud_score, 95)).astype(int)

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=float(np.sum(y == 0) / np.sum(y == 1)),
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X, y)

        logger.info("Trained demo XGBoost model")
        return model

    def _build_demo_pytorch_model(self) -> Any:
        """Build a demonstration PyTorch fraud detection network."""
        import torch
        import torch.nn as nn

        class FraudNet(nn.Module):
            def __init__(self, input_size: int = 10, hidden_size: int = 64):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.BatchNorm1d(hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size // 2, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.network(x)

        model = FraudNet(input_size=10, hidden_size=64)
        model.eval()
        return model


def main():
    parser = argparse.ArgumentParser(description="BentoML model packaging")
    parser.add_argument(
        "--action",
        choices=["save", "build", "containerize", "list", "info"],
        required=True,
        help="Action to perform",
    )
    parser.add_argument(
        "--framework",
        choices=["sklearn", "xgboost", "pytorch", "transformers"],
        default="sklearn",
        help="ML framework for save action",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-tag", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    packager = ModelPackager(config_path=args.config)

    if args.action == "save":
        if args.framework == "sklearn":
            packager.save_sklearn_model()
        elif args.framework == "xgboost":
            packager.save_xgboost_model()
        elif args.framework == "pytorch":
            packager.save_pytorch_model()
        elif args.framework == "transformers":
            packager.save_transformers_model()

    elif args.action == "build":
        packager.build_bento()

    elif args.action == "containerize":
        packager.containerize(bento_tag=args.model_tag)

    elif args.action == "list":
        models = packager.list_models()
        for m in models:
            print(f"  {m['tag']} ({m.get('module', 'unknown')})")

    elif args.action == "info":
        info = packager.get_model_info(args.model_tag)
        for k, v in info.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
