# BentoML Model Serving

ML model serving using BentoML. Features adaptive micro-batching, model packaging (Bentos), containerization, and deployment support. Provides a unified serving interface for multiple ML frameworks including scikit-learn, XGBoost, PyTorch, and Hugging Face Transformers.

## Overview

This project demonstrates ML model serving with BentoML's opinionated framework. The example use case is a fraud detection service that supports single and batch predictions with automatic adaptive batching, input validation, and Prometheus metrics export. The demo models are trained on generated synthetic data.

## Alternative Approach Note

This is an alternative to the custom FastAPI-based ML microservice approach. BentoML provides built-in adaptive batching, model versioning, and containerization out of the box. However, the custom FastAPI approach was chosen for the primary project because it offers greater flexibility in implementing A/B testing logic, custom drift monitoring integration, and fine-grained control over the request pipeline. BentoML is the better choice when rapid deployment with sensible defaults is the priority.

## Key Features

- **Adaptive Micro-Batching**: Automatically batches individual requests for GPU/CPU throughput optimization
- **Multi-Framework Support**: Unified serving for scikit-learn, XGBoost, PyTorch, and Transformers models
- **Model Packaging**: Build portable Bento archives with all dependencies
- **Containerization**: One-command Docker image generation from Bento
- **Custom Runners**: Specialized runners for XGBoost, PyTorch, and ensemble models
- **Benchmarking**: Latency, throughput, and batching effectiveness measurement

## Project Structure

```
bentoml-model-serving/
├── src/
│   ├── service.py            # BentoML service definition
│   ├── model_packaging.py    # Model saving and Bento building
│   ├── runners.py            # Custom BentoML runners
│   └── benchmark.py          # Serving performance benchmarks
├── docker/
│   └── bentofile.yaml        # Bento build configuration
├── configs/
│   └── config.yaml           # Service configuration
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

# Save a model to BentoML store
python -m src.model_packaging --action save --framework sklearn

# Start the service
bentoml serve src.service:FraudDetectionService

# Run benchmarks
python -m src.benchmark --url http://localhost:3000
```

## API Endpoints

| Method | Endpoint        | Description                          |
|--------|----------------|--------------------------------------|
| POST   | /predict        | Single fraud prediction              |
| POST   | /predict_batch  | Batch prediction with adaptive batching |
| POST   | /predict_proba  | Probability scores                   |
| GET    | /health         | Service health check                 |
| GET    | /model_info     | Model metadata                       |

## Containerization

```bash
# Build Bento
python -m src.model_packaging --action build

# Containerize
python -m src.model_packaging --action containerize

# Run container
docker run -p 3000:3000 fraud_detection:latest
```

## Benchmarks

No benchmark results are published in this repository. With the service running, `python -m src.benchmark --url http://localhost:3000` measures request latency percentiles, throughput under concurrent load (`--concurrency`, `--duration`) and batching behaviour, and writes a JSON report to `benchmark_results/report.json` (use `--test latency|throughput|batching|all` to pick tests). Results depend on your hardware and model.

## License

MIT License
