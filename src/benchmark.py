"""
Serving performance benchmarks for BentoML.

Measures latency, throughput, and adaptive batching effectiveness.
Provides side-by-side comparison with FastAPI-based serving.
"""

import argparse
import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import yaml

logger = logging.getLogger(__name__)


class ServingBenchmark:
    """Benchmark suite for ML model serving performance."""

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        config_path: str = "configs/config.yaml",
    ):
        self.base_url = base_url.rstrip("/")
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = {}

        self.results: Dict[str, Any] = {}

    def _generate_transaction(self, rng: np.random.RandomState) -> Dict:
        """Generate a single random transaction payload."""
        return {
            "amount": round(float(rng.exponential(100)), 2),
            "hour_of_day": int(rng.randint(0, 24)),
            "day_of_week": int(rng.randint(0, 7)),
            "merchant_category": int(rng.randint(0, 20)),
            "transaction_type": int(rng.randint(0, 5)),
            "distance_from_home": round(float(rng.exponential(30)), 2),
            "distance_from_last_transaction": round(
                float(rng.exponential(50)), 2
            ),
            "ratio_to_median_purchase_price": round(
                float(rng.exponential(1.5)), 2
            ),
            "used_pin": int(rng.randint(0, 2)),
            "online_order": int(rng.randint(0, 2)),
        }

    def _send_predict(self, payload: Dict) -> float:
        """Send a single prediction request and return latency in ms."""
        start = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/predict",
            json=payload,
            timeout=30,
        )
        elapsed = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        return elapsed

    def _send_batch_predict(self, payloads: List[Dict]) -> float:
        """Send a batch prediction request and return latency in ms."""
        start = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/predict_batch",
            json={"transactions": payloads},
            timeout=60,
        )
        elapsed = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        return elapsed

    def benchmark_latency(
        self,
        num_requests: int = 500,
        warmup: int = 20,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Measure end-to-end single-request prediction latency.

        Sends sequential requests and computes p50, p90, p95, p99,
        mean, and standard deviation of latency.
        """
        rng = np.random.RandomState(seed)
        logger.info(f"Running latency benchmark: {num_requests} requests")

        for _ in range(warmup):
            try:
                self._send_predict(self._generate_transaction(rng))
            except Exception:
                pass

        latencies = []
        errors = 0
        for i in range(num_requests):
            payload = self._generate_transaction(rng)
            try:
                latency = self._send_predict(payload)
                latencies.append(latency)
            except Exception as e:
                errors += 1
                if errors <= 3:
                    logger.warning(f"Request {i} failed: {e}")

        if not latencies:
            logger.error("All requests failed")
            return {"error": "All requests failed"}

        latencies.sort()
        result = {
            "num_requests": num_requests,
            "successful": len(latencies),
            "errors": errors,
            "mean_ms": round(statistics.mean(latencies), 3),
            "std_ms": round(statistics.stdev(latencies), 3) if len(latencies) > 1 else 0,
            "p50_ms": round(latencies[len(latencies) // 2], 3),
            "p90_ms": round(latencies[int(len(latencies) * 0.9)], 3),
            "p95_ms": round(latencies[int(len(latencies) * 0.95)], 3),
            "p99_ms": round(latencies[int(len(latencies) * 0.99)], 3),
            "min_ms": round(min(latencies), 3),
            "max_ms": round(max(latencies), 3),
        }

        self.results["latency"] = result
        logger.info(
            f"Latency: p50={result['p50_ms']}ms, "
            f"p99={result['p99_ms']}ms, mean={result['mean_ms']}ms"
        )
        return result

    def benchmark_throughput(
        self,
        duration_seconds: int = 30,
        concurrency: int = 16,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Measure maximum requests per second under concurrent load.

        Sends requests from multiple threads for the specified duration
        and measures aggregate throughput.
        """
        rng = np.random.RandomState(seed)
        logger.info(
            f"Running throughput benchmark: {duration_seconds}s, "
            f"concurrency={concurrency}"
        )

        payloads = [self._generate_transaction(rng) for _ in range(1000)]
        completed = 0
        errors = 0
        all_latencies: List[float] = []
        stop_time = time.time() + duration_seconds

        def worker(worker_id: int) -> Dict[str, Any]:
            nonlocal completed, errors
            local_latencies = []
            local_completed = 0
            local_errors = 0
            idx = worker_id

            while time.time() < stop_time:
                payload = payloads[idx % len(payloads)]
                idx += concurrency
                try:
                    latency = self._send_predict(payload)
                    local_latencies.append(latency)
                    local_completed += 1
                except Exception:
                    local_errors += 1

            return {
                "completed": local_completed,
                "errors": local_errors,
                "latencies": local_latencies,
            }

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(worker, i) for i in range(concurrency)
            ]
            for future in as_completed(futures):
                worker_result = future.result()
                completed += worker_result["completed"]
                errors += worker_result["errors"]
                all_latencies.extend(worker_result["latencies"])

        throughput = completed / duration_seconds if duration_seconds > 0 else 0

        all_latencies.sort()
        result = {
            "duration_seconds": duration_seconds,
            "concurrency": concurrency,
            "total_requests": completed,
            "errors": errors,
            "throughput_rps": round(throughput, 1),
            "mean_latency_ms": round(
                statistics.mean(all_latencies), 3
            ) if all_latencies else 0,
            "p99_latency_ms": round(
                all_latencies[int(len(all_latencies) * 0.99)], 3
            ) if all_latencies else 0,
        }

        self.results["throughput"] = result
        logger.info(
            f"Throughput: {result['throughput_rps']} req/s, "
            f"mean latency={result['mean_latency_ms']}ms"
        )
        return result

    def benchmark_batching(
        self,
        batch_sizes: Optional[List[int]] = None,
        num_iterations: int = 50,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Test adaptive batching effectiveness across batch sizes.

        Compares per-item latency and throughput for different batch sizes
        to demonstrate the benefit of BentoML's adaptive batching.
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32, 64]

        rng = np.random.RandomState(seed)
        logger.info(f"Running batching benchmark: sizes={batch_sizes}")

        results_by_size = {}
        for batch_size in batch_sizes:
            latencies = []
            for _ in range(num_iterations):
                payloads = [
                    self._generate_transaction(rng) for _ in range(batch_size)
                ]
                try:
                    latency = self._send_batch_predict(payloads)
                    latencies.append(latency)
                except Exception as e:
                    logger.warning(f"Batch size {batch_size} failed: {e}")

            if latencies:
                per_item = [l / batch_size for l in latencies]
                results_by_size[str(batch_size)] = {
                    "batch_size": batch_size,
                    "mean_total_ms": round(statistics.mean(latencies), 3),
                    "mean_per_item_ms": round(statistics.mean(per_item), 3),
                    "p99_total_ms": round(
                        sorted(latencies)[int(len(latencies) * 0.99)], 3
                    ) if len(latencies) > 1 else round(latencies[0], 3),
                    "throughput_items_per_sec": round(
                        batch_size * 1000 / statistics.mean(latencies), 1
                    ),
                }

        self.results["batching"] = results_by_size
        return results_by_size

    def compare_with_fastapi(
        self,
        fastapi_url: str = "http://localhost:8000",
        num_requests: int = 200,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Side-by-side comparison with FastAPI-based serving.

        Runs identical workloads against both BentoML and FastAPI
        endpoints and compares latency and throughput.
        """
        rng = np.random.RandomState(seed)
        payloads = [self._generate_transaction(rng) for _ in range(num_requests)]

        logger.info("Benchmarking BentoML endpoint")
        bentoml_latencies = []
        for payload in payloads:
            try:
                latency = self._send_predict(payload)
                bentoml_latencies.append(latency)
            except Exception:
                pass

        logger.info("Benchmarking FastAPI endpoint")
        fastapi_latencies = []
        for payload in payloads:
            try:
                start = time.perf_counter()
                resp = requests.post(
                    f"{fastapi_url}/predict",
                    json=payload,
                    timeout=30,
                )
                elapsed = (time.perf_counter() - start) * 1000
                resp.raise_for_status()
                fastapi_latencies.append(elapsed)
            except Exception:
                pass

        def _summarize(latencies: List[float], label: str) -> Dict[str, float]:
            if not latencies:
                return {"error": f"No successful requests for {label}"}
            latencies.sort()
            return {
                "mean_ms": round(statistics.mean(latencies), 3),
                "p50_ms": round(latencies[len(latencies) // 2], 3),
                "p99_ms": round(
                    latencies[int(len(latencies) * 0.99)], 3
                ),
                "successful": len(latencies),
            }

        comparison = {
            "bentoml": _summarize(bentoml_latencies, "BentoML"),
            "fastapi": _summarize(fastapi_latencies, "FastAPI"),
            "num_requests": num_requests,
        }

        if bentoml_latencies and fastapi_latencies:
            bm = statistics.mean(bentoml_latencies)
            fm = statistics.mean(fastapi_latencies)
            comparison["latency_diff_pct"] = round((fm - bm) / fm * 100, 2)

        self.results["comparison"] = comparison
        return comparison

    def generate_report(
        self,
        output_path: str = "benchmark_results/report.json",
    ) -> Dict[str, Any]:
        """Write all benchmark results to a JSON report."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "base_url": self.base_url,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "results": self.results,
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Benchmark report saved to {output_path}")
        return report


def main():
    parser = argparse.ArgumentParser(description="BentoML serving benchmarks")
    parser.add_argument("--url", default="http://localhost:3000")
    parser.add_argument(
        "--test",
        choices=["latency", "throughput", "batching", "all"],
        default="all",
    )
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--output", default="benchmark_results/report.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    bench = ServingBenchmark(base_url=args.url)

    if args.test in ("latency", "all"):
        bench.benchmark_latency(num_requests=args.requests)

    if args.test in ("throughput", "all"):
        bench.benchmark_throughput(
            duration_seconds=args.duration,
            concurrency=args.concurrency,
        )

    if args.test in ("batching", "all"):
        bench.benchmark_batching()

    bench.generate_report(output_path=args.output)


if __name__ == "__main__":
    main()
