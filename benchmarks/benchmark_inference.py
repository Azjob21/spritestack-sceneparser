"""
Benchmark Latency for Scene Construction Pipeline
==================================================
Measures latency of Stage 1 (NER), Stage 2 (Level Generator), and End-to-End Pipeline.
"""

from __future__ import annotations

import time
import numpy as np

from sceneparser.pipeline import SceneConstructionPipeline


def run_benchmark(iterations: int = 10) -> None:
    print("Initializing SceneConstructionPipeline...")
    pipeline = SceneConstructionPipeline()

    prompt = "A dungeon with 2 knights on the left, a dragon on the right, and a gold chest in the center."

    # Warmup
    print("Warming up models...")
    pipeline.run(prompt)

    stage1_latencies = []
    stage2_latencies = []
    e2e_latencies = []

    print(f"Running {iterations} benchmark iterations...")
    for _ in range(iterations):
        t0 = time.perf_counter()
        scene_json = pipeline.parse_only(prompt)
        t1 = time.perf_counter()
        grid = pipeline.generate_only(scene_json)
        t2 = time.perf_counter()

        stage1_latencies.append((t1 - t0) * 1000.0)
        stage2_latencies.append((t2 - t1) * 1000.0)
        e2e_latencies.append((t2 - t0) * 1000.0)

    print("\n--- Latency Benchmark Results ---")
    print(f"Device:                         {pipeline.generator.device}")
    print(f"Stage 1 DistilBERT Loaded:      {pipeline.parser.is_transformer_loaded}")
    print(f"Stage 2 Architect Loaded:       {pipeline.generator.is_loaded}")
    print(f"Iterations:                     {iterations}")
    print(f"Stage 1 NER Average:            {np.mean(stage1_latencies):.2f} ms")
    print(f"Stage 2 Level Gen Average:      {np.mean(stage2_latencies):.2f} ms")
    print(f"End-to-End Pipeline Average:    {np.mean(e2e_latencies):.2f} ms")
    print("---------------------------------\n")


if __name__ == "__main__":
    run_benchmark()
