"""Concurrent load test for the CPU-bound /api/v1/rebalance endpoint.

This script reuses the realistic payload generator from scripts/profile_rebalance_api.py
so the outbound requests match what the backend expects.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import httpx

# Ensure project root (so imports like app.* resolve)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.profile_rebalance_api import create_realistic_test_request  # noqa: E402

DEFAULT_URL = "http://localhost:8100/api/v1/rebalance"


@dataclass
class CallResult:
    idx: int
    ok: bool
    status_code: Optional[int]
    latency: float
    end_time: float
    error: Optional[str] = None


def build_payload(customer_id: Optional[int]) -> dict:
    request_model = create_realistic_test_request(customer_id=customer_id)
    return request_model.model_dump(mode="json")


async def invoke_rebalance(
    client: httpx.AsyncClient,
    url: str,
    idx: int,
    payload: dict,
    semaphore: asyncio.Semaphore,
) -> CallResult:
    started = time.perf_counter()
    async with semaphore:
        try:
            response = await client.post(url, json=payload)
            finished = time.perf_counter()
            latency = finished - started
            ok = response.status_code == 200
            print(
                f"[request {idx:03d}] status={response.status_code} "
                f"latency={latency:.3f}s"
            )
            return CallResult(
                idx=idx,
                ok=ok,
                status_code=response.status_code,
                latency=latency,
                end_time=finished,
                error=None if ok else response.text,
            )
        except Exception as exc:  # noqa: BLE001 - surface transport failures
            finished = time.perf_counter()
            latency = finished - started
            print(f"[request {idx:03d}] ERROR after {latency:.3f}s: {exc}")
            return CallResult(
                idx=idx,
                ok=False,
                status_code=None,
                latency=latency,
                end_time=finished,
                error=str(exc),
            )


def summarize(results: List[CallResult], total_time: float, batch_start: float) -> None:
    successes = sum(1 for r in results if r.ok)
    failures = len(results) - successes
    latencies = [r.latency for r in results]
    print("\n=== Concurrent Rebalance Summary ===")
    print(f"Total elapsed (wall): {total_time:.2f}s")
    print(f"Requests sent:        {len(results)}")
    print(f"Successes:            {successes}")
    print(f"Failures:             {failures}")
    if latencies:
        print(f"Mean latency:         {statistics.mean(latencies):.2f}s")
        print(f"Median latency:       {statistics.median(latencies):.2f}s")
        print(f"Min/Max latency:      {min(latencies):.2f}s / {max(latencies):.2f}s")
    if failures:
        print("\n--- Failure details ---")
        for r in results:
            if not r.ok:
                print(f"#{r.idx:03d} | status={r.status_code} | {r.error}")

    # Detailed per-request timing (relative to batch_start)
    print("\n--- Per-request timings (seconds since batch start) ---")
    print("idx  sent_at  resp_at  latency")
    for r in sorted(results, key=lambda x: x.idx):
        sent_at = r.end_time - r.latency - batch_start
        resp_at = r.end_time - batch_start
        print(f"{r.idx:03d}  {sent_at:7.3f}  {resp_at:7.3f}  {r.latency:7.3f}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Fire concurrent rebalance requests")
    parser.add_argument("--url", default=DEFAULT_URL, help="Rebalance endpoint URL")
    parser.add_argument(
        "--requests", type=int, default=10, help="Total identical requests to send"
    )
    parser.add_argument(
        "--concurrency", type=int, default=10, help="Maximum in-flight requests"
    )
    parser.add_argument(
        "--customer-id",
        type=int,
        default=None,
        help="Optional customer_id for payload builder",
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="Per-request timeout in seconds"
    )
    args = parser.parse_args()

    print("Preparing payload...")
    payload = build_payload(customer_id=args.customer_id)

    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(
        max_connections=args.concurrency * 2, max_keepalive_connections=args.concurrency
    )
    timeout = httpx.Timeout(args.timeout)

    started = time.perf_counter()
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        tasks = [
            asyncio.create_task(
                invoke_rebalance(
                    client=client,
                    url=args.url,
                    idx=i,
                    payload=payload,
                    semaphore=semaphore,
                )
            )
            for i in range(1, args.requests + 1)
        ]
        results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - started
    summarize(results, total_time, started)


if __name__ == "__main__":
    asyncio.run(main())
