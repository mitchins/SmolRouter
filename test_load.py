#!/usr/bin/env python3
"""
Load test for SmolRouter dummy provider
Target: 100 requests per second for 10 seconds (1000 total requests)
"""

import asyncio
import aiohttp
import time
from typing import Dict, Any


async def make_request(session: aiohttp.ClientSession, request_id: int) -> Dict[str, Any]:
    """Make a single request to the dummy provider"""
    url = "http://localhost:8103/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer test-key"}
    data = {
        "model": "dummy-realistic-4.0",
        "messages": [{"role": "user", "content": f"Test request {request_id}"}],
        "max_tokens": 50,
    }

    start_time = time.time()
    try:
        async with session.post(url, json=data, headers=headers) as response:
            result = await response.json()
            end_time = time.time()
            return {
                "id": request_id,
                "status": response.status,
                "duration": end_time - start_time,
                "success": response.status == 200,
                "error": result.get("error") if response.status != 200 else None,
            }
    except Exception as e:
        end_time = time.time()
        return {"id": request_id, "status": 0, "duration": end_time - start_time, "success": False, "error": str(e)}


async def run_load_test(target_rps: int = 100, duration_seconds: int = 10):
    """Run the load test"""
    total_requests = target_rps * duration_seconds
    request_interval = 1.0 / target_rps  # Time between request starts

    print(f"Starting load test: {target_rps} RPS for {duration_seconds} seconds ({total_requests} total requests)")
    print(f"Request interval: {request_interval * 1000:.2f}ms")
    print("-" * 60)

    # Create session with connection pooling
    connector = aiohttp.TCPConnector(limit=200, limit_per_host=200)
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        start_time = time.time()

        # Schedule all requests with proper spacing
        for i in range(total_requests):
            # Calculate when this request should start
            scheduled_time = start_time + (i * request_interval)
            current_time = time.time()

            # Wait if we're ahead of schedule
            if scheduled_time > current_time:
                await asyncio.sleep(scheduled_time - current_time)

            # Launch request (don't await it)
            task = asyncio.create_task(make_request(session, i))
            tasks.append(task)

            # Print progress every 100 requests
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                actual_rps = (i + 1) / elapsed
                print(
                    f"Launched {i + 1}/{total_requests} requests | Elapsed: {elapsed:.2f}s | Actual RPS: {actual_rps:.1f}"
                )

        # Wait for all requests to complete
        print("\nAll requests launched, waiting for completion...")
        results = await asyncio.gather(*tasks)

    # Analyze results
    end_time = time.time()
    total_duration = end_time - start_time

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    durations = [r["duration"] for r in results]

    print("\n" + "=" * 60)
    print("LOAD TEST RESULTS")
    print("=" * 60)
    print(f"Total requests: {len(results)}")
    print(f"Successful: {len(successful)} ({len(successful) / len(results) * 100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed) / len(results) * 100:.1f}%)")
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Actual RPS: {len(results) / total_duration:.1f}")

    if durations:
        durations.sort()
        print("\nResponse times:")
        print(f"  Min: {min(durations) * 1000:.2f}ms")
        print(f"  Max: {max(durations) * 1000:.2f}ms")
        print(f"  Avg: {sum(durations) / len(durations) * 1000:.2f}ms")
        print(f"  P50: {durations[len(durations) // 2] * 1000:.2f}ms")
        print(f"  P95: {durations[int(len(durations) * 0.95)] * 1000:.2f}ms")
        print(f"  P99: {durations[int(len(durations) * 0.99)] * 1000:.2f}ms")

    if failed:
        print("\nFirst 5 failures:")
        for failure in failed[:5]:
            print(f"  Request {failure['id']}: {failure['error']}")

    return results


if __name__ == "__main__":
    # Run the test
    asyncio.run(run_load_test(target_rps=100, duration_seconds=10))
