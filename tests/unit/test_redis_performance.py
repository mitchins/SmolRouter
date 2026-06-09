"""
Redis Performance Tests

Validates that FakeRedis can handle production-level throughput and provides
performance baselines for Redis configuration system. These tests ensure
the fallback behavior meets minimum production requirements.

Performance targets:
- TPS >= 100 sustained over 10 seconds
- Concurrent operations without degradation
- Request logging and quota tracking under load
- Circuit breaker functionality under stress
"""

import asyncio
import pytest
import pytest_asyncio
import time
from unittest.mock import patch
import os

from smolrouter.redis_backend import RedisRequestLog, RedisApiKeyQuota, _circuit_breaker
from smolrouter.redis_config import redis_client, is_fake_redis

# redis-py >= 8 defaults the async connection pool to max_connections=100.
# These baselines deliberately fan out hundreds/thousands of operations at
# once, so cap the number of simultaneously in-flight operations below that
# limit. Without this, enough command checkouts overlap on slower CI runners
# to exhaust the shared (Fake)Redis pool, raising MaxConnectionsError and
# tripping the circuit breaker. 64 still drives throughput far above the
# asserted targets while leaving headroom in the pool.
MAX_INFLIGHT_OPS = 64


@pytest.fixture(autouse=True)
def ensure_fakeredis():
    """Ensure tests run with FakeRedis for consistent performance testing"""
    # Force test environment to use FakeRedis
    with patch.dict(os.environ, {"APP_ENV": "test"}):
        # Clear any existing REDIS_URL to force FakeRedis
        if "REDIS_URL" in os.environ:
            original_redis_url = os.environ.pop("REDIS_URL")
        else:
            original_redis_url = None

        yield

        # Restore original REDIS_URL if it existed
        if original_redis_url is not None:
            os.environ["REDIS_URL"] = original_redis_url


@pytest_asyncio.fixture
async def fresh_redis():
    """Provide fresh Redis instance for each performance test"""
    await redis_client.flushall()
    yield redis_client
    await redis_client.flushall()


class TestRedisPerformanceBaselines:
    """Performance tests ensuring FakeRedis meets production requirements"""

    @pytest.mark.asyncio
    async def test_sustained_throughput_100_tps(self, fresh_redis):
        """
        Validate TPS >= 100 sustained over 10 seconds

        This test ensures FakeRedis can handle production-level request rates
        without performance degradation over time.
        """
        assert is_fake_redis(), "Performance tests should run with FakeRedis"

        duration_seconds = 10
        target_tps = 100
        target_total_ops = duration_seconds * target_tps

        print(f"\n🏁 Performance Test: {target_tps} TPS over {duration_seconds}s")
        print(f"Target operations: {target_total_ops}")

        start_time = time.time()
        completed_ops = 0
        sem = asyncio.Semaphore(MAX_INFLIGHT_OPS)

        # Run sustained load test
        async def single_operation(op_id: int):
            nonlocal completed_ops

            async with sem:
                # Simulate realistic production operation: quota increment
                await RedisApiKeyQuota.increment_usage(
                    api_key=f"sk-perf-test-{op_id % 10}",  # 10 unique keys
                    provider_id="openai",
                    model_name="gpt-4",
                    request_count=1,
                    token_count=100 + (op_id % 50),  # Variable token counts
                )
            completed_ops += 1

        # Execute operations with controlled timing
        tasks = []
        for i in range(target_total_ops):
            task = asyncio.create_task(single_operation(i))
            tasks.append(task)

            # Control rate to avoid overwhelming the system
            if i > 0 and i % 50 == 0:
                await asyncio.sleep(0.01)  # Brief pause every 50 operations

        # Wait for all operations to complete
        await asyncio.gather(*tasks)

        end_time = time.time()
        actual_duration = end_time - start_time
        actual_tps = completed_ops / actual_duration

        print(f"✅ Completed {completed_ops} operations in {actual_duration:.2f}s")
        print(f"✅ Actual TPS: {actual_tps:.1f}")
        print(f"✅ Target TPS: {target_tps}")

        # Performance assertions
        assert completed_ops == target_total_ops, f"Expected {target_total_ops} ops, got {completed_ops}"
        assert actual_tps >= target_tps * 0.8, f"TPS {actual_tps:.1f} below 80% of target {target_tps}"
        assert actual_duration <= duration_seconds * 1.5, f"Test took too long: {actual_duration:.2f}s"

        print(f"🎯 PASS: FakeRedis sustained {actual_tps:.1f} TPS (target: {target_tps})")

    @pytest.mark.asyncio
    async def test_concurrent_operations_no_degradation(self, fresh_redis):
        """
        Test concurrent operations maintain performance without degradation

        Validates that concurrent request logging and quota tracking
        don't interfere with each other under load.
        """
        assert is_fake_redis(), "Performance tests should run with FakeRedis"

        concurrent_batches = 20
        ops_per_batch = 25
        total_ops = concurrent_batches * ops_per_batch

        print("\n🔄 Concurrent Operations Test")
        print(f"Batches: {concurrent_batches}, Ops per batch: {ops_per_batch}")
        print(f"Total operations: {total_ops}")

        async def request_logging_batch(batch_id: int):
            """Simulate concurrent request logging"""
            async def one(i: int):
                async with sem:
                    return await RedisRequestLog.create(
                        source_ip=f"192.168.1.{(batch_id * ops_per_batch + i) % 255}",
                        method="POST",
                        path="/v1/chat/completions",
                        service_type="openai",
                        upstream_url=f"https://api.openai.com/batch/{batch_id}/op/{i}",
                    )

            return await asyncio.gather(*[one(i) for i in range(ops_per_batch)])

        async def quota_tracking_batch(batch_id: int):
            """Simulate concurrent quota tracking"""
            async def one(i: int):
                async with sem:
                    return await RedisApiKeyQuota.increment_usage(
                        api_key=f"sk-concurrent-{batch_id}-{i % 5}",
                        provider_id="openai",
                        model_name="gpt-4",
                        request_count=1,
                        token_count=150 + i,
                    )

            return await asyncio.gather(*[one(i) for i in range(ops_per_batch)])

        start_time = time.time()
        sem = asyncio.Semaphore(MAX_INFLIGHT_OPS)

        # Execute both types of operations concurrently
        logging_tasks = [request_logging_batch(i) for i in range(concurrent_batches)]
        quota_tasks = [quota_tracking_batch(i) for i in range(concurrent_batches)]

        # Run all batches concurrently
        logging_results, quota_results = await asyncio.gather(
            asyncio.gather(*logging_tasks), asyncio.gather(*quota_tasks)
        )

        end_time = time.time()
        duration = end_time - start_time
        total_ops_completed = len(logging_results) * ops_per_batch + len(quota_results) * ops_per_batch
        tps = total_ops_completed / duration

        print(f"✅ Completed {total_ops_completed} operations in {duration:.2f}s")
        print(f"✅ Concurrent TPS: {tps:.1f}")

        # Verify all operations completed successfully
        assert len(logging_results) == concurrent_batches
        assert len(quota_results) == concurrent_batches

        # Verify reasonable performance under concurrency
        min_expected_tps = 80  # Lower threshold for concurrent operations
        assert tps >= min_expected_tps, f"Concurrent TPS {tps:.1f} below minimum {min_expected_tps}"

        print(f"🎯 PASS: Concurrent operations achieved {tps:.1f} TPS")

    @pytest.mark.asyncio
    async def test_request_logging_performance(self, fresh_redis):
        """
        Dedicated test for request logging performance under load

        Ensures request logging can handle realistic production request rates.
        """
        assert is_fake_redis(), "Performance tests should run with FakeRedis"

        num_requests = 500
        print(f"\n📝 Request Logging Performance: {num_requests} logs")

        start_time = time.time()
        sem = asyncio.Semaphore(MAX_INFLIGHT_OPS)

        # Create request logs in batches for realistic load pattern
        async def one(i: int):
            async with sem:
                return await RedisRequestLog.create(
                    source_ip=f"10.0.{i // 256}.{i % 256}",
                    method="POST" if i % 2 == 0 else "GET",
                    path=f"/v1/chat/completions/{i}",
                    service_type="openai",
                    upstream_url="https://api.openai.com/v1/chat/completions",
                    original_model="gpt-4",
                    mapped_model="gpt-4",
                    request_size=1000 + (i % 500),
                )

        request_ids = await asyncio.gather(*[one(i) for i in range(num_requests)])

        end_time = time.time()
        duration = end_time - start_time
        tps = num_requests / duration

        print(f"✅ Created {len(request_ids)} request logs in {duration:.2f}s")
        print(f"✅ Request logging TPS: {tps:.1f}")

        # Verify all requests were logged
        assert len(request_ids) == num_requests
        assert all(req_id for req_id in request_ids)

        # Performance assertion
        min_logging_tps = 50
        assert tps >= min_logging_tps, f"Request logging TPS {tps:.1f} below minimum {min_logging_tps}"

        print(f"🎯 PASS: Request logging achieved {tps:.1f} TPS")

    @pytest.mark.asyncio
    async def test_quota_tracking_performance(self, fresh_redis):
        """
        Dedicated test for quota tracking performance under load

        Validates atomic quota operations can handle production API key usage patterns.
        """
        assert is_fake_redis(), "Performance tests should run with FakeRedis"

        num_api_keys = 20
        ops_per_key = 50
        total_ops = num_api_keys * ops_per_key

        print(f"\n📊 Quota Tracking Performance: {total_ops} operations")
        print(f"API keys: {num_api_keys}, Operations per key: {ops_per_key}")

        start_time = time.time()
        sem = asyncio.Semaphore(MAX_INFLIGHT_OPS)

        # Simulate realistic quota tracking patterns
        async def one(key_id: int, op_id: int):
            async with sem:
                return await RedisApiKeyQuota.increment_usage(
                    api_key=f"sk-quota-perf-{key_id:03d}",
                    provider_id="openai",
                    model_name=f"gpt-{3.5 if op_id % 2 == 0 else 4}",
                    request_count=1,
                    token_count=50 + (op_id * 10),
                )

        tasks = [one(key_id, op_id) for key_id in range(num_api_keys) for op_id in range(ops_per_key)]

        quota_results = await asyncio.gather(*tasks)

        end_time = time.time()
        duration = end_time - start_time
        tps = total_ops / duration

        print(f"✅ Processed {len(quota_results)} quota operations in {duration:.2f}s")
        print(f"✅ Quota tracking TPS: {tps:.1f}")

        # Verify all operations completed
        assert len(quota_results) == total_ops
        assert all(result for result in quota_results)

        # Validate quota data integrity
        sample_result = quota_results[0]
        assert "requests_today" in sample_result
        assert "tokens_today" in sample_result
        assert sample_result["requests_today"] > 0
        assert sample_result["tokens_today"] > 0

        # Performance assertion
        min_quota_tps = 60
        assert tps >= min_quota_tps, f"Quota tracking TPS {tps:.1f} below minimum {min_quota_tps}"

        print(f"🎯 PASS: Quota tracking achieved {tps:.1f} TPS")

    @pytest.mark.asyncio
    async def test_mixed_workload_performance(self, fresh_redis):
        """
        Test mixed workload performance simulating real production usage

        Combines request logging, quota tracking, and data retrieval operations
        to simulate realistic production load patterns.
        """
        assert is_fake_redis(), "Performance tests should run with FakeRedis"

        num_cycles = 100
        print(f"\n🔀 Mixed Workload Performance: {num_cycles} cycles")

        start_time = time.time()
        sem = asyncio.Semaphore(MAX_INFLIGHT_OPS)

        async def mixed_operation_cycle(cycle_id: int):
            """Single cycle of mixed operations"""
            async with sem:
                # 1. Create request log
                request_id = await RedisRequestLog.create(
                    source_ip=f"172.16.{cycle_id % 255}.1",
                    method="POST",
                    path="/v1/chat/completions",
                    service_type="openai",
                    upstream_url="https://api.openai.com/v1/chat/completions",
                )

                # 2. Track quota usage
                quota_result = await RedisApiKeyQuota.increment_usage(
                    api_key=f"sk-mixed-{cycle_id % 10}",
                    provider_id="openai",
                    model_name="gpt-4",
                    request_count=1,
                    token_count=200 + cycle_id,
                )

                # 3. Retrieve request data (read operation)
                request_data = await RedisRequestLog.get_by_id(request_id)

                # 4. Complete the request
                await RedisRequestLog.update_completion(
                    request_id=request_id, status_code=200, response_size=500 + cycle_id, error_message=None
                )

            return {"request_id": request_id, "quota": quota_result, "request_data": request_data}

        # Execute all cycles concurrently
        tasks = [mixed_operation_cycle(i) for i in range(num_cycles)]
        results = await asyncio.gather(*tasks)

        end_time = time.time()
        duration = end_time - start_time
        operations_per_cycle = 4  # create, increment, get, update
        total_ops = num_cycles * operations_per_cycle
        tps = total_ops / duration

        print(f"✅ Completed {num_cycles} mixed cycles ({total_ops} total ops) in {duration:.2f}s")
        print(f"✅ Mixed workload TPS: {tps:.1f}")

        # Verify all cycles completed successfully
        assert len(results) == num_cycles
        assert all(result["request_id"] for result in results)
        assert all(result["quota"] for result in results)
        assert all(result["request_data"] for result in results)

        # Performance assertion for mixed workload
        min_mixed_tps = 40  # Lower due to complexity of mixed operations
        assert tps >= min_mixed_tps, f"Mixed workload TPS {tps:.1f} below minimum {min_mixed_tps}"

        print(f"🎯 PASS: Mixed workload achieved {tps:.1f} TPS")

    @pytest.mark.asyncio
    async def test_circuit_breaker_does_not_interfere(self, fresh_redis):
        """
        Ensure circuit breaker doesn't interfere with normal FakeRedis operations

        Validates that circuit breaker remains closed during normal operations
        and doesn't add performance overhead.
        """
        assert is_fake_redis(), "Performance tests should run with FakeRedis"

        num_operations = 200
        print(f"\n🛡️  Circuit Breaker Non-Interference: {num_operations} operations")

        # Reset circuit breaker to known state
        _circuit_breaker.failure_count = 0
        _circuit_breaker.state = "CLOSED"

        start_time = time.time()
        sem = asyncio.Semaphore(MAX_INFLIGHT_OPS)

        # Perform operations that should succeed with FakeRedis
        async def one(i: int):
            async with sem:
                return await RedisApiKeyQuota.increment_usage(
                    api_key=f"sk-circuit-test-{i % 5}",
                    provider_id="openai",
                    model_name="gpt-4",
                    request_count=1,
                    token_count=100,
                )

        results = await asyncio.gather(*[one(i) for i in range(num_operations)])

        end_time = time.time()
        duration = end_time - start_time
        tps = num_operations / duration

        print(f"✅ Completed {len(results)} operations in {duration:.2f}s")
        print(f"✅ TPS with circuit breaker: {tps:.1f}")
        print(f"✅ Circuit breaker state: {_circuit_breaker.state}")
        print(f"✅ Circuit breaker failures: {_circuit_breaker.failure_count}")

        # Verify circuit breaker didn't interfere
        assert _circuit_breaker.state == "CLOSED", "Circuit breaker should remain closed"
        assert _circuit_breaker.failure_count == 0, "No failures should be recorded with FakeRedis"
        assert len(results) == num_operations
        assert all(result for result in results)

        # Performance should be similar to other tests
        min_tps = 50
        assert tps >= min_tps, f"Circuit breaker overhead too high: {tps:.1f} TPS"

        print(f"🎯 PASS: Circuit breaker non-interference verified at {tps:.1f} TPS")


@pytest.mark.performance
class TestRedisConfigurationPerformance:
    """Performance tests for Redis configuration system behaviors"""

    @pytest.mark.asyncio
    async def test_fakeredis_startup_time(self):
        """Validate FakeRedis has fast startup time for development workflow"""

        print("\n⚡ FakeRedis Startup Performance Test")

        startup_iterations = 10
        startup_times = []

        for i in range(startup_iterations):
            start_time = time.time()

            # Simulate application startup Redis operations
            await redis_client.ping()
            await redis_client.set("startup_test", f"iteration_{i}")
            await redis_client.get("startup_test")
            await redis_client.delete("startup_test")

            end_time = time.time()
            startup_times.append(end_time - start_time)

        avg_startup_time = sum(startup_times) / len(startup_times)
        max_startup_time = max(startup_times)

        print(f"✅ Average startup time: {avg_startup_time * 1000:.2f}ms")
        print(f"✅ Max startup time: {max_startup_time * 1000:.2f}ms")

        # Startup performance assertions
        assert avg_startup_time < 0.01, f"Average startup too slow: {avg_startup_time * 1000:.2f}ms"
        assert max_startup_time < 0.05, f"Max startup too slow: {max_startup_time * 1000:.2f}ms"

        print("🎯 PASS: FakeRedis startup performance excellent")

    def test_configuration_detection_performance(self):
        """Ensure Redis configuration detection is fast"""

        print("\n🔍 Configuration Detection Performance")

        detection_iterations = 100
        start_time = time.time()

        for i in range(detection_iterations):
            # These should be very fast as they're just checking variables
            fake_status = is_fake_redis()
            assert fake_status is True  # Should be using FakeRedis in tests

        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_check = total_time / detection_iterations

        print(f"✅ {detection_iterations} configuration checks in {total_time * 1000:.2f}ms")
        print(f"✅ Average time per check: {avg_time_per_check * 1000000:.2f}μs")

        # Configuration detection should be extremely fast
        assert avg_time_per_check < 0.0001, f"Configuration detection too slow: {avg_time_per_check * 1000000:.2f}μs"

        print("🎯 PASS: Configuration detection performance excellent")


if __name__ == "__main__":
    print("Redis Performance Tests")
    print("======================")
    print("Run with: pytest tests/unit/test_redis_performance.py -v")
    print("Run performance tests only: pytest -m performance tests/unit/test_redis_performance.py -v")
