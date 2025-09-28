"""
Unit tests for load balancer functionality.

Tests the model instance load balancing logic with various naming patterns.
"""

import pytest
import asyncio
from unittest.mock import patch

from smolrouter.load_balancer import ModelLoadBalancer, DistributionStrategy


class TestModelLoadBalancer:
    """Test the ModelLoadBalancer class functionality."""

    @pytest.fixture
    def load_balancer(self):
        """Create a fresh load balancer for each test."""
        return ModelLoadBalancer()

    def test_parse_model_name_colon_format(self, load_balancer):
        """Test parsing LM Studio colon format (model:instance)."""
        # Base model (no instance)
        assert load_balancer.parse_model_name("gpt-oss-20b") == ("gpt-oss-20b", 0)

        # Instance formats
        assert load_balancer.parse_model_name("gpt-oss-20b:2") == ("gpt-oss-20b", 2)
        assert load_balancer.parse_model_name("llama3-8b:3") == ("llama3-8b", 3)
        assert load_balancer.parse_model_name("mistral-7b:10") == ("mistral-7b", 10)

    def test_parse_model_name_dash_format(self, load_balancer):
        """Test parsing dash format (model-instance)."""
        # Should parse these as instances
        assert load_balancer.parse_model_name("my-custom-model-2") == ("my-custom-model", 2)
        assert load_balancer.parse_model_name("local-llm-3") == ("local-llm", 3)

        # Should NOT parse these as instances (common model names)
        assert load_balancer.parse_model_name("gpt-4") == ("gpt-4", 0)
        assert load_balancer.parse_model_name("llama-7b") == ("llama-7b", 0)
        assert load_balancer.parse_model_name("mistral-7b") == ("mistral-7b", 0)
        assert load_balancer.parse_model_name("qwen-2.5") == ("qwen-2.5", 0)

    def test_parse_model_name_edge_cases(self, load_balancer):
        """Test edge cases in model name parsing."""
        # Complex names
        assert load_balancer.parse_model_name("microsoft/DialoGPT-medium:2") == ("microsoft/DialoGPT-medium", 2)
        assert load_balancer.parse_model_name("huggingface/CodeBERTa-small-v1") == ("huggingface/CodeBERTa-small-v1", 0)

        # Numbers in base name
        assert load_balancer.parse_model_name("gpt-3.5-turbo:2") == ("gpt-3.5-turbo", 2)
        assert load_balancer.parse_model_name("claude-2.1") == ("claude-2.1", 0)

    def test_register_model_instance(self, load_balancer):
        """Test registering model instances."""
        # Register base model
        load_balancer.register_model_instance("gpt-oss-20b", "provider1", "http://localhost:1234")

        # Register instances
        load_balancer.register_model_instance("gpt-oss-20b:2", "provider1", "http://localhost:1234")
        load_balancer.register_model_instance("gpt-oss-20b:3", "provider1", "http://localhost:1234")

        # Check they're grouped correctly
        instances = load_balancer.instances["gpt-oss-20b"]
        assert len(instances) == 3

        model_ids = [i.model_id for i in instances]
        assert "gpt-oss-20b" in model_ids
        assert "gpt-oss-20b:2" in model_ids
        assert "gpt-oss-20b:3" in model_ids

    def test_register_duplicate_instance(self, load_balancer):
        """Test that duplicate registrations are ignored."""
        load_balancer.register_model_instance("gpt-oss-20b", "provider1", "http://localhost:1234")
        load_balancer.register_model_instance("gpt-oss-20b", "provider1", "http://localhost:1234")  # Duplicate

        instances = load_balancer.instances["gpt-oss-20b"]
        assert len(instances) == 1  # Should not duplicate

    def test_get_available_instances_healthy_only(self, load_balancer):
        """Test that only healthy instances are returned."""
        # Register instances
        load_balancer.register_model_instance("gpt-oss-20b", "provider1", "http://localhost:1234")
        load_balancer.register_model_instance("gpt-oss-20b:2", "provider1", "http://localhost:1234")

        # Mark one as unhealthy
        instances = load_balancer.instances["gpt-oss-20b"]
        instances[1].is_healthy = False

        # Should only return healthy instances
        available = load_balancer.get_available_instances("gpt-oss-20b")
        assert len(available) == 1
        assert available[0].model_id == "gpt-oss-20b"

    def test_get_available_instances_sorted_by_load(self, load_balancer):
        """Test that instances are sorted by load (active requests)."""
        # Register instances
        load_balancer.register_model_instance("gpt-oss-20b", "provider1", "http://localhost:1234")
        load_balancer.register_model_instance("gpt-oss-20b:2", "provider1", "http://localhost:1234")
        load_balancer.register_model_instance("gpt-oss-20b:3", "provider1", "http://localhost:1234")

        # Set different loads
        instances = load_balancer.instances["gpt-oss-20b"]
        instances[0].active_requests = 5  # High load
        instances[1].active_requests = 0  # No load
        instances[2].active_requests = 2  # Medium load

        # Should return sorted by load (lowest first)
        available = load_balancer.get_available_instances("gpt-oss-20b")
        loads = [i.active_requests for i in available]
        assert loads == [0, 2, 5]

    @pytest.mark.asyncio
    async def test_select_instance_least_busy(self, load_balancer):
        """Test that select_instance chooses the least busy instance."""
        # Register instances with different loads
        load_balancer.register_model_instance("gpt-oss-20b", "provider1", "http://localhost:1234")
        load_balancer.register_model_instance("gpt-oss-20b:2", "provider1", "http://localhost:1234")

        instances = load_balancer.instances["gpt-oss-20b"]
        instances[0].active_requests = 3
        instances[1].active_requests = 1

        # Should select the less busy instance
        selected = await load_balancer.select_instance("gpt-oss-20b")
        assert selected.active_requests == 1
        assert selected.model_id == "gpt-oss-20b:2"

    @pytest.mark.asyncio
    async def test_select_instance_no_available(self, load_balancer):
        """Test select_instance when no instances are available."""
        # No instances registered
        selected = await load_balancer.select_instance("nonexistent-model")
        assert selected is None

        # All instances unhealthy
        load_balancer.register_model_instance("gpt-oss-20b", "provider1", "http://localhost:1234")
        load_balancer.instances["gpt-oss-20b"][0].is_healthy = False

        selected = await load_balancer.select_instance("gpt-oss-20b")
        assert selected is None

    @pytest.mark.asyncio
    async def test_start_end_request_tracking(self, load_balancer):
        """Test request tracking with start and end."""
        load_balancer.register_model_instance("gpt-oss-20b", "provider1", "http://localhost:1234")
        instance = load_balancer.instances["gpt-oss-20b"][0]

        # Start request
        await load_balancer.start_request(instance)
        assert instance.active_requests == 1
        assert instance.total_requests == 1

        # End request
        await load_balancer.end_request(instance, response_time=1.5, success=True)
        assert instance.active_requests == 0
        assert instance.avg_response_time == 1.5

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, load_balancer):
        """Test handling multiple concurrent requests."""
        load_balancer.register_model_instance("gpt-oss-20b", "provider1", "http://localhost:1234")
        instance = load_balancer.instances["gpt-oss-20b"][0]

        # Start multiple requests
        await load_balancer.start_request(instance)
        await load_balancer.start_request(instance)
        await load_balancer.start_request(instance)

        assert instance.active_requests == 3

        # End requests
        await load_balancer.end_request(instance, 1.0, True)
        await load_balancer.end_request(instance, 2.0, True)

        assert instance.active_requests == 1
        assert instance.total_requests == 3

    @pytest.mark.asyncio
    async def test_response_time_averaging(self, load_balancer):
        """Test that response times are averaged correctly."""
        load_balancer.register_model_instance("gpt-oss-20b", "provider1", "http://localhost:1234")
        instance = load_balancer.instances["gpt-oss-20b"][0]

        # Process multiple requests with different response times
        await load_balancer.start_request(instance)
        await load_balancer.end_request(instance, 1.0, True)

        await load_balancer.start_request(instance)
        await load_balancer.end_request(instance, 3.0, True)

        # Average should be (1.0 + 3.0) / 2 = 2.0
        assert instance.avg_response_time == 2.0

    @pytest.mark.asyncio
    async def test_failure_handling(self, load_balancer):
        """Test handling of failed requests."""
        load_balancer.register_model_instance("gpt-oss-20b", "provider1", "http://localhost:1234")
        instance = load_balancer.instances["gpt-oss-20b"][0]

        # Successful request
        await load_balancer.start_request(instance)
        await load_balancer.end_request(instance, 1.0, success=True)

        # Failed request
        await load_balancer.start_request(instance)
        await load_balancer.end_request(instance, 0.5, success=False)

        # Check stats
        assert load_balancer.stats.total_requests == 2
        assert load_balancer.stats.successful_requests == 1
        assert load_balancer.stats.failed_requests == 1

    def test_mark_instance_unhealthy(self, load_balancer):
        """Test marking instances as unhealthy."""
        load_balancer.register_model_instance("gpt-oss-20b", "provider1", "http://localhost:1234")
        instance = load_balancer.instances["gpt-oss-20b"][0]

        assert instance.is_healthy is True

        load_balancer.mark_instance_unhealthy(instance)
        assert instance.is_healthy is False

        load_balancer.mark_instance_healthy(instance)
        assert instance.is_healthy is True

    def test_get_stats(self, load_balancer):
        """Test getting load balancer statistics."""
        # Register instances
        load_balancer.register_model_instance("gpt-oss-20b", "provider1", "http://localhost:1234")
        load_balancer.register_model_instance("gpt-oss-20b:2", "provider1", "http://localhost:1234")

        # Set some stats
        load_balancer.stats.total_requests = 10
        load_balancer.stats.successful_requests = 8
        load_balancer.stats.failed_requests = 2

        stats = load_balancer.get_stats()

        assert stats["total_requests"] == 10
        assert stats["successful_requests"] == 8
        assert stats["failed_requests"] == 2
        assert stats["success_rate"] == 0.8
        assert "gpt-oss-20b" in stats["instances"]
        assert len(stats["instances"]["gpt-oss-20b"]) == 2

    def test_get_model_groups(self, load_balancer):
        """Test getting model groups."""
        # Register different model families
        load_balancer.register_model_instance("gpt-oss-20b", "provider1", "http://localhost:1234")
        load_balancer.register_model_instance("gpt-oss-20b:2", "provider1", "http://localhost:1234")
        load_balancer.register_model_instance("llama3-8b", "provider1", "http://localhost:1234")
        load_balancer.register_model_instance("llama3-8b:2", "provider1", "http://localhost:1234")

        groups = load_balancer.get_model_groups()

        assert "gpt-oss-20b" in groups
        assert "llama3-8b" in groups
        assert set(groups["gpt-oss-20b"]) == {"gpt-oss-20b", "gpt-oss-20b:2"}
        assert set(groups["llama3-8b"]) == {"llama3-8b", "llama3-8b:2"}

    @pytest.mark.asyncio
    async def test_load_balancing_round_robin_behavior(self, load_balancer):
        """Test that load balancing distributes requests evenly."""
        # Register multiple instances
        load_balancer.register_model_instance("gpt-oss-20b", "provider1", "http://localhost:1234")
        load_balancer.register_model_instance("gpt-oss-20b:2", "provider1", "http://localhost:1234")
        load_balancer.register_model_instance("gpt-oss-20b:3", "provider1", "http://localhost:1234")

        # Track which instances get selected
        selected_instances = []

        # Simulate 6 requests with realistic async behavior
        async def simulate_request():
            instance = await load_balancer.select_instance("gpt-oss-20b")
            await load_balancer.start_request(instance)
            selected_instances.append(instance.model_id)

            # Simulate processing time (realistic request duration)
            await asyncio.sleep(0.05)  # 50ms processing time

            await load_balancer.end_request(instance, 1.0, True)

        # Launch requests with slight staggered timing to simulate real load
        tasks = []
        for i in range(6):
            # Small staggered delay so requests start at slightly different times
            await asyncio.sleep(0.01)
            task = asyncio.create_task(simulate_request())
            tasks.append(task)

        # Wait for all requests to complete
        await asyncio.gather(*tasks)

        # Should have used all instances relatively evenly
        from collections import Counter

        usage_count = Counter(selected_instances)

        # Each instance should be used at least once
        assert len(usage_count) == 3
        # Distribution should be relatively even (each used 2 times)
        assert all(count == 2 for count in usage_count.values())

    @pytest.mark.asyncio
    async def test_concurrent_access_thread_safety(self, load_balancer):
        """Test that concurrent access is handled safely."""
        load_balancer.register_model_instance("gpt-oss-20b", "provider1", "http://localhost:1234")

        async def simulate_request():
            instance = await load_balancer.select_instance("gpt-oss-20b")
            if instance:
                await load_balancer.start_request(instance)
                await asyncio.sleep(0.01)  # Simulate processing time
                await load_balancer.end_request(instance, 0.01, True)

        # Run many concurrent requests
        tasks = [simulate_request() for _ in range(50)]
        await asyncio.gather(*tasks)

        # Verify final state is consistent
        instance = load_balancer.instances["gpt-oss-20b"][0]
        assert instance.active_requests == 0  # All requests should be completed
        assert instance.total_requests == 50


class TestDistributionStrategies:
    """Test the distribution strategy functionality."""

    @pytest.fixture
    def multi_host_load_balancer(self):
        """Create load balancer with instances on multiple hosts."""
        lb = ModelLoadBalancer()

        # Host A: Fast host (RTX 3090 equivalent)
        lb.register_model_instance("gpt-oss-20b", "provider1", "http://192.168.1.10:1234")
        lb.register_model_instance("gpt-oss-20b:2", "provider1", "http://192.168.1.10:1234")

        # Host B: Slower host (RTX 3060 equivalent)
        lb.register_model_instance("gpt-oss-20b:3", "provider2", "http://192.168.1.11:1234")
        lb.register_model_instance("gpt-oss-20b:4", "provider2", "http://192.168.1.11:1234")

        # Set performance metrics to simulate host differences
        # Host A (fast)
        lb.hosts["192.168.1.10:1234"].avg_ttft = 0.1
        lb.hosts["192.168.1.10:1234"].avg_response_time = 1.0
        lb.hosts["192.168.1.10:1234"].success_count = 100
        lb.hosts["192.168.1.10:1234"].total_requests = 100

        # Host B (slower)
        lb.hosts["192.168.1.11:1234"].avg_ttft = 0.3
        lb.hosts["192.168.1.11:1234"].avg_response_time = 2.0
        lb.hosts["192.168.1.11:1234"].success_count = 90
        lb.hosts["192.168.1.11:1234"].total_requests = 100

        return lb

    def test_spread_all_hosts_strategy(self, multi_host_load_balancer):
        """Test SPREAD_ALL_HOSTS includes instances from all hosts."""
        instances = multi_host_load_balancer.get_available_instances(
            "gpt-oss-20b", distribution_strategy=DistributionStrategy.SPREAD_ALL_HOSTS
        )

        # Should include all 4 instances from both hosts
        assert len(instances) == 4
        host_ids = {i.host_id for i in instances}
        assert "192.168.1.10:1234" in host_ids
        assert "192.168.1.11:1234" in host_ids

    def test_spread_best_host_strategy(self, multi_host_load_balancer):
        """Test SPREAD_BEST_HOST only uses instances from the fastest host."""
        instances = multi_host_load_balancer.get_available_instances(
            "gpt-oss-20b", distribution_strategy=DistributionStrategy.SPREAD_BEST_HOST
        )

        # Should only include instances from the fast host (192.168.1.10:1234)
        assert len(instances) == 2
        for instance in instances:
            assert instance.host_id == "192.168.1.10:1234"

    def test_spread_first_host_strategy(self, multi_host_load_balancer):
        """Test SPREAD_FIRST_HOST only uses instances from the first host."""
        instances = multi_host_load_balancer.get_available_instances(
            "gpt-oss-20b", distribution_strategy=DistributionStrategy.SPREAD_FIRST_HOST
        )

        # Should only include instances from the first host that was registered
        assert len(instances) == 2
        # All instances should be from the same host (the first one)
        first_host_id = instances[0].host_id
        for instance in instances:
            assert instance.host_id == first_host_id

    @pytest.mark.asyncio
    async def test_select_instance_with_distribution_strategy(self, multi_host_load_balancer):
        """Test select_instance respects distribution strategy."""
        # Test SPREAD_BEST_HOST
        selected = await multi_host_load_balancer.select_instance(
            "gpt-oss-20b", distribution_strategy=DistributionStrategy.SPREAD_BEST_HOST
        )

        assert selected is not None
        assert selected.host_id == "192.168.1.10:1234"  # Should select from fast host

    def test_environment_variable_configuration(self):
        """Test distribution strategy configuration from environment."""
        import os

        # Test valid environment variable
        with patch.dict(os.environ, {"LOAD_BALANCER_DISTRIBUTION_STRATEGY": "spread_best_host"}):
            lb = ModelLoadBalancer()
            assert lb.default_distribution_strategy == DistributionStrategy.SPREAD_BEST_HOST

        # Test invalid environment variable falls back to default
        with patch.dict(os.environ, {"LOAD_BALANCER_DISTRIBUTION_STRATEGY": "invalid_strategy"}):
            lb = ModelLoadBalancer()
            assert lb.default_distribution_strategy == DistributionStrategy.SPREAD_ALL_HOSTS

    def test_set_distribution_strategy(self, multi_host_load_balancer):
        """Test setting distribution strategy programmatically."""
        # Default should be SPREAD_ALL_HOSTS
        assert multi_host_load_balancer.default_distribution_strategy == DistributionStrategy.SPREAD_ALL_HOSTS

        # Change to SPREAD_BEST_HOST
        multi_host_load_balancer.set_distribution_strategy(DistributionStrategy.SPREAD_BEST_HOST)
        assert multi_host_load_balancer.default_distribution_strategy == DistributionStrategy.SPREAD_BEST_HOST

    @pytest.mark.asyncio
    async def test_select_instance_with_default_strategy(self, multi_host_load_balancer):
        """Test selecting instance with configured default strategy."""
        # Set default to SPREAD_BEST_HOST
        multi_host_load_balancer.set_distribution_strategy(DistributionStrategy.SPREAD_BEST_HOST)

        # Select using default strategy
        selected = await multi_host_load_balancer.select_instance_with_default_strategy("gpt-oss-20b")

        assert selected is not None
        assert selected.host_id == "192.168.1.10:1234"  # Should select from fast host

    def test_tba_strategy_fallback(self, multi_host_load_balancer):
        """Test that TBA strategy falls back to SPREAD_ALL_HOSTS."""
        instances = multi_host_load_balancer.get_available_instances(
            "gpt-oss-20b", distribution_strategy=DistributionStrategy.TBA_SPREAD_EQUAL_BEST_HOSTS
        )

        # Should fall back to SPREAD_ALL_HOSTS behavior
        assert len(instances) == 4
