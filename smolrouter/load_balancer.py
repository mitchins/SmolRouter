"""
Load balancer for multiple instances of the same model.

Supports LM Studio's model instance naming patterns:
- gpt-oss-20b (base model)
- gpt-oss-20b:2, gpt-oss-20b:3 (additional instances)
- gpt-oss-20b-2, gpt-oss-20b-3 (alternative format)

Uses round-robin-by-least-busy strategy for optimal performance.
"""

import re
import time
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from urllib.parse import urlparse
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class DistributionStrategy(Enum):
    """Distribution strategies for load balancing across hosts."""

    SPREAD_ALL_HOSTS = "spread_all_hosts"  # Distribute load across all available hosts
    SPREAD_BEST_HOST = "spread_best_host"  # Only use instances on the fastest/best performing host
    SPREAD_FIRST_HOST = "spread_first_host"  # Use instances only on the first host in provider order
    TBA_SPREAD_EQUAL_BEST_HOSTS = "tba_spread_equal_best_hosts"  # Reserved for future implementation


@dataclass
class HostMetrics:
    """Performance metrics for a specific host."""

    host_id: str
    url: str
    active_requests: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    avg_ttft: float = 0.0  # Time to first token
    last_seen: float = 0.0
    is_healthy: bool = True
    failure_count: int = 0
    success_count: int = 0
    network_latency: float = 0.0


@dataclass
class ModelInstance:
    """Represents a single model instance."""

    model_id: str
    base_name: str
    instance_num: int
    provider_id: str
    provider_url: str
    host_id: str  # Added for host grouping
    active_requests: int = 0
    last_used: float = 0.0
    is_healthy: bool = True
    total_requests: int = 0
    avg_response_time: float = 0.0
    avg_ttft: float = 0.0  # Time to first token


@dataclass
class LoadBalancerStats:
    """Statistics for load balancer performance."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    instance_stats: Dict[str, Dict] = field(default_factory=dict)


class ModelLoadBalancer:
    """Load balancer for multiple instances of the same model across multiple hosts."""

    def __init__(self):
        self.instances: Dict[str, List[ModelInstance]] = defaultdict(list)  # base_name -> instances
        self.hosts: Dict[str, HostMetrics] = {}  # host_id -> metrics
        self.active_requests: Dict[str, int] = defaultdict(int)  # instance_id -> count
        self.stats = LoadBalancerStats()
        self._lock = asyncio.Lock()
        self.default_distribution_strategy = self._get_distribution_strategy_from_env()
        self._round_robin_counter: Dict[str, int] = defaultdict(int)  # base_name -> counter

    def _get_host_id(self, provider_url: str) -> str:
        """Extract host ID from provider URL.

        Args:
            provider_url: Provider URL like http://192.168.1.14:1234

        Returns:
            Host identifier like 192.168.1.14:1234
        """
        parsed = urlparse(provider_url)
        return f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname or provider_url

    def parse_model_name(self, model_id: str) -> Tuple[str, int]:
        """Parse model name to extract base name and instance number.

        Supports formats:
        - gpt-oss-20b -> (gpt-oss-20b, 0)
        - gpt-oss-20b:2 -> (gpt-oss-20b, 2)
        - gpt-oss-20b-2 -> (gpt-oss-20b, 2)

        Args:
            model_id: Full model identifier

        Returns:
            Tuple of (base_name, instance_number)
        """
        # Pattern 1: model:number format (LM Studio standard)
        colon_match = re.match(r"^(.+):(\d+)$", model_id)
        if colon_match:
            return colon_match.group(1), int(colon_match.group(2))

        # Pattern 2: model-number format (alternative)
        dash_match = re.match(r"^(.+)-(\d+)$", model_id)
        if dash_match:
            base = dash_match.group(1)
            num = int(dash_match.group(2))
            # Only treat as instance if it's not part of the model name
            # (e.g., gpt-4 should not become gpt with instance 4)
            if not base.endswith(("gpt", "llama", "mistral", "qwen")):
                return base, num

        # No instance number found - treat as base model (instance 0)
        return model_id, 0

    def register_model_instance(self, model_id: str, provider_id: str, provider_url: str) -> None:
        """Register a new model instance for load balancing.

        Args:
            model_id: Full model identifier (e.g., gpt-oss-20b:2)
            provider_id: Provider identifier
            provider_url: Provider URL
        """
        base_name, instance_num = self.parse_model_name(model_id)
        host_id = self._get_host_id(provider_url)

        # Register host if not exists
        if host_id not in self.hosts:
            self.hosts[host_id] = HostMetrics(host_id=host_id, url=provider_url, last_seen=time.time())
            logger.info(f"Registered new host: {host_id}")

        instance = ModelInstance(
            model_id=model_id,
            base_name=base_name,
            instance_num=instance_num,
            provider_id=provider_id,
            provider_url=provider_url,
            host_id=host_id,
        )

        # Check if already registered
        existing = [i for i in self.instances[base_name] if i.model_id == model_id and i.provider_url == provider_url]
        if not existing:
            self.instances[base_name].append(instance)
            logger.info(
                f"Registered model instance: {model_id} on host {host_id} (base: {base_name}, instance: {instance_num})"
            )

    def get_available_instances(
        self,
        requested_model: str,
        strategy: str = "smart",
        distribution_strategy: DistributionStrategy = DistributionStrategy.SPREAD_ALL_HOSTS,
    ) -> List[ModelInstance]:
        """Get all available instances for a requested model.

        Args:
            requested_model: Model name requested by client
            strategy: Selection strategy - "smart", "host_round_robin", "fastest_host"
            distribution_strategy: Distribution strategy for host selection

        Returns:
            List of available model instances, sorted by preference
        """
        base_name, _ = self.parse_model_name(requested_model)
        instances = self.instances.get(base_name, [])

        # Filter to healthy instances with healthy hosts
        healthy_instances = [
            i for i in instances if i.is_healthy and self.hosts.get(i.host_id, HostMetrics("", "")).is_healthy
        ]

        if not healthy_instances:
            return []

        # Apply distribution strategy
        healthy_instances = self._apply_distribution_strategy(healthy_instances, distribution_strategy)

        if strategy == "smart":
            return self._sort_by_smart_strategy(healthy_instances)
        elif strategy == "host_round_robin":
            return self._sort_by_host_round_robin(healthy_instances)
        elif strategy == "fastest_host":
            return self._sort_by_fastest_host(healthy_instances)
        else:
            # Default: sort by load then by last used time
            healthy_instances.sort(key=lambda x: (x.active_requests, x.last_used))
            return healthy_instances

    def _apply_distribution_strategy(
        self, instances: List[ModelInstance], distribution_strategy: DistributionStrategy
    ) -> List[ModelInstance]:
        """Apply distribution strategy to filter instances by host selection.

        Args:
            instances: List of healthy instances
            distribution_strategy: Strategy for host selection

        Returns:
            Filtered list of instances based on distribution strategy
        """
        if distribution_strategy == DistributionStrategy.SPREAD_ALL_HOSTS:
            # Default behavior - use all healthy instances
            return instances

        elif distribution_strategy == DistributionStrategy.SPREAD_BEST_HOST:
            # Only use instances on the fastest/best performing host
            if not instances:
                return instances

            # Find the best host based on performance metrics
            best_host_id = self._find_best_host([i.host_id for i in instances])
            if best_host_id:
                return [i for i in instances if i.host_id == best_host_id]
            return instances

        elif distribution_strategy == DistributionStrategy.SPREAD_FIRST_HOST:
            # Use instances only on the first host in provider order
            if not instances:
                return instances

            # Group by host and use the first host that appears in the instances list
            first_host_id = instances[0].host_id
            return [i for i in instances if i.host_id == first_host_id]

        elif distribution_strategy == DistributionStrategy.TBA_SPREAD_EQUAL_BEST_HOSTS:
            # Reserved for future implementation - currently falls back to SPREAD_ALL_HOSTS
            logger.warning("TBA_SPREAD_EQUAL_BEST_HOSTS not yet implemented, falling back to SPREAD_ALL_HOSTS")
            return instances

        else:
            # Unknown strategy - log warning and fall back to all hosts
            logger.warning(f"Unknown distribution strategy: {distribution_strategy}, falling back to SPREAD_ALL_HOSTS")
            return instances

    def _find_best_host(self, host_ids: List[str]) -> Optional[str]:
        """Find the best performing host from a list of host IDs.

        Args:
            host_ids: List of host IDs to evaluate

        Returns:
            Host ID of the best performing host, or None if none found
        """
        if not host_ids:
            return None

        best_host_id = None
        best_score = float("inf")

        for host_id in set(host_ids):  # Remove duplicates
            host = self.hosts.get(host_id)
            if not host or not host.is_healthy:
                continue

            # Calculate performance score (lower is better)
            # Factor in TTFT, response time, and current load
            score = (
                host.avg_ttft * 2  # Weight TTFT heavily for performance
                + host.avg_response_time
                + host.active_requests * 5  # Penalize current load
            )

            # Bonus for high success rate
            if host.total_requests > 0:
                success_rate = host.success_count / host.total_requests
                score *= 2 - success_rate  # Multiply by 1.0 to 2.0 based on success rate

            if score < best_score:
                best_score = score
                best_host_id = host_id

        return best_host_id

    def _sort_by_smart_strategy(self, instances: List[ModelInstance]) -> List[ModelInstance]:
        """Smart strategy: Prioritize least-loaded instances with tie-breaking.

        The PRIMARY factor is current load (active_requests). When loads are equal
        or very close, secondary factors like host performance and round-robin
        are used for tie-breaking.

        This ensures that an instance with 8 active requests will NEVER be selected
        over one with 5 active requests, regardless of other factors.
        """
        if not instances:
            return instances

        base_name = instances[0].base_name

        # STEP 1: Sort purely by active_requests first
        instances_by_load = sorted(instances, key=lambda x: x.active_requests)

        # STEP 2: Find the minimum load
        min_load = instances_by_load[0].active_requests

        # STEP 3: Find all instances with EXACTLY the minimum load
        # This is the ONLY set we consider for selection
        # NO tolerance - we always want the truly least-loaded instance(s)
        candidates = [i for i in instances_by_load if i.active_requests == min_load]

        logger.debug(
            f"Smart strategy for {base_name}: min_load={min_load}, "
            f"candidates={[(i.model_id, i.active_requests) for i in candidates]}, "
            f"all_instances={[(i.model_id, i.active_requests) for i in instances_by_load]}"
        )

        # STEP 4: If only one candidate, use it
        if len(candidates) == 1:
            result = [candidates[0]]
            result.extend([i for i in instances_by_load if i not in result])
            return result

        # STEP 5: Multiple candidates with same/similar load - use secondary scoring for tie-break
        def tiebreak_score(instance: ModelInstance) -> float:
            """Secondary score for tie-breaking among equally-loaded instances."""
            host = self.hosts.get(instance.host_id)

            # Factor 1: Host performance (lower TTFT/response time is better)
            if host:
                host_speed = host.avg_ttft + host.avg_response_time
            else:
                host_speed = 0

            # Factor 2: Slight preference for less recently used
            # But this is ONLY a tie-breaker, not a primary factor
            if instance.last_used == 0.0:
                recency = 0
            else:
                time_since_use = time.time() - instance.last_used
                # Small factor: 0 to 1 based on seconds since last use
                recency = max(0, 1 - (time_since_use / 10))  # Decays over 10 seconds

            return host_speed + recency

        # Score and sort candidates
        scored_candidates = [(tiebreak_score(c), c) for c in candidates]
        scored_candidates.sort(key=lambda x: x[0])

        # Check if scores are very similar - use round-robin for fairness
        min_tiebreak = scored_candidates[0][0]
        similar_threshold = max(0.5, min_tiebreak * 0.2)  # More generous for tie-breaking

        similar_candidates = [c for score, c in scored_candidates if score <= min_tiebreak + similar_threshold]

        if len(similar_candidates) > 1:
            # Round-robin among similar candidates
            counter = self._round_robin_counter[base_name]
            selected_index = counter % len(similar_candidates)
            selected = similar_candidates[selected_index]
            self._round_robin_counter[base_name] = (counter + 1) % len(similar_candidates)

            logger.debug(
                f"Round-robin tie-break for {base_name}: selected={selected.model_id}, "
                f"from candidates={[c.model_id for c in similar_candidates]}"
            )

            result = [selected]
            result.extend([i for i in instances_by_load if i != selected])
            return result
        else:
            # Use the best scoring candidate
            result = [scored_candidates[0][1]]
            result.extend([i for i in instances_by_load if i not in result])
            return result

    def _sort_by_host_round_robin(self, instances: List[ModelInstance]) -> List[ModelInstance]:
        """Distribute across hosts in round-robin fashion."""
        # Group by host
        by_host = defaultdict(list)
        for instance in instances:
            by_host[instance.host_id].append(instance)

        # Sort each host's instances by load
        for host_instances in by_host.values():
            host_instances.sort(key=lambda x: x.active_requests)

        # Sort hosts by total load
        sorted_hosts = sorted(by_host.keys(), key=lambda h: sum(i.active_requests for i in by_host[h]))

        # Round-robin through hosts
        result = []
        host_indices = {h: 0 for h in sorted_hosts}

        while any(host_indices[h] < len(by_host[h]) for h in sorted_hosts):
            for host_id in sorted_hosts:
                if host_indices[host_id] < len(by_host[host_id]):
                    result.append(by_host[host_id][host_indices[host_id]])
                    host_indices[host_id] += 1

        return result

    def _sort_by_fastest_host(self, instances: List[ModelInstance]) -> List[ModelInstance]:
        """Sort by fastest hosts first (based on TTFT and response time)."""

        def host_speed_score(instance: ModelInstance) -> float:
            host = self.hosts.get(instance.host_id)
            if not host:
                return float("inf")

            # Combine TTFT and response time (lower is better)
            return host.avg_ttft + host.avg_response_time + (instance.active_requests * 2)

        instances.sort(key=host_speed_score)
        return instances

    async def select_instance(
        self, requested_model: str, distribution_strategy: DistributionStrategy = DistributionStrategy.SPREAD_ALL_HOSTS
    ) -> Optional[ModelInstance]:
        """Select the best instance for a request using least-busy strategy.

        This method atomically selects an instance AND marks the request as started
        to prevent race conditions where multiple concurrent requests all see the
        same load and get routed to the same instance.

        Args:
            requested_model: Model name requested by client
            distribution_strategy: Distribution strategy for host selection

        Returns:
            Selected model instance or None if none available
        """
        async with self._lock:
            instances = self.get_available_instances(requested_model, distribution_strategy=distribution_strategy)

            if not instances:
                base_name, _ = self.parse_model_name(requested_model)
                logger.warning(f"No healthy instances available for model: {base_name}")
                return None

            # Select least busy instance
            selected = instances[0]
            selected.last_used = time.time()

            # CRITICAL: Atomically increment active_requests while holding the lock
            # This prevents race conditions where concurrent requests all see the same
            # load (0) and all get routed to the same instance
            selected.active_requests += 1
            selected.total_requests += 1
            self.active_requests[selected.model_id] = selected.active_requests

            # Also update host metrics
            host = self.hosts.get(selected.host_id)
            if host:
                host.active_requests += 1

            logger.debug(
                f"Selected instance {selected.model_id} for {requested_model} "
                f"(active_requests: {selected.active_requests})"
            )

            return selected

    async def start_request(self, instance: ModelInstance) -> None:
        """Mark the start of a request on an instance.

        NOTE: This method is now a no-op. Request tracking is handled atomically
        inside select_instance() to prevent race conditions. This method is kept
        for backward compatibility but does nothing.

        Args:
            instance: Model instance handling the request
        """
        # No-op: Request tracking is now done atomically in select_instance()
        # to prevent race conditions where multiple concurrent requests all
        # see the same load and get routed to the same instance.
        pass

    async def end_request(
        self, instance: ModelInstance, response_time: float, success: bool = True, ttft: float = 0.0
    ) -> None:
        """Mark the end of a request on an instance.

        Args:
            instance: Model instance that handled the request
            response_time: Time taken to process the request in seconds
            success: Whether the request was successful
            ttft: Time to first token in seconds (for performance tracking)
        """
        async with self._lock:
            instance.active_requests = max(0, instance.active_requests - 1)
            self.active_requests[instance.model_id] = instance.active_requests

            # Update instance metrics
            if instance.total_requests > 0:
                instance.avg_response_time = (
                    instance.avg_response_time * (instance.total_requests - 1) + response_time
                ) / instance.total_requests
                if ttft > 0:
                    instance.avg_ttft = (
                        instance.avg_ttft * (instance.total_requests - 1) + ttft
                    ) / instance.total_requests

            # Update host metrics
            host = self.hosts.get(instance.host_id)
            if host:
                host.active_requests = max(0, host.active_requests - 1)
                host.total_requests += 1
                host.last_seen = time.time()

                if success:
                    host.success_count += 1
                else:
                    host.failure_count += 1

                # Update host averages
                if host.total_requests > 0:
                    host.avg_response_time = (
                        host.avg_response_time * (host.total_requests - 1) + response_time
                    ) / host.total_requests
                    if ttft > 0:
                        host.avg_ttft = (host.avg_ttft * (host.total_requests - 1) + ttft) / host.total_requests

                # Health check for host
                if host.total_requests > 10:
                    failure_rate = host.failure_count / host.total_requests
                    if failure_rate > 0.5 and host.is_healthy:
                        host.is_healthy = False
                        logger.warning(f"Marking host {host.host_id} as unhealthy (failure rate: {failure_rate:.2%})")
                    elif failure_rate < 0.1 and not host.is_healthy:
                        host.is_healthy = True
                        logger.info(f"Marking host {host.host_id} as healthy (failure rate: {failure_rate:.2%})")

            # Update global stats
            self.stats.total_requests += 1
            if success:
                self.stats.successful_requests += 1
            else:
                self.stats.failed_requests += 1
                # Mark instance as unhealthy if too many failures
                if instance.total_requests > 10 and (self.stats.failed_requests / self.stats.total_requests) > 0.5:
                    instance.is_healthy = False
                    logger.warning(f"Marking instance {instance.model_id} as unhealthy due to high failure rate")

    def mark_instance_unhealthy(self, instance: ModelInstance) -> None:
        """Mark an instance as unhealthy.

        Args:
            instance: Model instance to mark as unhealthy
        """
        instance.is_healthy = False
        logger.warning(f"Marked instance {instance.model_id} as unhealthy")

    def mark_instance_healthy(self, instance: ModelInstance) -> None:
        """Mark an instance as healthy.

        Args:
            instance: Model instance to mark as healthy
        """
        instance.is_healthy = True
        logger.info(f"Marked instance {instance.model_id} as healthy")

    def mark_host_unhealthy(self, host_id: str) -> None:
        """Mark a host as unhealthy and all its instances.

        Args:
            host_id: Host identifier to mark as unhealthy
        """
        if host_id in self.hosts:
            self.hosts[host_id].is_healthy = False
            logger.warning(f"Marked host {host_id} as unhealthy")

            # Mark all instances on this host as unhealthy
            for instances in self.instances.values():
                for instance in instances:
                    if instance.host_id == host_id:
                        instance.is_healthy = False

    def mark_host_healthy(self, host_id: str) -> None:
        """Mark a host as healthy and potentially its instances.

        Args:
            host_id: Host identifier to mark as healthy
        """
        if host_id in self.hosts:
            self.hosts[host_id].is_healthy = True
            logger.info(f"Marked host {host_id} as healthy")

    def get_host_stats(self) -> Dict[str, Dict]:
        """Get statistics for all hosts.

        Returns:
            Dictionary mapping host_id to host statistics
        """
        host_stats = {}
        for host_id, host in self.hosts.items():
            host_stats[host_id] = {
                "url": host.url,
                "active_requests": host.active_requests,
                "total_requests": host.total_requests,
                "avg_response_time": host.avg_response_time,
                "avg_ttft": host.avg_ttft,
                "success_rate": (host.success_count / host.total_requests if host.total_requests > 0 else 0),
                "is_healthy": host.is_healthy,
                "last_seen": host.last_seen,
                "models_count": sum(
                    1 for instances in self.instances.values() for instance in instances if instance.host_id == host_id
                ),
            }
        return host_stats

    def get_stats(self) -> Dict:
        """Get comprehensive load balancer statistics.

        Returns:
            Dictionary containing statistics
        """
        stats = {
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
            "success_rate": (
                self.stats.successful_requests / self.stats.total_requests if self.stats.total_requests > 0 else 0
            ),
            "hosts": self.get_host_stats(),
            "instances": {},
        }

        for base_name, instances in self.instances.items():
            stats["instances"][base_name] = []
            for instance in instances:
                stats["instances"][base_name].append(
                    {
                        "model_id": instance.model_id,
                        "instance_num": instance.instance_num,
                        "provider_id": instance.provider_id,
                        "host_id": instance.host_id,
                        "active_requests": instance.active_requests,
                        "total_requests": instance.total_requests,
                        "avg_response_time": instance.avg_response_time,
                        "avg_ttft": instance.avg_ttft,
                        "is_healthy": instance.is_healthy,
                        "last_used": instance.last_used,
                    }
                )

        return stats

    def get_model_groups(self) -> Dict[str, List[str]]:
        """Get grouped models by base name.

        Returns:
            Dictionary mapping base names to list of instance IDs
        """
        groups = {}
        for base_name, instances in self.instances.items():
            groups[base_name] = [i.model_id for i in instances]
        return groups

    def _get_distribution_strategy_from_env(self) -> DistributionStrategy:
        """Get distribution strategy from environment variable.

        Returns:
            Distribution strategy from environment or default SPREAD_ALL_HOSTS
        """
        import os

        strategy_name = os.getenv("LOAD_BALANCER_DISTRIBUTION_STRATEGY", "spread_all_hosts").lower()

        try:
            return DistributionStrategy(strategy_name)
        except ValueError:
            logger.warning(f"Invalid distribution strategy '{strategy_name}', falling back to SPREAD_ALL_HOSTS")
            return DistributionStrategy.SPREAD_ALL_HOSTS

    def set_distribution_strategy(self, strategy: DistributionStrategy):
        """Set the default distribution strategy.

        Args:
            strategy: Distribution strategy to use as default
        """
        self.default_distribution_strategy = strategy
        logger.info(f"Set default distribution strategy to: {strategy.value}")

    async def select_instance_with_default_strategy(self, requested_model: str) -> Optional[ModelInstance]:
        """Select instance using the configured default distribution strategy.

        Args:
            requested_model: Model name requested by client

        Returns:
            Selected model instance or None if none available
        """
        return await self.select_instance(requested_model, self.default_distribution_strategy)


# Global load balancer instance
model_load_balancer = ModelLoadBalancer()
