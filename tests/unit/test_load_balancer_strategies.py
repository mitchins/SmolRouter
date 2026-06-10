"""Unit tests for ModelLoadBalancer distribution strategies and metric updates.

These exercise the pure host-selection / sorting / metric-accounting helpers
directly (no async, no network) which were largely uncovered: distribution
strategy filtering, best-host scoring, round-robin and fastest-host ordering,
instance/host/global metric updates, and host health marking.
"""

import pytest

from smolrouter.load_balancer import (
    DistributionStrategy,
    HostMetrics,
    ModelInstance,
    ModelLoadBalancer,
)


def _instance(host_id, *, model_id=None, base_name="llama", instance_num=0, active=0, total=0):
    return ModelInstance(
        model_id=model_id or f"{base_name}@{host_id}",
        base_name=base_name,
        instance_num=instance_num,
        provider_id=f"prov-{host_id}",
        provider_url=f"http://{host_id}",
        host_id=host_id,
        active_requests=active,
        total_requests=total,
    )


def _register_host(lb, host_id, *, ttft=0.0, rt=0.0, active=0, total=0, success=0, healthy=True):
    lb.hosts[host_id] = HostMetrics(
        host_id=host_id,
        url=f"http://{host_id}",
        avg_ttft=ttft,
        avg_response_time=rt,
        active_requests=active,
        total_requests=total,
        success_count=success,
        is_healthy=healthy,
    )


def _make_lb():
    return ModelLoadBalancer()


# ==========================================================================
# _apply_distribution_strategy
# ==========================================================================


def test_spread_all_hosts_returns_everything():
    lb = _make_lb()
    instances = [_instance("a"), _instance("b")]
    out = lb._apply_distribution_strategy(instances, DistributionStrategy.SPREAD_ALL_HOSTS)
    assert out == instances


def test_spread_first_host_filters_to_first_host():
    lb = _make_lb()
    instances = [_instance("a"), _instance("b"), _instance("a", model_id="a2")]
    out = lb._apply_distribution_strategy(instances, DistributionStrategy.SPREAD_FIRST_HOST)
    assert {i.host_id for i in out} == {"a"}
    assert len(out) == 2


def test_spread_first_host_empty_returns_empty():
    lb = _make_lb()
    assert lb._apply_distribution_strategy([], DistributionStrategy.SPREAD_FIRST_HOST) == []


def test_spread_best_host_filters_to_best():
    lb = _make_lb()
    _register_host(lb, "fast", ttft=0.1, rt=0.1)
    _register_host(lb, "slow", ttft=2.0, rt=2.0)
    instances = [_instance("slow"), _instance("fast"), _instance("fast", model_id="f2")]
    out = lb._apply_distribution_strategy(instances, DistributionStrategy.SPREAD_BEST_HOST)
    assert {i.host_id for i in out} == {"fast"}


def test_spread_best_host_empty_returns_empty():
    lb = _make_lb()
    assert lb._apply_distribution_strategy([], DistributionStrategy.SPREAD_BEST_HOST) == []


def test_spread_best_host_no_known_hosts_returns_all():
    lb = _make_lb()
    instances = [_instance("a"), _instance("b")]
    # No hosts registered -> _find_best_host returns None -> all returned
    out = lb._apply_distribution_strategy(instances, DistributionStrategy.SPREAD_BEST_HOST)
    assert out == instances


def test_tba_strategy_falls_back_to_all():
    lb = _make_lb()
    instances = [_instance("a"), _instance("b")]
    out = lb._apply_distribution_strategy(instances, DistributionStrategy.TBA_SPREAD_EQUAL_BEST_HOSTS)
    assert out == instances


# ==========================================================================
# _find_best_host
# ==========================================================================


def test_find_best_host_empty_returns_none():
    assert _make_lb()._find_best_host([]) is None


def test_find_best_host_picks_lowest_score():
    lb = _make_lb()
    _register_host(lb, "fast", ttft=0.1, rt=0.1, active=0)
    _register_host(lb, "slow", ttft=1.0, rt=1.0, active=3)
    assert lb._find_best_host(["fast", "slow"]) == "fast"


def test_find_best_host_skips_unhealthy_and_unknown():
    lb = _make_lb()
    _register_host(lb, "down", ttft=0.0, rt=0.0, healthy=False)
    # "ghost" not registered at all
    assert lb._find_best_host(["down", "ghost"]) is None


def test_find_best_host_success_rate_bonus():
    lb = _make_lb()
    # Same raw latency, but reliable host has better success rate -> lower score
    _register_host(lb, "reliable", ttft=0.5, rt=0.5, total=100, success=100)
    _register_host(lb, "flaky", ttft=0.5, rt=0.5, total=100, success=10)
    assert lb._find_best_host(["reliable", "flaky"]) == "reliable"


# ==========================================================================
# _sort_by_host_round_robin
# ==========================================================================


def test_round_robin_interleaves_hosts_and_sorts_within_host():
    lb = _make_lb()
    # host "a": loads 5 and 0; host "b": load 1. Totals: a=5, b=1 -> b first.
    a_busy = _instance("a", model_id="a_busy", active=5)
    a_idle = _instance("a", model_id="a_idle", active=0)
    b_one = _instance("b", model_id="b_one", active=1)
    result = lb._sort_by_host_round_robin([a_busy, a_idle, b_one])
    ids = [i.model_id for i in result]
    # b (lower total load) leads; within host a, idle before busy; then a's leftover
    assert ids == ["b_one", "a_idle", "a_busy"]


def test_round_robin_single_host():
    lb = _make_lb()
    insts = [_instance("a", model_id="x", active=2), _instance("a", model_id="y", active=1)]
    result = lb._sort_by_host_round_robin(insts)
    assert [i.model_id for i in result] == ["y", "x"]  # sorted by load within host


# ==========================================================================
# _sort_by_fastest_host
# ==========================================================================


def test_fastest_host_orders_by_speed_score():
    lb = _make_lb()
    _register_host(lb, "fast", ttft=0.1, rt=0.1)
    _register_host(lb, "slow", ttft=1.0, rt=1.0)
    insts = [_instance("slow", model_id="s"), _instance("fast", model_id="f")]
    result = lb._sort_by_fastest_host(insts)
    assert [i.model_id for i in result] == ["f", "s"]


def test_fastest_host_unknown_host_sinks_to_bottom():
    lb = _make_lb()
    _register_host(lb, "known", ttft=0.1, rt=0.1)
    insts = [_instance("ghost", model_id="g"), _instance("known", model_id="k")]
    result = lb._sort_by_fastest_host(insts)
    assert result[-1].model_id == "g"  # inf score -> last


# ==========================================================================
# _update_instance_metrics
# ==========================================================================


def test_update_instance_metrics_no_requests_is_noop():
    lb = _make_lb()
    inst = _instance("a", total=0)
    lb._update_instance_metrics(inst, response_time=1.0, ttft=0.5)
    assert inst.avg_response_time == 0.0
    assert inst.avg_ttft == 0.0


def test_update_instance_metrics_running_average():
    lb = _make_lb()
    # total>1 with existing averages so the prior-average term is actually exercised:
    # new_avg = (prior * (total-1) + sample) / total
    inst = _instance("a", total=2)
    inst.avg_response_time = 1.0
    inst.avg_ttft = 0.2
    lb._update_instance_metrics(inst, response_time=3.0, ttft=0.4)
    assert inst.avg_response_time == 2.0  # (1.0*1 + 3.0)/2
    assert inst.avg_ttft == pytest.approx(0.3)  # (0.2*1 + 0.4)/2


def test_update_instance_metrics_skips_ttft_when_zero():
    lb = _make_lb()
    inst = _instance("a", total=1)
    inst.avg_ttft = 0.9
    lb._update_instance_metrics(inst, response_time=1.0, ttft=0.0)
    assert inst.avg_ttft == 0.9  # unchanged


# ==========================================================================
# _update_host_metrics
# ==========================================================================


def test_update_host_metrics_unknown_host_is_noop():
    lb = _make_lb()
    inst = _instance("ghost")
    lb._update_host_metrics(inst, response_time=1.0, success=True, ttft=0.5)  # no raise


def test_update_host_metrics_success_accounting():
    lb = _make_lb()
    # Pre-seed prior averages/total so the running-average math is genuinely tested.
    _register_host(lb, "a", active=2, total=1, rt=1.0, ttft=0.2)
    inst = _instance("a")
    lb._update_host_metrics(inst, response_time=3.0, success=True, ttft=0.4)
    host = lb.hosts["a"]
    assert host.active_requests == 1  # decremented
    assert host.total_requests == 2
    assert host.success_count == 1
    assert host.avg_response_time == 2.0  # (1.0*1 + 3.0)/2
    assert host.avg_ttft == pytest.approx(0.3)  # (0.2*1 + 0.4)/2


def test_update_host_metrics_marks_unhealthy_on_high_failure_rate():
    lb = _make_lb()
    _register_host(lb, "a", total=11, success=0)
    lb.hosts["a"].failure_count = 11
    inst = _instance("a")
    lb._update_host_metrics(inst, response_time=1.0, success=False, ttft=0.0)
    assert lb.hosts["a"].is_healthy is False


# ==========================================================================
# _update_global_stats
# ==========================================================================


def test_update_global_stats_success():
    lb = _make_lb()
    inst = _instance("a")
    lb._update_global_stats(inst, success=True)
    assert lb.stats.total_requests == 1
    assert lb.stats.successful_requests == 1
    assert inst.success_count == 1


def test_update_global_stats_failure():
    lb = _make_lb()
    inst = _instance("a")
    lb._update_global_stats(inst, success=False)
    assert lb.stats.failed_requests == 1
    assert inst.failure_count == 1


def test_update_global_stats_marks_instance_unhealthy():
    lb = _make_lb()
    inst = _instance("a", total=11)
    inst.failure_count = 6  # 6/11 > 0.5 after this failure
    lb._update_global_stats(inst, success=False)
    assert inst.is_healthy is False


# ==========================================================================
# host health marking
# ==========================================================================


def test_mark_host_unhealthy_propagates_to_instances():
    lb = _make_lb()
    _register_host(lb, "a")
    lb.instances["llama"] = [_instance("a", model_id="i1"), _instance("b", model_id="i2")]
    lb.mark_host_unhealthy("a")
    assert lb.hosts["a"].is_healthy is False
    assert lb.instances["llama"][0].is_healthy is False  # on host a
    assert lb.instances["llama"][1].is_healthy is True  # on host b, untouched


def test_mark_host_unhealthy_unknown_host_is_noop():
    lb = _make_lb()
    lb.mark_host_unhealthy("nope")  # no raise, nothing registered


def test_mark_host_healthy():
    lb = _make_lb()
    _register_host(lb, "a", healthy=False)
    lb.mark_host_healthy("a")
    assert lb.hosts["a"].is_healthy is True
