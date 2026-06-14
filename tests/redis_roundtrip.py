"""
Redis round-trip counting harness for performance regression tests.

The dashboard slowdown that motivated this harness was a classic N+1: a code
path issued one awaited Redis command per record (``hgetall`` in a loop) instead
of batching them into a single pipeline. Wall-clock latency only reveals this
against a *networked* Redis, so it is invisible with in-memory FakeRedis. What
*is* invariant regardless of backend is the number of **round-trips** the code
makes to Redis - so we count those and assert on them.

Round-trip accounting:
- Each awaited standalone command on the client (hgetall, zrevrange, smembers,
  hset, ...) counts as one round-trip.
- A pipeline counts as exactly one round-trip, recorded when ``execute()`` is
  awaited, regardless of how many commands were buffered into it.

Usage::

    from tests.redis_roundtrip import count_round_trips

    with count_round_trips() as counter:
        await RequestLog.get_recent(1000)
    assert counter.round_trips <= 3  # not 1001
"""

from __future__ import annotations

import collections
import inspect
from contextlib import contextmanager
from unittest.mock import patch

import smolrouter.redis_backend as redis_backend


class _CountingPipeline:
    """Wraps a Redis pipeline, counting a single round-trip per execute()."""

    def __init__(self, inner, counter: "RoundTripCounter"):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_counter", counter)

    async def __aenter__(self):
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc):
        return await self._inner.__aexit__(*exc)

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if name == "execute":

            async def execute(*args, **kwargs):
                self._counter.round_trips += 1
                self._counter.calls["pipeline.execute"] += 1
                return await attr(*args, **kwargs)

            return execute
        # Buffered commands (pipe.hgetall, ...) return the inner pipeline; that
        # is fine because we only care about the single execute() round-trip.
        return attr


class RoundTripCounter:
    """Proxy around a Redis client that tallies round-trips by command."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "round_trips", 0)
        object.__setattr__(self, "calls", collections.Counter())

    def __getattr__(self, name):
        inner_attr = getattr(self._inner, name)

        if name == "pipeline":
            def make_pipeline(*args, **kwargs):
                return _CountingPipeline(inner_attr(*args, **kwargs), self)

            return make_pipeline

        if not callable(inner_attr):
            return inner_attr

        # redis-py / fakeredis command methods are not reliably detectable via
        # iscoroutinefunction, so decide at call-time: an awaited command is a
        # round-trip; a plain value (e.g. a property accessor) is not.
        def wrapper(*args, **kwargs):
            result = inner_attr(*args, **kwargs)
            if inspect.isawaitable(result):
                self.round_trips += 1
                self.calls[name] += 1
            return result

        return wrapper

    def reset(self):
        self.round_trips = 0
        self.calls = collections.Counter()


@contextmanager
def count_round_trips():
    """Patch the Redis accessor so backend code sees a counting proxy.

    Yields the :class:`RoundTripCounter`. The proxy wraps the real (Fake)Redis
    client returned by ``get_redis`` so behaviour is unchanged - only observed.
    """
    counter = RoundTripCounter(redis_backend.get_redis())
    with patch.object(redis_backend, "get_redis", return_value=counter):
        yield counter
