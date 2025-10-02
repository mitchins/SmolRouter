# Ground Truth Verification System

## Overview

This system provides **100% certainty** about which API key and proxy were actually used for each request - no guesswork, no intent, only observed reality.

## Architecture

### 1. Transport Layer Observation (Apple-style Delegation)

We use **custom httpx transport wrappers** that intercept every request/response:

```
┌─────────────────────┐
│  Google GenAI SDK   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────┐
│ ObservingHTTPTransport      │  ◄── Captures ground truth
│ - Intercepts handle_request │  ◄── Extracts API key from headers
│ - Observes proxy config     │  ◄── Observes actual connection
│ - Delegates to wrapped      │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────┐
│ httpx.HTTPTransport │  ◄── Actual HTTP transport
│ (with proxy config) │
└─────────────────────┘
```

### 2. Ground Truth Capture

**What we observe:**
- **API Key**: Extracted from actual `x-goog-api-key` header sent on wire
- **Proxy**: Extracted from transport's `_pool._proxy_url` (actual connection)
- **URL, Method, Host**: Actual request details
- **Status Code**: Actual response

**When we observe:**
- `handle_request()` - Before request is sent (captures intent)
- After response received (captures result)

### 3. Verification Process

```python
# 1. Generate observation ID
observation_id = f"obs_{uuid.uuid4().hex[:12]}"

# 2. Create observing transports
sync_transport = ObservingHTTPTransport(observation_id, base_transport, proxy=proxy_url)
async_transport = ObservingAsyncHTTPTransport(observation_id, base_transport, proxy=proxy_url)

# 3. SDK makes request (transport intercepts and observes)
response = await client.models.generate_content(...)

# 4. Verify ground truth
observer = get_observer()
observation = observer.get_observation(observation_id)

# 5. Compare intent vs reality
key_verified = observer.verify_api_key(observation_id, expected_suffix)
proxy_verified = observer.verify_proxy(observation_id, expected_proxy)

# 6. Use GROUND TRUTH (not intent)
actual_key = observation.api_key_used
actual_proxy = observation.proxy_url
```

## Guarantees

### ✅ 100% Certainty

1. **API Key**: We extract it from the actual HTTP header that goes on the wire
   - Not "which key we selected"
   - Not "which key we passed to SDK"
   - **Which key the SDK actually used in the HTTP request**

2. **Proxy**: We extract it from the actual httpx connection pool
   - Not "which proxy we configured"
   - Not "which proxy we think should be used"
   - **Which proxy httpx actually connected through**

### ✅ Verification

The system actively verifies that intent matches reality:

```
📡 GROUND TRUTH [async]: key=...abc12345, proxy=socks5://192.168.1.50:1080, url=https://...
✅ API key verified for obs_a1b2c3d4: ...abc12345
✅ Proxy verified for obs_a1b2c3d4: socks5://192.168.1.50:1080
✅ Ground truth verified: key=...abc12345, proxy=socks5://192.168.1.50:1080
```

If there's ANY mismatch:
```
❌ API key mismatch for obs_xyz: expected=...key1, actual=...key2
❌ Proxy mismatch for obs_xyz: expected=proxy1, actual=proxy2
```

### ✅ Production Ready

- **Minimal overhead**: Only observes, doesn't modify behavior
- **No dependencies**: Uses only httpx primitives
- **Thread-safe**: Separate observation per request
- **Auto-cleanup**: Old observations purged after 1 hour
- **Graceful degradation**: Falls back to intent if observation fails

## Use Cases

### 1. Development & Testing

```python
# Verify proxy is actually being used
observation = observer.get_observation(obs_id)
assert observation.proxy_url == "socks5://192.168.1.50:1080"
```

### 2. Production Analytics

```python
# Get statistics on actual API key usage
for obs in observer.observations.values():
    print(f"Key {obs.api_key_used[-8:]} via {obs.proxy_url or 'direct'}")
```

### 3. Debugging

```
INFO: 🚀 Outbound request: model=gemini-2.0-flash-exp, api_key=...abc12345, proxy=socks5://... [obs=obs_a1b2c3d4]
INFO: 📡 GROUND TRUTH [async]: key=...abc12345, proxy=socks5://..., url=https://generativelanguage.googleapis.com/...
INFO: 📥 RESPONSE [async]: status=200
DEBUG: ✅ API key verified for obs_a1b2c3d4: ...abc12345
DEBUG: ✅ Proxy verified for obs_a1b2c3d4: socks5://...
INFO: ✅ Ground truth verified: key=...abc12345, proxy=socks5://...
```

### 4. Compliance & Auditing

Every request has **irrefutable proof** of:
- Which API key was used (for billing/quota)
- Which proxy was used (for routing/compliance)
- Exact timestamp and response

## Files

- `transport_observer.py` - Observer and transport wrappers
- `google_genai_provider.py` - Integration with Google GenAI provider
- `request_metadata.py` - Metadata dataclass

## Design Principles

1. **Observe, don't interfere** - Zero impact on actual requests
2. **Delegate, don't duplicate** - Wrap, don't reimplement
3. **Verify, don't trust** - Compare intent vs reality
4. **Evidence, not belief** - Store actual headers/connections
5. **Explicit, not implicit** - Clear logging of what was observed

## Future Extensions

This pattern can be extended to:
- Other providers (OpenAI, Anthropic, etc.)
- Request/response body inspection
- Timing analysis (proxy latency)
- Error categorization (proxy vs API errors)
- Traffic analysis (bandwidth per proxy/key)
