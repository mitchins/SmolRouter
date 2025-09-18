# SmolRouter navigation guide

## Quick access to the Upstreams view

1. Open the **Dashboard** (`/`).
2. Use either navigation option:
   - Top navigation bar → **Upstreams**
   - "Recent Requests" action buttons → **Upstreams** (purple button)
3. The Upstreams page is also linked from the **Performance** view header.

## Page-to-page flow

```
Dashboard (/)
 ├─ Performance (/performance)
 │   └─ Upstreams (/upstreams)
 └─ Upstreams (/upstreams)
     ├─ Dashboard (/)
     └─ Performance (/performance)
```

## What the Upstreams page shows

- **Summary cards** — provider count, health status, total models, cache entries.
- **Controls** — refresh provider data or clear the discovery cache.
- **Provider cards** — health indicator, provider type (OpenAI/Ollama), endpoint URL, priority, available models, and alias coverage.
- **Cache metrics** — TTL settings and hit counters for each provider.

## Tips for daily use

- Bookmark `/upstreams` for direct access when troubleshooting.
- The page refreshes automatically every 30 seconds; use **Refresh** for immediate updates.
- Mobile layouts are supported for quick checks from a phone or tablet.
