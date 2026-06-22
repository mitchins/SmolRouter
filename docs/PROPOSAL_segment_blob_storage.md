# Proposal: Segment (block) blob storage backend

## Goal

Replace one-OS-file-per-body blob storage with a **log-structured segment store**:
payloads are appended into a small number of bounded segment files, and the blob
"key" encodes `(segment, offset, length)`. This collapses inode usage to a handful
of files, amortizes `fsync`, and makes retention a whole-segment `O(1)` drop
instead of a per-file unlink. It is the directionally-correct "block storage"
shape (à la SQLite/InnoDB pages-in-a-file) specialized for this workload — but
**log-structured, not B-tree+free-list**, because Redis already owns the index and
retention is purely temporal.

This is an optimization behind the existing `BlobStorage` ABC, not a correctness
fix. See the prerequisite below — that one *is* the correctness/perf bug.

## Prerequisite (separate, urgent): kill the O(N²) hot-path scan

`FilesystemBlobStorage.store()` (`smolrouter/storage.py:111`) calls
`_total_size_bytes()` → `_dir_size_bytes(base_path)` → `rglob("*.blob") + .stat()`
over the **entire** tree on **every** write, and `_archive_bodies_after_completion`
(`smolrouter/database.py:766`) calls `store()` **twice per request** (request +
response body). That is an `O(N)` Python tree-walk per write — `O(N²)` over the
store's life — run **synchronously on the asyncio event loop**. On the production
box (local ext4/NVMe, ~73k blobs) this serializes archival behind ~1–2s scans and
produced a **multi-hour** body-archival backlog (a request from 21:00 whose blob
landed at 07:48 next morning). Disk speed is irrelevant; it's the algorithm + loop
blocking.

**Fix independently, first, paradigm-agnostic:**
- Track size incrementally (counter `+len` on write / `−size` on prune) instead of
  scanning; enforce the cap **only in the janitor** (`_janitor_loop`, already
  off-thread via `asyncio.to_thread`).
- Offload the write off the event loop (`asyncio.to_thread`).
- Cache created hour-dirs to drop the per-write `mkdir(parents)` + `exists()`.

After this fix, file-per-blob on local ext4 is *acceptable*. The segment store
below is the next-step efficiency upgrade, not a substitute for this fix.

## Problem (why file-per-blob is the wrong long-term shape)

Bodies are small (~1–25 KB), write-once, **read-rarely** (only when an operator
opens a request in the dashboard), disposable, and capped + time-pruned. Against
that profile, one OS file per body is the worst fit:

- **Inode + dir-entry churn** — one inode per body; large hour-buckets; retention
  is a per-file `unlink` walk.
- **Block-size amplification** — a 999-byte body still consumes a full 4 KB ext4
  block.
- **No fsync amortization** — every body is its own create/write/(sync).
- **Retention is a scan** — `cleanup_old`/`_cleanup_for_space` walk and stat the
  tree.

(NB: the `95% inodes` figure seen during diagnosis came from `df` **over SMB**,
which reports synthetic numbers for the share — not the box's real ext4. Confirm
with `df -i /` on the box before treating inode exhaustion as live. The segment
store removes the question regardless.)

## Key insight

We do **not** need the heavy parts of SQLite/InnoDB:

- **The index already lives in Redis.** The blob key is ours and is stored on the
  request hash as `request_body_key` / `response_body_key`
  (`smolrouter/redis_backend.py` `update_body_keys`). So no on-disk B-tree is
  needed — Redis *is* the index.
- **Retention is purely temporal** (drop oldest). So we don't need a free-list /
  in-place reuse — we drop whole old segments.

That collapses "block storage" to its **log-structured / Bitcask** form, which is
the right specialization here.

SQLite specifically is rejected as the *engine*: single-writer, database-level
write lock; even WAL serializes writers and chokes under concurrent archival. The
*page-in-a-file direction* is right; the *engine* is wrong for this concurrency
profile.

## Design

### Principles

1. Implement behind the existing `BlobStorage` ABC (`smolrouter/storage.py:29`):
   `store(data) -> key`, `retrieve(key) -> bytes|None`, `delete`, `exists`,
   `cleanup_old`. The key becomes opaque-but-structured; callers (database.py)
   are unchanged.
2. **Single appender, many readers.** One writer serializes appends (no lock
   contention, sequential I/O, batched `fsync`). Reads are lock-free `pread`.
3. **Redis is the index.** Nothing on disk needs to be scanned to find a body.
4. **Retention = drop whole segments.** `O(1)`, no tree walk.

### Key format

`seg:{segment_id}:{offset}:{length}` (opaque to callers; round-trips through
Redis `*_body_key` fields unchanged). `segment_id` is a zero-padded monotonic
counter so segments sort by age lexically.

### On-disk layout

```text
blob_storage/
  segments/
    0000000123.seg        # bounded append-only file (e.g. 64–256 MB)
    0000000124.seg        # current active segment
  manifest.json           # active segment id, sealed segment list, sizes
```

Each appended record: `[u32 length][bytes payload]` (length-prefixed for
self-description / recovery; the authoritative `(offset,length)` is in Redis).

### Write path (`store`)

1. Serialize through the single appender (an `asyncio.Queue` drained by one worker,
   or a lock around a thread-offloaded append).
2. `offset = current_active.size`; append length-prefix + payload; `size += n`.
3. If `size >= SEGMENT_MAX_BYTES`, **seal** active and open the next segment.
4. `fsync` is **batched** — on seal and/or every `FSYNC_INTERVAL_MS` — not per
   record. Return key `seg:{id}:{offset}:{length}` once the offset is reserved.
5. The actual file I/O runs via `asyncio.to_thread` so the loop never blocks.

### Read path (`retrieve`)

Parse key → `pread(fd, offset, length)` on the segment file (cached fd, LRU of
open segment fds). Concurrent and lock-free. Missing segment/oob → `None` (same
contract as today).

### Retention (`cleanup_old` / cap enforcement)

- Maintain total size from the manifest (sum of segment sizes) — no scan.
- When over cap or past `max_age_days`, **delete oldest sealed segment files
  whole** and drop them from the manifest. `O(segments)`, not `O(blobs)`.
- Orphaned keys (Redis points into a dropped segment) resolve to `None`, which the
  UI already tolerates — and is the same outcome as today's pruned files.

### Concurrency & crash safety

- One appender ⇒ no writer contention (the explicit answer to "SQLite has terrible
  threading").
- Crash window = unsynced tail of the active segment: at most the last
  `FSYNC_INTERVAL_MS` of bodies. Acceptable for disposable debug data. On restart,
  trust the manifest + length-prefixes to find the valid tail; truncate a partial
  trailing record.
- Redis remains the source of truth for *which* keys exist; the segment store only
  resolves bytes.

### Config (env)

- `BLOB_BACKEND=filesystem|segment|memory` (default stays `filesystem` until proven)
- `SEGMENT_MAX_BYTES` (default 128 MB)
- `FSYNC_INTERVAL_MS` (default e.g. 1000)
- Existing `MAX_TOTAL_STORAGE_SIZE`, `MAX_BLOB_SIZE`, janitor knobs reused.

## Phasing

1. **Prereq fix** (separate PR): remove the O(N²) hot-path scan + offload I/O.
   This is what actually ends the production lag.
2. **`SegmentBlobStorage` MVP** behind the ABC: single appender, append/seal/rotate,
   `pread` reads, manifest, batched fsync. Default off.
3. **Retention**: whole-segment drop by age/cap; manifest-driven size accounting.
4. **Crash recovery**: manifest + length-prefix tail validation on boot.
5. **Flip default** to `segment` after soak; keep `filesystem` (post-prereq) as the
   block-on-local-disk fallback and `memory` for tests.

## Non-goals (for now)

- On-disk indexing / B-tree (Redis is the index).
- In-place update / free-list reuse / compaction of live segments (retention is
  temporal; we drop whole segments).
- Moving bodies into Redis RAM (considered; rejected — on a router box NVMe is
  cheaper to spend than RAM, and durability-on-disk is wanted).
- Cross-host / object-store backends (S3/MinIO) — out of scope for a LAN box.

## Open questions

- Appender model: dedicated `asyncio` worker + queue, or a thread with a lock?
  (Queue gives natural backpressure + batching; thread is simpler.)
- Open-fd cache size / eviction for reads across many sealed segments.
- Do we need a tiny per-segment footer index for standalone recovery, or is
  "Redis + length-prefixes" enough? (Leaning: enough.)
- Migration: leave existing `*.blob` files readable via the filesystem backend in
  parallel, or one-shot import into segments? (Leaning: dual-read, no import —
  old bodies age out under retention.)

## Verification

1. Round-trip: `store` then `retrieve` returns identical bytes across segment
   rotation boundaries.
2. **O(1) write**: assert `store()` does not scan the tree / is constant in
   existing-blob count (round-trip/stat-count harness, as in the dashboard perf
   tests).
3. Retention drops whole segments and keeps total under cap without a per-blob walk.
4. Crash-tail: kill mid-append; on restart, sealed data intact, partial tail
   truncated, no corruption.
5. Concurrent reads during active appends return correct bytes.
