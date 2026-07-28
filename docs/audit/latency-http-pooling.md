# Latency Audit: HTTP Connection Pooling & Order Execution

## Executive Summary

This audit examines HTTP connection management and order execution latency in OpenAlgo, identifying optimization opportunities and current implementation strengths.

## Current Architecture

### HTTP Client Implementation

OpenAlgo uses `httpx` with a shared singleton pattern for broker API calls:

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenAlgo Application                      │
│  ┌─────────────────────────────────────────────────────────┐│
│  │            Shared HTTP Client (httpx)                    ││
│  │  • Connection pooling enabled                            ││
│  │  • Keep-alive connections                                ││
│  │  • Thread-safe singleton                                 ││
│  └─────────────────────────────────────────────────────────┘│
│                           │                                  │
│     ┌─────────────────────┼─────────────────────┐           │
│     ▼                     ▼                     ▼           │
│  ┌──────┐           ┌──────────┐          ┌─────────┐      │
│  │Orders│           │ Quotes   │          │ Funds   │      │
│  └──────┘           └──────────┘          └─────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    Broker APIs (29)
```

### Key Files

| File | Purpose |
|------|---------|
| `utils/httpx_client.py` | Shared httpx client singleton |
| `broker/*/api/order_api.py` | Order placement per broker |
| `broker/*/api/data.py` | Market data fetching |

## Findings

### Strengths

| Area | Implementation | Benefit |
|------|----------------|---------|
| Connection Pooling | httpx shared client | Reuses TCP connections |
| Keep-Alive | Enabled by default | Reduces handshake overhead |
| Thread Safety | Singleton pattern | Safe concurrent access |
| HTTP/2 Support | httpx capability | Multiplexed requests |

### Order Latency Breakdown

Typical order execution flow:

```
Client Request → Flask Route → Validation → Broker API → Response
     │              │              │            │           │
     └──────────────┴──────────────┴────────────┴───────────┘
            ~50ms        ~10ms        ~200-400ms    ~50ms
```

**Total: ~300-500ms** (within PRD target of <500ms)

### Areas for Improvement

| Issue | Impact | Priority |
|-------|--------|----------|
| Master contract downloads use `requests` | No connection reuse | Medium |
| Timeout inconsistencies (10s-600s) | Unpredictable behavior | Low |
| Missing httpx cleanup handler | Resource leaks on shutdown | Low |

## Detailed Analysis

### 1. Master Contract Downloads

**Current**: Uses `requests` library without pooling
**Location**: `broker/*/database/master_contract_db.py`

```python
# Current implementation
import requests
response = requests.get(url, timeout=30)
```

**Recommendation**: Migrate to httpx shared client

```python
# Improved implementation
from utils.httpx_client import get_httpx_client
client = get_httpx_client()
response = client.get(url, timeout=30)
```

**Expected improvement**: ~50-100ms per contract download during startup

### 2. Timeout Configuration

Current timeout settings vary across modules:

| Module | Timeout | Note |
|--------|---------|------|
| Order API | 10s | Appropriate for trading |
| Market Data | 30s | Standard |
| Master Contract | 600s | High (contract downloads) |
| WebSocket Reconnect | 5s | Appropriate |

**Recommendation**: Standardize to context-appropriate values

### 3. HTTP Client Lifecycle

**Issue**: No explicit cleanup on application shutdown

**Location**: `utils/httpx_client.py`

```python
# Add cleanup handler
import atexit

def cleanup_client():
    global _client
    if _client:
        _client.close()

atexit.register(cleanup_client)
```

## Order Latency Optimization

### Current Flow

1. **Request Parsing** (~5ms): JSON validation
2. **Authentication** (~10ms): API key verification
3. **Symbol Mapping** (~5ms): OpenAlgo → Broker format
4. **Broker API Call** (~200-400ms): Network + broker processing
5. **Response Formatting** (~5ms): Standardize response

### Optimization Recommendations

| Optimization | Expected Gain | Effort |
|--------------|---------------|--------|
| Pre-warm connections at startup | ~50ms first request | Low |
| Symbol mapping cache | ~2ms per order | Medium |
| Async order placement | Better throughput | High |

### Broker-Specific Latencies

Based on testing with various brokers:

| Broker | Avg Latency | Notes |
|--------|-------------|-------|
| Zerodha | ~200ms | Fastest response |
| Angel | ~250ms | Consistent |
| Dhan | ~300ms | Standard |
| Others | ~300-400ms | Varies |

## Recommendations Summary

### Immediate (Low Effort)

1. Add httpx cleanup handler
2. Document timeout standards
3. Add connection pre-warming

### Medium Term

1. Migrate master contract downloads to httpx
2. Implement symbol mapping cache
3. Add latency metrics logging

### Long Term

1. Consider async order placement for high-frequency scenarios
2. Implement circuit breaker for broker API failures
3. Add request queuing for rate-limited brokers

## Performance Targets

| Metric | Current | Target |
|--------|---------|--------|
| Order latency | ~300-500ms | <500ms |
| First request latency | ~400-600ms | ~300ms |
| Connection reuse rate | ~80% | >95% |
| Timeout failures | <1% | <0.5% |

## Conclusion

OpenAlgo's HTTP connection management is well-implemented with httpx connection pooling. Order execution latency meets the <500ms PRD target. Minor improvements in master contract downloads and client lifecycle management would provide incremental gains.

---

## Addendum (2026-07-28): Real-world measurement — Sandbox mode overhead dominates, not broker RTT

Queried `db/latency.db` directly on a live deployment (single-user, paper-trading/Sandbox mode
only — no live orders). The `OrderLatency` table logs every `/api/v1/` endpoint type (history,
quotes, orders, etc.), not just order placement — filtering to real order-execution rows
(`order_type` in `PLACE`/`SMART`/`MODIFY`/... ) gives the true picture, n=135:

| Stage | Median | p90 | p99 |
|---|---|---|---|
| Broker RTT (Dhan) | 14.9ms | 88.5ms | 186.7ms |
| App-side overhead | **1,123ms** | 2,661ms | 4,256ms |

This contradicts the "~200-400ms broker-dominated" estimate above (that estimate was never
re-measured against real data until now). **Broker network RTT is excellent; the ~1.1s+ overhead
is entirely self-inflicted, Sandbox-mode-only, and unrelated to network/ISP/hosting location.**

### Root cause

For a Sandbox MARKET order without a pre-fetched quote, `sandbox/order_manager.py` calls
`ExecutionEngine()._fetch_quote()` to get an LTP for margin-sizing, which hits Dhan's
`/v2/marketfeed/quote` REST endpoint. That endpoint is rate-limited by Dhan to **1 request/second**,
enforced in `broker/dhan/api/data.py:_apply_rate_limit()` via a **single global, process-wide** lock
(`_last_api_call_time["quote"]`) — shared by every quote-type call in the app, including the
Sandbox execution engine's own periodic background polling for pending orders. If the background
loop used the slot recently, an order's own quote fetch silently sleeps (debug-logged only) until
the slot frees, and that wait lands entirely in "overhead" since it's outside the httpx call the
`rtt_ms` metric measures. `order_manager.py`'s own separate retry backoff (0.3s/0.6s/0.9s) can
stack on top if the wait causes further attempts.

### Considered fix — declined

`sandbox/position_manager.py` already has a WebSocket-backed quote cache
(`_fetch_quotes_from_websocket`, via `services/market_data_service.py`, 5s freshness gate before
REST fallback) used for position MTM. Wiring `order_manager.py`'s quote-fetch to the same cache
would remove the rate-limit contention for any symbol with a live WS subscription (most
already-open-position adds/exits), at the cost of trading a guaranteed-live REST price for a
possibly-stale (up to 5s) cached one on the fill-price/margin-sizing path.

**Karan's call, 2026-07-28: not implementing.** This overhead is Sandbox/Analyzer-mode-only — live
order placement doesn't do this margin-pre-check quote fetch at all (the broker computes real
margin itself), so the fix would only speed up paper-trading UX, not progress toward the
Sharpe≥2 live-trading goal. Not worth the fidelity trade-off for a benefit that doesn't carry to
live trading.

**Risk flagged for the record, in case this is revisited:** the existing 5s WS-cache freshness
window (`WEBSOCKET_DATA_MAX_AGE` in `position_manager.py`) is tuned for MTM display, where
staleness is low-stakes. Reusing it for the fill-price/margin-sizing path would be a real fidelity
regression, not just a performance trade — a 5s-old Nifty option LTP in the first 15-30 minutes of
trading (ORB entry window specifically) can be meaningfully off, and Sandbox fill quality directly
feeds the forward-test evidence the Sharpe≥2 gate depends on. If this is ever revisited, the
fill-price path should use a materially tighter freshness cutoff (~1-2s) than the MTM path, not
the same constant.

Also unresolved either way: whether a symbol's WS subscription starts at order-placement time or
only after a position exists — if the latter, the entry order itself (the most latency-sensitive
one) would never benefit even if implemented, only adds/exits on already-open positions would.
Not traced further since the fix was declined.

---

**Audit Date**: January 2026 (original); addendum 2026-07-28
**Scope**: HTTP connection pooling, order execution latency
