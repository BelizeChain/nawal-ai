# Nawal AI — Remaining Audit Findings Remediation Plan

**Created**: 2026-03-01  
**Audit Reference**: `docs/audits/2026-02-28-catastrophic-audit.md` (131 findings)  
**Prior Remediation**: Tasks 1–23 (COMPLETE — 240 tests passing)  
**Remaining Findings**: ~81 findings (1 Critical, 11 High, 35 Medium, 33 Low)  
**Estimated Effort**: 29 tasks across 8 phases  

---

## Status of Prior Remediation (Tasks 1–23)

All 23 tasks from the initial remediation plan have been executed and verified:

| Task | Findings Fixed | Status |
|------|---------------|--------|
| 1. `torch.load` → `weights_only=True` | C5 (9 sites) | ✅ |
| 2. CORS + Auth | C3, C4 | ✅ |
| 3. `trust_remote_code` removal | H2 (P3-26) | ✅ |
| 4. Mock encryption hard-fail | H5 (P2-09) | ✅ |
| 5. `DummyBelizeIDVerifier` guard | H3 (P1-11) | ✅ |
| 6. `DPInferenceGuard` implementation | C2 (P2-14, P2-15) | ✅ |
| 7. Pakit upload fix | H10, B10 (P5-07) | ✅ |
| 8. `//Alice` removal | H7 (P5-04) | ✅ |
| 9. Error leak mitigation | M9 (P5-10, P2-19) | ✅ |
| 10. Path traversal sanitization | H4 (P1-28, P1-29) | ✅ |
| 11. ZK-proof stub hardening | C1 (P1-21, P1-22) | ✅ |
| 12. Dockerfile hardening | M11, L5 (P5-13) | ✅ |
| 13. Bug fixes B1–B9 | B1–B9 (P3-04, P3-17, P3-20, P3-32–P3-34, B2, B8, B9) | ✅ |
| 14. Dependency alignment | M10 (P5-12) | ✅ |
| 15. Rate limiting | API endpoints | ✅ |
| 16. Config fixes | L1 (P5-14) | ✅ |
| 17. DP budget enforcement | M1 (P2-02) | ✅ |
| 18. Mesh signature verification | H1 (P1-15, P1-16) | ✅ |
| 19. FL client integrity checks | H6 (P4-02) | ✅ |
| 20. Monitoring dedup | M12 (P5-15) | ✅ |
| 21. Kinich async fix | M13 (P5-16) | ✅ |
| 22. Byzantine detection upgrade | P2-16, P2-17 | ✅ |
| 23. Final validation | All tests green | ✅ |

**Baseline**: 240 passed, 3 skipped, 0 failures  
(1 intermittent failure in `test_integrated_poisoning_detection_clean_data` — test ordering issue, passes in isolation)

---

## Remaining Findings Summary

| Severity | Count | Key Areas |
|----------|-------|-----------|
| **Critical** | 1 | Secure aggregation mask generation (P2-08) |
| **High** | 11 | Code bugs, DP clipping, checkpoint integrity, replay attacks, PII, tax brackets |
| **Medium** | 35 | Genome operators, DP mechanics, infra robustness, API binding, caching |
| **Low** | 33 | datetime deprecation, dead code, logging inconsistency, type hints |

---

## Phase A — Critical + High Code Bugs (Tasks 1–3)

> **Goal**: Fix code-level bugs that will crash at runtime  
> **Estimated effort**: 2–3 hours  

### Task 1: Fix `GenomeModel.forward()` indentation bug + Pydantic fallback

**Findings**: P3-19 (HIGH), P3-14 (HIGH)

**File 1**: `genome/model_builder.py` (~L700–710)

**Problem (P3-19)**: Three lines after the `else` block are dedented and execute unconditionally — overwriting `hidden_states` even when rotary/ALiBi embeddings were already applied:
```python
        # These run OUTSIDE the else block (wrong):
        position_ids = torch.arange(seq_length, device=x.device).unsqueeze(0)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = token_embeds + position_embeds
```

**Fix**: Indent these three lines into the `else` block so they only execute for learned positional embeddings.

**File 2**: `genome/fitness.py` (~L24–28) and `genome/encoding.py` (~L19–23)

**Problem (P3-14)**: Pydantic fallback silently disables all validation:
```python
try:
    from pydantic import BaseModel, Field, ConfigDict
except ImportError:
    from typing import Any as BaseModel
```

**Fix**: Replace fallback with an immediate `raise ImportError("pydantic is required")` or remove the try/except entirely (pydantic is a hard dependency).

**Verification**:
- `python -m pytest tests/test_model_builder.py tests/test_genome.py -v`
- Manual: `from genome.model_builder import GenomeModel` succeeds
- Manual: Remove pydantic temporarily, confirm hard failure instead of silent degradation

---

### Task 2: Fix `.get()` on Pydantic model in `nawal_adapter.py`

**Findings**: P3-16 (HIGH)

**File**: `genome/nawal_adapter.py` (~L82, ~L86)

**Problem**: `Hyperparameters` is a Pydantic `BaseModel`, not a dict — `.get()` raises `AttributeError`:
```python
genome.hyperparameters.get("activation", "gelu").lower()
genome.hyperparameters.get("dropout_rate", 0.1)
```

**Fix**: Replace with `getattr()`:
```python
getattr(genome.hyperparameters, "activation", "gelu").lower()
getattr(genome.hyperparameters, "dropout_rate", 0.1)
```

**Verification**:
- `python -m pytest tests/test_genome.py tests/test_evolution.py -v`
- Manual: `from genome.nawal_adapter import GenomeToNawalAdapter` and call conversion

---

### Task 3: Fix silent mock mode fallback in `staking_connector.py`

**Findings**: P1-04 (HIGH)

**File**: `blockchain/staking_connector.py` (~L134–142)

**Problem**: If `substrateinterface` isn't installed, connector silently falls back to mock mode:
```python
self.mock_mode = mock_mode or not SUBSTRATE_AVAILABLE
```

**Fix**: 
- If `mock_mode` is explicitly `False` but `SUBSTRATE_AVAILABLE` is `False`, raise `ImportError`
- Add a startup warning log when mock mode is auto-engaged
- Add a `strict_mode` parameter that fails hard when substrate is unavailable

```python
if not mock_mode and not SUBSTRATE_AVAILABLE:
    logger.critical("substrateinterface not installed — cannot run in production mode")
    raise ImportError(
        "substrateinterface is required for non-mock mode. "
        "Install with: pip install substrate-interface"
    )
self.mock_mode = mock_mode
```

**Verification**:
- `python -m pytest tests/test_blockchain.py -v`
- Manual: Confirm mock mode still works when explicitly requested

---

## Phase B — High Security Hardening (Tasks 4–9)

> **Goal**: Fix critical security architecture issues  
> **Estimated effort**: 3–4 days  

### Task 4: Redesign secure aggregation — client-generated masks

**Findings**: P2-08 (CRITICAL)

**File**: `security/secure_aggregation.py` (~L370–400)

**Problem**: Server generates ALL pairwise masks centrally in `generate_pairwise_masks()`. The server knows all individual masks, completely defeating the purpose of secure aggregation.

**Fix**:
1. Implement Diffie-Hellman key exchange between client pairs
2. Clients generate their own pairwise masks from shared secrets
3. Server only receives masked (encrypted) updates
4. Server aggregates masked values — individual values never visible

**Implementation approach**:
- Add `ClientKeyExchange` class using `cryptography.hazmat.primitives.asymmetric.x25519`
- Each client generates X25519 keypair, shares public key with server
- Server distributes public keys to client pairs (server never sees private keys)
- Clients compute `shared_secret = private_key.exchange(peer_public_key)`
- Mask derived from `HKDF(shared_secret)` 
- Server's `generate_pairwise_masks()` replaced with `distribute_public_keys()`

**Verification**:
- Unit test: 3 clients perform secure aggregation, server cannot recover individual updates
- `python -m pytest tests/ -v` — no regressions

---

### Task 5: Fix DP — per-example clipping + dropout mask compensation

**Findings**: P2-01 (HIGH), P2-10 (HIGH)

**File 1**: `security/differential_privacy.py` (~L161–178)

**Problem (P2-01)**: Clips each parameter's gradient independently, not per-example:
```python
for param in model.parameters():
    if param.grad is not None:
        torch.nn.utils.clip_grad_norm_([param], max_norm=self.max_grad_norm)
```

**Fix**: Replace per-parameter clipping with whole-model gradient norm clipping (standard DP-SGD):
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_grad_norm)
```
Note: True per-sample clipping requires Opacus integration. For now, whole-model clipping is the correct non-Opacus approach.

**File 2**: `security/secure_aggregation.py` (~L600–620)

**Problem (P2-10)**: `dropout_resilient_aggregation()` scales up remaining updates but doesn't compensate for non-cancelled masks from dropped clients.

**Fix**: Track which client pairs have both members present. For pairs where one member dropped:
- Subtract the missing client's mask contribution from the present client before aggregation
- Or: Re-run mask generation excluding dropped clients (expensive but correct)

**Verification**:
- `python -m pytest tests/test_differential_privacy.py -v`
- Manual: Verify gradient norms are properly bounded

---

### Task 6: Add checkpoint integrity verification

**Findings**: P5-08 (HIGH)

**File**: `storage/checkpoint_manager.py` (~L115–155)

**Problem**: Checkpoints downloaded from Pakit are loaded via `torch.load()` without verifying content hash against expected CID.

**Fix**:
1. Before `torch.load()`, compute SHA-256 of downloaded file
2. Compare against expected content hash from registry
3. Reject and raise if mismatch
4. Add Ed25519 signature verification (optional, if signing infrastructure available)

```python
import hashlib

def _verify_checkpoint_integrity(self, filepath: Path, expected_hash: str) -> bool:
    """Verify checkpoint file matches expected content hash."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    computed_hash = sha256.hexdigest()
    if computed_hash != expected_hash:
        raise ValueError(
            f"Checkpoint integrity check failed: "
            f"expected {expected_hash}, got {computed_hash}"
        )
    return True
```

**Verification**:
- `python -m pytest tests/ -v`
- Manual: Tamper with a checkpoint file, confirm load fails

---

### Task 7: Fix `seen_messages` replay + mesh rate limiting + binding

**Findings**: P1-17 (HIGH), P1-18 (MEDIUM), P1-20 (MEDIUM)

**File**: `blockchain/mesh_network.py`

**Problem (P1-17)** (~L660–670): `seen_messages` clears entirely at 10K entries, enabling replay of ALL previously seen messages:
```python
if len(self.seen_messages) > 10000:
    self.seen_messages.clear()
```

**Fix**: Replace with LRU eviction — remove oldest entries instead of clearing all:
```python
from collections import OrderedDict

# In __init__:
self.seen_messages: OrderedDict[str, float] = OrderedDict()
self.max_seen_messages = 10000

# In cleanup:
while len(self.seen_messages) > self.max_seen_messages:
    self.seen_messages.popitem(last=False)  # Remove oldest
```

**Problem (P1-18)**: No rate limiting on `/message` HTTP endpoint.

**Fix**: Add per-IP rate limiter (token bucket or sliding window) to the message handler. Reject with HTTP 429 when exceeded.

**Problem (P1-20)**: Mesh HTTP binds to `0.0.0.0`.

**Fix**: Default to `127.0.0.1` and require explicit configuration for external binding:
```python
listen_host = config.get("mesh_host", "127.0.0.1")
```

**Verification**:
- `python -m pytest tests/test_mesh_network.py -v`
- Manual: Verify old messages can't be replayed after eviction

---

### Task 8: Add content integrity verification for genome retrieval

**Findings**: P1-30 (HIGH)

**File**: `blockchain/genome_registry.py` (~L231–249)

**Problem**: Data retrieved from IPFS/Arweave/local is never verified against expected content hash — tampered content accepted silently.

**Fix**: After retrieval from any backend, compute hash and compare:
```python
def _verify_content(self, data: bytes, expected_hash: str) -> bool:
    computed = hashlib.sha256(data).hexdigest()
    if computed != expected_hash:
        raise IntegrityError(
            f"Content hash mismatch: expected {expected_hash}, got {computed}"
        )
    return True
```

Call `_verify_content()` in `retrieve_genome()` before returning data.

**Verification**:
- `python -m pytest tests/test_blockchain.py -v`
- Manual: Corrupt a local genome file, confirm retrieval raises error

---

### Task 9: Hash/encrypt PII before chain storage + Merkle input hashing

**Findings**: P1-08 (HIGH), P1-24 (MEDIUM)

**File 1**: `blockchain/validator_manager.py` (~L80–91)

**Problem (P1-08)**: `to_dict()` serializes `email`, `legal_name`, `tax_id` as plaintext for on-chain storage.

**Fix**: Hash PII fields before chain serialization:
```python
def to_dict(self) -> dict:
    return {
        "email_hash": hashlib.sha256(self.email.encode()).hexdigest(),
        "legal_name_hash": hashlib.sha256(self.legal_name.encode()).hexdigest(),
        "tax_id_hash": hashlib.sha256(self.tax_id.encode()).hexdigest(),
        # ... non-PII fields remain as-is
    }
```
Keep plaintext PII in local encrypted storage only, not on-chain.

**File 2**: `blockchain/payroll_connector.py` (~L613–629)

**Problem (P1-24)**: Merkle tree constructed from plaintext salary data — `employee_id + gross_salary + net_salary` concatenated as strings.

**Fix**: Hash each leaf BEFORE constructing the Merkle tree:
```python
def _compute_merkle_leaf(self, employee_id: str, gross: float, net: float) -> bytes:
    data = f"{employee_id}:{gross}:{net}".encode()
    return hashlib.sha256(data).digest()
```
This ensures the Merkle tree doesn't leak salary data if intermediate nodes are inspected.

**Verification**:
- `python -m pytest tests/test_payroll.py tests/test_blockchain.py -v`
- Manual: Verify on-chain data no longer contains plaintext PII

---

## Phase C — Data Correctness & Privacy (Tasks 10–13)

> **Goal**: Fix data handling errors and privacy leaks  
> **Estimated effort**: 1–2 days  

### Task 10: Fix tax brackets in payroll_connector

**Findings**: P1-23 (HIGH)

**File**: `blockchain/payroll_connector.py` (~L672–678)

**Problem**: Tax brackets hardcoded as 0%/20%/25% but Belize Income Tax Act specifies:
- 0% for income ≤ BZD 26,000/year
- 25% for income > BZD 26,000/year

The 20% intermediate bracket appears incorrect.

**Fix**: Align with Belize Income Tax Act:
```python
def _calculate_tax(self, annual_income: float) -> float:
    STANDARD_DEDUCTION = 26_000.00  # BZD
    TAX_RATE = 0.25
    taxable = max(0.0, annual_income - STANDARD_DEDUCTION)
    return taxable * TAX_RATE
```

**Verification**:
- `python -m pytest tests/test_payroll.py -v`
- Manual: Verify BZD 30,000 income → BZD 1,000 tax (25% of 4,000)

---

### Task 11: Fix perplexity token shift + PII redaction in fallback log

**Findings**: P3-24 (MEDIUM), P3-23 (MEDIUM)

**File 1**: `hybrid/confidence.py` (~L107–113)

**Problem (P3-24)**: Perplexity computation doesn't shift tokens — compares logits against same input instead of next token:
```python
F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
```

**Fix**: Standard next-token perplexity:
```python
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = target_ids[:, 1:].contiguous()
loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1)
)
perplexity = torch.exp(loss)
```

**File 2**: `hybrid/router.py` (~L131–142)

**Problem (P3-23)**: Full user query text logged to JSONL file — PII/privacy violation:
```python
f.write(json.dumps({"query": query, ...}) + "\n")
```

**Fix**: Redact or hash the query before logging:
```python
import hashlib
query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
f.write(json.dumps({
    "query_hash": query_hash,
    "query_length": len(query),
    "reason": reason,
    "timestamp": timestamp,
}) + "\n")
```

**Verification**:
- `python -m pytest tests/ -v`
- Manual: Check `logs/fallback_queries.jsonl` no longer contains raw queries

---

### Task 12: Add LRU eviction to teacher cache + fix cache key

**Findings**: P3-27 (MEDIUM), P3-28 (LOW)

**File**: `hybrid/teacher.py` (~L50, ~L155)

**Problem (P3-27)**: `self.cache = {}` grows without bound — memory leak.  
**Problem (P3-28)**: Cache key `(prompt, max_tokens, temperature)` ignores `top_p`/`top_k`.

**Fix**: Use `functools.lru_cache` or implement manual LRU with `OrderedDict`:
```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, maxsize: int = 1000):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
    
    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
    
    def set(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
```

Include `top_p` and `top_k` in cache key:
```python
cache_key = (prompt, max_tokens, temperature, top_p, top_k)
```

**Verification**:
- `python -m pytest tests/ -v`
- Manual: Verify cache doesn't grow past maxsize

---

### Task 13: Fix Arweave fallback + rewards persistence warnings

**Findings**: P1-25/P1-32 (MEDIUM), P1-26 (MEDIUM)

**File 1**: `blockchain/genome_registry.py` (~L410–425)

**Problem (P1-25/P1-32)**: Arweave storage silently falls back to local storage. Code claims decentralized but stores locally.

**Fix**: 
- Log at `WARNING` level when Arweave fallback occurs
- Add a `storage_backend` field to return metadata so callers know where data actually went
- In strict mode, raise instead of falling back

```python
if self.strict_storage:
    raise StorageError("Arweave storage unavailable and strict_storage=True")
logger.warning(
    f"Arweave storage unavailable for {genome_id}, falling back to local storage. "
    "Data is NOT decentralized."
)
```

**File 2**: `blockchain/rewards.py` (~L310–312)

**Problem (P1-26)**: All reward tracking is in-memory only — lost on restart.

**Fix**: Add startup warning and optional persistence:
```python
logger.warning(
    "RewardDistributor using in-memory storage — "
    "pending rewards will be lost on process restart. "
    "Set NAWAL_REWARDS_PERSIST=true for file-based persistence."
)
```
Optionally add JSON file persistence for pending rewards.

**Verification**:
- `python -m pytest tests/test_blockchain.py -v`
- Manual: Verify Arweave fallback now shows visible warning

---

## Phase D — Genome/Evolution Fixes (Tasks 14–15)

> **Goal**: Fix genome operator bugs that corrupt evolution  
> **Estimated effort**: 4–6 hours  

### Task 14: Fix genome operators — `_mutate_add_moe`, crossover decoders, `output_normalization`

**Findings**: P3-07 (MEDIUM), P3-08 (MEDIUM), P3-06 (MEDIUM)

**File 1**: `genome/operators.py` (~L380)

**Problem (P3-07)**: `_mutate_add_moe()` accesses `genome.hidden_size` directly — should be `genome.hyperparameters.hidden_size`:
```python
# Bug:
genome.hidden_size
# Fix:
genome.hyperparameters.hidden_size
```

**File 2**: `genome/operators.py` (~L560–620)

**Problem (P3-08)**: All crossover methods only operate on `encoder_layers`. Decoder layers are silently dropped from offspring.

**Fix**: Apply same crossover logic to `decoder_layers`:
```python
# After encoder crossover:
if parent1.decoder_layers and parent2.decoder_layers:
    child.decoder_layers = self._single_point_crossover_layers(
        parent1.decoder_layers, parent2.decoder_layers
    )
elif parent1.decoder_layers:
    child.decoder_layers = copy.deepcopy(parent1.decoder_layers)
elif parent2.decoder_layers:
    child.decoder_layers = copy.deepcopy(parent2.decoder_layers)
```

**File 3**: `genome/encoding.py` (~L269)

**Problem (P3-06)**: `output_normalization` property returns `LayerType.LAYER_NORM` enum, but `NormalizationFactory` expects string `"layer_norm"`:
```python
def output_normalization(self) -> str:
    return LayerType.LAYER_NORM  # Returns enum, not str
```

**Fix**: Return the string value:
```python
def output_normalization(self) -> str:
    return LayerType.LAYER_NORM.value  # or return "layer_norm"
```

**Verification**:
- `python -m pytest tests/test_evolution.py tests/test_genome.py tests/test_model_builder.py -v`

---

### Task 15: Fix `asyncio.run` in fitness + `language_detector` + fitness scale

**Findings**: P3-15 (MEDIUM), P3-29 (MEDIUM), P3-18 (MEDIUM)

**File 1**: `genome/fitness.py` (~L312)

**Problem (P3-15)**: `asyncio.run()` crashes if called from async context (FastAPI/Flower):
```python
return asyncio.run(self.evaluate_async(genome, training_metrics))
```

**Fix**: Check for running event loop:
```python
import asyncio

def evaluate(self, genome, training_metrics=None):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop and loop.is_running():
        # Already in async context — create task or use nest_asyncio
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(self.evaluate_async(genome, training_metrics))
    else:
        return asyncio.run(self.evaluate_async(genome, training_metrics))
```

Or better: make `evaluate()` sync by extracting the logic, and keep `evaluate_async()` for async callers.

**File 2**: `hybrid/engine.py` (~L193)

**Problem (P3-29)**: Assumes `self.nawal.language_detector` exists:
```python
self.nawal.language_detector.detect(prompt)
```

**Fix**: Add `hasattr` guard:
```python
if hasattr(self.nawal, 'language_detector') and self.nawal.language_detector:
    detected_lang = self.nawal.language_detector.detect(prompt)
else:
    detected_lang = "en"  # Default fallback
    logger.debug("No language_detector available, defaulting to English")
```

**File 3**: `genome/nawal_adapter.py` (~L320)

**Problem (P3-18)**: Fitness scores return 0–1 range but `PoUWAlignment` uses 0–100 scale.

**Fix**: Normalize output to 0–100:
```python
# In get_genome_fitness_score():
loss_quality = min(100.0, max(0.0, (1.0 - loss / 10.0) * 100.0))
```
Or document that the scale is 0–1 and adjust PoUWAlignment to accept it.

**Verification**:
- `python -m pytest tests/test_evolution.py tests/test_genome.py -v`

---

## Phase E — Server/Client Robustness (Tasks 16–18)

> **Goal**: Fix silent failures and unsafe defaults in server/client  
> **Estimated effort**: 1 day  

### Task 16: Fix aggregation failure handling

**Findings**: P4-08 (MEDIUM)

**File**: `server/aggregator.py` (~L582)

**Problem**: `asyncio.create_task(self._aggregate_round(...))` is fire-and-forget — task exceptions silently lost.

**Fix**: Add exception callback:
```python
task = asyncio.create_task(self._aggregate_round(update.round_number))
task.add_done_callback(self._handle_aggregation_result)

def _handle_aggregation_result(self, task: asyncio.Task):
    try:
        task.result()
    except Exception as e:
        logger.error(f"Aggregation round failed: {e}")
        # Mark round as failed, notify participants
        self._mark_round_failed(task)
```

**Verification**:
- `python -m pytest tests/test_federation.py -v`

---

### Task 17: Fix synthetic data fallback + EducationModel + IoT validation

**Findings**: P4-09 (MEDIUM), P4-10 (MEDIUM), P4-11 (MEDIUM)

**File 1**: `client/data_loader.py` — synthetic data fallback

**Problem (P4-09)**: `_create_synthetic_data()` silently trains on random garbage when participant data directory missing.

**Fix**: Log warning and optionally raise:
```python
logger.warning(
    "No training data found — generating synthetic data. "
    "This produces MEANINGLESS model updates. "
    "Set NAWAL_REQUIRE_REAL_DATA=true to fail instead."
)
if os.getenv("NAWAL_REQUIRE_REAL_DATA", "").lower() == "true":
    raise FileNotFoundError("Training data directory not found and NAWAL_REQUIRE_REAL_DATA=true")
```

**File 2**: `client/domain_models.py` (~L800–820) — EducationModel

**Problem (P4-10)**: Uses `config.tokenizer_name` (default `"gpt2"`) as `model_name`:
```python
self.llm = BelizeChainLLM(model_name=config.tokenizer_name, ...)
```

**Fix**: Use a proper model name parameter or make the intent explicit:
```python
model_name = getattr(config, "model_name", None) or config.tokenizer_name
self.llm = BelizeChainLLM(model_name=model_name, ...)
```

**File 3**: `client/domain_models.py` (~L395, ~L1018) — IoT byte deserialization

**Problem (P4-11)**: `np.frombuffer(sensor_data, dtype=np.float32)` with no validation.

**Fix**: Add length and bounds validation:
```python
def _safe_frombuffer(data: bytes, dtype=np.float32) -> np.ndarray:
    if len(data) == 0:
        raise ValueError("Empty sensor data")
    if len(data) % np.dtype(dtype).itemsize != 0:
        raise ValueError(
            f"Data length {len(data)} not aligned to {dtype} "
            f"(itemsize={np.dtype(dtype).itemsize})"
        )
    arr = np.frombuffer(data, dtype=dtype)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Sensor data contains NaN or Inf values")
    return arr
```

**Verification**:
- `python -m pytest tests/test_training.py -v`

---

### Task 18: Fix API `0.0.0.0` binding + global mutable state

**Findings**: P5-09 (MEDIUM), P5-11 (MEDIUM)

**File 1**: `api_server.py` (~L725)

**Problem (P5-09)**: `main()` overrides `ServerConfig` default of `127.0.0.1` to `0.0.0.0`:
```python
host = os.getenv("NAWAL_API_HOST", os.getenv("NAWAL_HOST", os.getenv("HOST", "0.0.0.0")))
```

**Fix**: Use `ServerConfig.host` default as final fallback:
```python
host = os.getenv("NAWAL_API_HOST", os.getenv("NAWAL_HOST", os.getenv("HOST", "127.0.0.1")))
```

**File 2**: `api/inference_server.py` (~L38–42, ~L123)

**Problem (P5-11)**: Module-level mutable globals (`global_model`, `dp_guard`). Deprecated `@app.on_event("startup")`.

**Fix**: 
- Move state into `app.state` via lifespan context manager
- Replace `@app.on_event("startup")` with `lifespan` parameter

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.model = load_model()
    app.state.dp_guard = DPInferenceGuard(epsilon=2.0)
    yield
    # Shutdown
    app.state.model = None

app = FastAPI(lifespan=lifespan)
```

**Verification**:
- `python -m pytest tests/ -v`
- Manual: Verify API starts on 127.0.0.1 by default

---

## Phase F — Infrastructure Quality (Tasks 19–24)

> **Goal**: Fix infra-level issues — retries, imports, types, config  
> **Estimated effort**: 2–3 days  

### Task 19: Add RPC retry/backoff + fix receipt API mismatch

**Findings**: P1-01 (MEDIUM), P1-05 (MEDIUM)

**File**: `blockchain/substrate_client.py` (~L156–172) and `blockchain/staking_connector.py` (~L283, ~L365)

**Problem (P1-01)**: Single connection attempt, no retry.  
**Problem (P1-05)**: `is_success`/`error_message` vs `success`/`error` — the `ExtrinsicReceipt` dataclass uses different field names than what calling code expects.

**Fix (P1-01)**: Add exponential backoff:
```python
import time

def connect(self, max_retries: int = 3, base_delay: float = 1.0):
    for attempt in range(max_retries):
        try:
            self.substrate = SubstrateInterface(url=self.url, ...)
            logger.info(f"Connected to Substrate RPC at {self.url}")
            return
        except Exception as e:
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Connection attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {delay}s")
            if attempt < max_retries - 1:
                time.sleep(delay)
    raise ConnectionError(f"Failed to connect to {self.url} after {max_retries} attempts")
```

**Fix (P1-05)**: Align receipt field names — either rename the dataclass fields or update callers.

**Verification**:
- `python -m pytest tests/test_blockchain.py -v`

---

### Task 20: Fix circular import + 64-bit identity hash + IPFS timeout

**Findings**: P1-09 (MEDIUM), P1-12 (MEDIUM), P1-31 (MEDIUM)

**File 1**: `blockchain/validator_manager.py` (~L272)

**Problem (P1-09)**: `from .staking_interface import StakingInterface` inside method body — latent circular import.

**Fix**: Move import to top of file or restructure dependency.

**File 2**: `blockchain/identity_verifier.py` (~L125–128)

**Problem (P1-12)**: Uses first 8 bytes of SHA-256 → only 64-bit collision resistance.

**Fix**: Use full SHA-256 or at least 16 bytes (128-bit):
```python
hash_bytes = hashlib.sha256(belizeid.encode()).digest()
identity_id = int.from_bytes(hash_bytes[:16], byteorder='big')
```

**File 3**: `blockchain/genome_registry.py` (~L390–410)

**Problem (P1-31)**: `requests.post()` to IPFS with no timeout.

**Fix**: Add timeout:
```python
response = requests.post(
    f"{self.ipfs_gateway}/api/v0/add",
    files={"file": data},
    timeout=30,  # 30 second timeout
)
```

**Verification**:
- `python -m pytest tests/test_blockchain.py -v`

---

### Task 21: Fix security `__init__.py` exports + DP optimizer wrapper

**Findings**: P2-18 (MEDIUM), P2-03 (MEDIUM), P2-04 (MEDIUM)

**File 1**: `security/__init__.py`

**Problem (P2-18)**: `DPInferenceGuard` and `PrivacyAccountant` not exported.

**Fix**: Add to `__all__` and imports:
```python
from .dp_inference import DPInferenceGuard
from .differential_privacy import PrivacyAccountant

__all__ = [
    "DifferentialPrivacy", "PrivacyBudget", "PrivacyAccountant",
    "SecureAggregator", "ByzantineDetector", "AggregationMethod",
    "DPInferenceGuard",
]
```

**File 2**: `security/differential_privacy.py` (~L377–409)

**Problem (P2-03/P2-04)**: `create_dp_optimizer()` returns separate objects — easy to misuse. Simplified accounting.

**Fix**: Create a wrapper class that encapsulates the correct call order:
```python
class DPOptimizer:
    """Wraps optimizer + DP to enforce correct call sequence."""
    def __init__(self, optimizer, dp: DifferentialPrivacy):
        self.optimizer = optimizer
        self.dp = dp
    
    def step(self, model):
        self.dp.clip_gradients(model)
        self.dp.add_noise(model)
        self.optimizer.step()
        self.dp.update_privacy_budget()
```

**Verification**:
- `python -m pytest tests/test_differential_privacy.py -v`
- Manual: `from security import DPInferenceGuard` succeeds

---

### Task 22: Fix float scaling precision + legacy key gen fields

**Findings**: P2-11 (MEDIUM), P2-12 (MEDIUM)

**File**: `security/secure_aggregation.py` (~L200, ~L346–358)

**Problem (P2-11)**: `scale: int = 1000` — only 3 decimal places of gradient precision.

**Fix**: Increase scale to preserve gradient precision:
```python
scale: int = 1_000_000  # 6 decimal places — sufficient for most gradient values
```

**Problem (P2-12)**: Key generation uses `secrets.randbelow(2**16)` for legacy fields (16-bit keys — trivially brutable).

**Fix**: Increase to cryptographically appropriate key size:
```python
public_key=secrets.randbelow(2**256)
private_key=secrets.randbelow(2**256)
```
Or remove legacy fields entirely if they're unused.

**Verification**:
- `python -m pytest tests/ -v`

---

### Task 23: Fix `NawalConfig` dual-name conflict

**Findings**: P3-03 (MEDIUM)

**Files**: `architecture/config.py` and `config.py`

**Problem**: Two classes named `NawalConfig` — one is a dataclass (architecture params), one is Pydantic BaseModel (system config). Easy to import wrong one.

**Fix**: Rename the architecture config:
```python
# architecture/config.py
@dataclass
class NawalModelConfig:  # Was: NawalConfig
    """Transformer model architecture configuration."""
    d_model: int = 768
    ...
```

Update all imports:
- `architecture/transformer.py`
- `genome/nawal_adapter.py`
- `genome/model_builder.py`
- Any test files

**Verification**:
- `python -m pytest tests/ -v`
- `grep -r "from architecture.config import NawalConfig" .` — should be 0 results

---

### Task 24: Fix Laplace → Gaussian DP mechanism alignment

**Findings**: P4-07 (MEDIUM)

**File**: `client/train.py` (~L284–295)

**Problem**: Uses Laplace noise with L-infinity sensitivity:
```python
sensitivity = np.max(np.abs(param))
noise = np.random.laplace(0, noise_scale, param.shape)
```
Standard DP-SGD uses Gaussian mechanism with L2 sensitivity. This misalignment means epsilon accounting is invalid.

**Fix**: Switch to Gaussian mechanism aligned with `security/differential_privacy.py`:
```python
sensitivity = self.max_grad_norm  # Use clipping bound as sensitivity
noise_scale = sensitivity * self.noise_multiplier
noise = np.random.normal(0, noise_scale, param.shape)
```

Or use the existing `DifferentialPrivacy.add_noise()` from the security module instead of reimplementing.

**Verification**:
- `python -m pytest tests/test_differential_privacy.py tests/test_training.py -v`

---

## Phase G — LOW Priority Cleanup (Tasks 25–28)

> **Goal**: Fix deprecation warnings, dead code, and quality issues  
> **Estimated effort**: 2–3 days  

### Task 25: Replace all `datetime.utcnow()` with `datetime.now(timezone.utc)`

**Findings**: P1-10, P1-14, P3-13, P3-25, P3-30, P4-12, P5-19 and additional occurrences

**Files** (all occurrences):
- `blockchain/identity_verifier.py` (~L73, ~L78, ~L84)
- `blockchain/validator_manager.py` (~L192 — `datetime.fromtimestamp()` without tz)
- `genome/encoding.py` (~L218)
- `genome/fitness.py` (~L63)
- `hybrid/router.py` (~L107)
- `hybrid/engine.py` (~L186, ~L228)
- `hybrid/confidence.py`
- `api_server.py` (~L475, ~L510, ~L550, ~L600)
- `api/inference_server.py` (~L149, ~L199, ~L206)
- `client/model.py`

**Fix pattern**:
```python
# Before:
from datetime import datetime
datetime.utcnow()

# After:
from datetime import datetime, timezone
datetime.now(timezone.utc)

# For datetime.fromtimestamp:
datetime.fromtimestamp(ts, tz=timezone.utc)
```

**Verification**:
- `grep -rn "utcnow\|fromtimestamp(" --include="*.py" . | grep -v "timezone"` — should be 0 results
- `python -m pytest tests/ -v`

---

### Task 26: Fix logging inconsistencies

**Findings**: P3-02, P5-22 and stdlib logging throughout

**Files using `print()` → convert to `logger`:
- `integration/oracle_pipeline.py` (~L158, 186, 224, 362, 470, 480, 495, 540)

**Files using stdlib `logging` → convert to `loguru`:
- `architecture/transformer.py` (~L14)
- `blockchain/identity_verifier.py`
- `genome/fitness.py`
- `genome/nawal_adapter.py`

**Fix pattern**:
```python
# Before:
import logging
logger = logging.getLogger(__name__)

# After:
from loguru import logger

# Before:
print(f"Error: {e}")

# After:
logger.error(f"Error: {e}")
```

**Verification**:
- `grep -rn "^import logging$\|^from logging " --include="*.py" .` — confirm expected reduction
- `grep -rn "^\s*print(" --include="*.py" . | grep -v test | grep -v __pycache__` — confirm reduction
- `python -m pytest tests/ -v`

---

### Task 27: Dead code cleanup

**Findings**: P3-22, P3-05, P3-11

**File 1**: `genome/model_builder_stub.py` — entire file is dead code (all methods raise `NotImplementedError`).

**Fix**: Delete the file and remove any imports.

**File 2**: `genome/dna.py` — `DNA` backward compat wrapper adds maintenance surface (P3-05).

**Fix**: Add deprecation warning to `DNA` class:
```python
import warnings

class DNA:
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "DNA is deprecated, use Genome directly",
            DeprecationWarning,
            stacklevel=2,
        )
        ...
```

**File 3**: `genome/operators.py` (~L640) — dead variable `other` in `_hyperparameter_crossover()` (P3-11).

**Fix**: Remove the unused variable.

**Verification**:
- `python -m pytest tests/ -v`
- Confirm `model_builder_stub.py` is gone

---

### Task 28: Remaining LOW items batch

**Findings**: Multiple LOW severity items

This task bundles remaining LOW-severity findings that are individually small:

| Finding | File | Fix |
|---------|------|-----|
| P1-06: Binary honesty score (0/100) | `staking_interface.py:~472` | Scale to continuous 0–100 based on compliance history |
| P1-07: No SS58 validation | `staking_connector.py` | Add `ss58_decode()` validation on `account_id` |
| P1-27: `dict[str, any]` typo | `rewards.py:~379` | Change to `dict[str, Any]` |
| P1-33: Mock SRS hardcoded | `community_connector.py:~144` | Add comment documenting mock nature |
| P1-34: `str(metadata)` | `community_connector.py:~224` | Use `json.dumps(metadata)` |
| P2-07: Asymmetric reputation recovery | `byzantine_detection.py:~80` | Adjust recovery rate closer to decay rate |
| P2-13: SecureChannel placeholder | `secure_aggregation.py:~640` | Either implement or remove + document |
| P3-09: SSM creates LINEAR | `operators.py:~320` | Change to `LayerType.STATE_SPACE_MODEL` |
| P3-10: mutate_add_attention generic | `operators.py:~340` | Add attention-specific config (n_heads, etc.) |
| P3-12: No resource limits on hidden_size | `operators.py` | Add max bounds for hyperparameter mutations |
| P4-13: GENERAL→AgriTech fallback | `domain_models.py` | Create a `GeneralModel` or use `BelizeChainLLM` directly |
| P4-14: collate_fn empty dict | `data_loader.py:~393` | Return `None` and check in caller |
| P4-15: Hardcoded localhost:8080 | `train.py:~305` | Read from env/config |
| P4-18: hasattr lazy init | `metrics_tracker.py` | Initialize in `__init__` |
| P5-17: Sequential batch inference | `inference_server.py` | Add TODO comment or implement actual batching |
| P5-18: word count ≠ token count | `inference_server.py` | Use tokenizer to count tokens |
| P5-19: Unix timestamp format | `api_server.py` | Use ISO format consistently |
| P5-20: from_env parsing | `config.py` | Fix scientific notation and negative number parsing |
| P5-21: Kinich cache unbounded | `kinich_connector.py` | Add LRU eviction |
| P5-23: Dual SubstrateInterface | `oracle_pipeline.py` | Share connection between fetcher and submitter |
| P1-02: Debug storage logging | `substrate_client.py` | Ensure it's at DEBUG level only |
| P1-03: subscribe_events blocks | `substrate_client.py` | Add async wrapper or threading |
| P4-16: Lenient Byzantine threshold | `participant_manager.py` | Increase penalties |
| P4-17: In-memory rewards | `participant_manager.py` | Document limitation |
| P4-19: ByzantineRobust FedAvg fallback | `aggregator.py` | Log warning on fallback, add minimum participant check |

**Verification**:
- `python -m pytest tests/ -v`
- `grep -rn "dict\[str, any\]" --include="*.py" .` — should be 0 results

---

## Phase H — Validation (Task 29)

> **Goal**: Full validation of all remediation  
> **Estimated effort**: 2–4 hours  

### Task 29: Final validation

1. **Full test suite**: `python -m pytest tests/ -v --tb=short`
2. **Error check**: Verify no new compile/lint errors
3. **Search for regressions**:
   - `grep -rn "utcnow()" --include="*.py" . | grep -v test` — should be 0
   - `grep -rn "trust_remote_code" --include="*.py" .` — should be 0
   - `grep -rn "torch\.load(" --include="*.py" . | grep -v "weights_only"` — should be 0
   - `grep -rn "//Alice" --include="*.py" .` — should be 0
4. **Import validation**:
   ```python
   from security import DPInferenceGuard, PrivacyAccountant
   from security import DifferentialPrivacy, SecureAggregator
   from genome.model_builder import GenomeModel, ModelBuilder
   from genome.nawal_adapter import GenomeToNawalAdapter
   from blockchain.staking_connector import StakingConnector
   ```
5. **Update audit document**: Add remediation status to `docs/audits/2026-02-28-catastrophic-audit.md`
6. **Coverage check**: `python -m pytest tests/ --cov --cov-report=term-missing`

---

## Execution Order

| Order | Phase | Tasks | Dependency | Est. Time |
|-------|-------|-------|------------|-----------|
| 1 | **A** | 1–3 | None | 2–3 hours |
| 2 | **B** | 4–9 | None (can start with A) | 3–4 days |
| 3 | **C** | 10–13 | After A (needs fixed adapters) | 1–2 days |
| 4 | **D** | 14–15 | After A (needs fixed Pydantic) | 4–6 hours |
| 5 | **E** | 16–18 | After B (needs secure agg refactor) | 1 day |
| 6 | **F** | 19–24 | After D (needs fixed genome types) | 2–3 days |
| 7 | **G** | 25–28 | After F (global cleanup) | 2–3 days |
| 8 | **H** | 29 | After all phases | 2–4 hours |

**Total estimated effort**: ~10–15 days of focused work

---

## Risk Notes

1. **Task 4 (Secure aggregation redesign)** is the highest-risk change — requires careful testing with multi-client scenarios
2. **Task 23 (NawalConfig rename)** will touch many files — high regression risk, do atomically
3. **Task 25 (datetime)** is mechanical but touches ~15+ files — use grep + sed for consistency
4. **Tasks in Phase G** are individually low-risk but collectively touch many files — commit frequently
5. The intermittent test failure in `test_integrated_poisoning_detection_clean_data` should be investigated during Phase H to determine if it's a test isolation issue

---

**Document ready for execution. Begin with Phase A, Task 1.**
