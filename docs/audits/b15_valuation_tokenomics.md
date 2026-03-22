# B15 — Valuation & Tokenomics Rewards Audit

**Auditor**: Copilot  
**Date**: 2025-07-15  
**Scope**: `valuation/`, `blockchain/rewards.py`, `blockchain/staking_connector.py`, `blockchain/payroll_connector.py`  
**Baseline**: 3852 passed, 2 skipped, 0 warnings  

---

## Architectural Note

The `valuation/` directory does **not** contain tokenomics reward logic. It implements an RLHF-style "Limbic System / Basal Ganglia" for scoring candidate responses using internal drives (safety, alignment, novelty, curiosity). The actual tokenomics — quality/timeliness/honesty scoring, DALLA rewards, and Planck denomination — lives in `blockchain/rewards.py` and `blockchain/staking_connector.py`. This audit covers both.

---

## Summary

| Check | Title | Verdict | Findings |
|-------|-------|---------|----------|
| C15.1 | Quality scoring weight consistency (40/30/30) | **PASS** | Weights correct, constants exported, sum verified |
| C15.2 | Accuracy score computation | **PASS** (info) | FitnessScores uses defensive clamping; batch-3 tests use wrong scale (0–1 vs 0–100) |
| C15.3 | Timeliness score computation | **PASS** | Validated in TrainingSubmission; clamped in calculate_overall |
| C15.4 | Token denomination & overflow | **FAIL** | Payroll uses 10^8 denomination vs rewards' 10^12; staking_connector has hardcoded `1e12` magic numbers |

---

## C15.1 — Quality Scoring Weight Consistency

**Verdict: PASS**

`blockchain/rewards.py` defines:
```python
QUALITY_WEIGHT = 0.40   # 40% - Model accuracy
TIMELINESS_WEIGHT = 0.30  # 30% - Training speed
HONESTY_WEIGHT = 0.30   # 30% - Privacy compliance
```

`FitnessScores.calculate_overall()` uses these constants:
```python
return QUALITY_WEIGHT * q + TIMELINESS_WEIGHT * t + HONESTY_WEIGHT * h
```

Weights sum to 1.0, are exported in `__all__`, and are already tested in `test_b10_blockchain_pouw.py` (tests: `test_weights_sum_to_one`, `test_quality_weight_is_040`, `test_timeliness_weight_is_030`, `test_honesty_weight_is_030`, `test_calculate_overall_formula`).

No findings.

---

## C15.2 — Accuracy Score Computation

**Verdict: PASS (with informational finding)**

`FitnessScores.calculate_overall()` defensively clamps all three inputs to [0, 100]:
```python
q = min(100.0, max(0.0, self.quality))
```

`TrainingSubmission.validate()` rejects scores outside [0, 100]. `FitnessScores.validate()` does the same.

**F15.2a [INFO] — Batch-3 tests use wrong scale for FitnessScores**

`tests/test_rewards_staking_connector.py` (batch-3 coverage tests) create `FitnessScores(quality=0.8, timeliness=0.7, honesty=0.9)` — using 0–1 scale instead of 0–100. The assertion `assert 0.0 <= overall <= 1.0` passes accidentally because sub-1.0 inputs produce sub-1.0 output. These tests don't catch real bugs (the B10 tests use the correct 0–100 scale). Not a code defect — only a test accuracy issue.

---

## C15.3 — Timeliness Score Computation

**Verdict: PASS**

Timeliness is an externally-provided score (0–100) submitted in `TrainingSubmission.timeliness_score`. The system validates the range, clamps in `calculate_overall()`, and applies `TIMELINESS_WEIGHT` (0.30). No server-side deadline enforcement or decay function exists — timeliness is measured by the training orchestrator and reported. This is consistent with the PoUW design where the blockchain verifies scores post-submission.

No findings.

---

## C15.4 — Token Denomination & Overflow

**Verdict: FAIL**

### F15.4a [HIGH] — Payroll connector uses different denomination from rewards

`blockchain/payroll_connector.py` operates on `1 DALLA = 10^8 Planck` (Satoshi-style), while `blockchain/rewards.py` uses:
```python
DALLA_DECIMALS = 12
PLANCK_PER_DALLA = 10 ** DALLA_DECIMALS  # 1 DALLA = 10^12 Planck
```

The payroll module is **internally consistent** with 10^8 denomination: all salary values, tax thresholds, mock paystubs, and example scripts use this convention. The two modules are currently isolated (no cross-boundary value passing), so there is no live bug today.

**Impact**: Any future code path that mixes payroll Planck values with rewards Planck values would be off by 10^4 (10,000×). This is architectural debt.

**Fix applied**: Replaced the hardcoded tax threshold `2600000000000` with `26_000 * PAYROLL_PLANCK_PER_DALLA` using a new module-level constant `PAYROLL_PLANCK_PER_DALLA = 10**8`. Added a comment documenting the denomination difference from `blockchain/rewards.py`. A comprehensive migration of all payroll values to 10^12 denomination is deferred as future work.

### F15.4b [MEDIUM] — Hardcoded `1e12` magic numbers in staking_connector

`blockchain/staking_connector.py` uses `1e12` instead of the `PLANCK_PER_DALLA` constant it already imports:

- Line 629: `reward_dalla=reward / 1e12`
- Line 669: `reward_dalla=reward_amount / 1e12`

If `DALLA_DECIMALS` ever changes, these hardcoded values would silently diverge.

**Fix**: Replace `1e12` with `PLANCK_PER_DALLA`.

### F15.4c [INFO] — Integer conversion is correct in rewards calculator

`blockchain/rewards.py` line 235:
```python
total_reward_planck = int(total_reward_dalla * PLANCK_PER_DALLA)
```

This correctly produces an integer Planck value. Maximum possible reward: `10 DALLA * 1.0 * 2.0 = 20 DALLA = 20 × 10^12 Planck` — well within int64 range. No overflow risk.

---

## Recommendations Implemented

| ID | Severity | Fix |
|----|----------|-----|
| F15.4a | HIGH | Define `PAYROLL_PLANCK_PER_DALLA = 10**8` constant, replace hardcoded threshold, document denomination difference |
| F15.4b | MEDIUM | Replace `1e12` magic numbers with `PLANCK_PER_DALLA` in staking_connector |
| F15.2a | INFO | Fix batch-3 FitnessScores test scale (0–1 → 0–100) |

---

## Test Impact

New tests added in `tests/test_b15_valuation_tokenomics.py`:
- Denomination consistency between rewards and payroll
- staking_connector uses PLANCK_PER_DALLA constant (not magic 1e12)
- FitnessScores clamping edge cases
- valuation/rewards layer isolation (no tokenomics leakage)
- Reward max bound within int64 range
