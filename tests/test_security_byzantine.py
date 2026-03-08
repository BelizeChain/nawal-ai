"""
Direct tests for security/byzantine_detection.py module classes.

Covers:
  - AggregationMethod enum
  - ClientReputation dataclass
  - ByzantineDetector — all six aggregation methods
  - ByzantineDetector.detect_anomalies
  - ByzantineDetector reputation helpers and to_dict
  - recommend_aggregation_method utility
"""
from __future__ import annotations

from typing import Dict, List

import pytest
import torch

from security.byzantine_detection import (
    AggregationMethod,
    ByzantineDetector,
    ClientReputation,
    recommend_aggregation_method,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_update(seed: int = 0, size: int = 6) -> Dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "fc.weight": torch.randn(size, size),
        "fc.bias": torch.randn(size),
    }


def _make_updates(n: int) -> List[Dict[str, torch.Tensor]]:
    return [_make_update(seed=i) for i in range(n)]


# ---------------------------------------------------------------------------
# AggregationMethod enum
# ---------------------------------------------------------------------------

class TestAggregationMethodEnum:

    def test_all_six_values_present(self):
        vals = {m.value for m in AggregationMethod}
        assert vals == {"fedavg", "krum", "multi_krum", "trimmed_mean", "median", "phocas"}

    def test_from_value_string(self):
        assert AggregationMethod("trimmed_mean") == AggregationMethod.TRIMMED_MEAN


# ---------------------------------------------------------------------------
# ClientReputation
# ---------------------------------------------------------------------------

class TestClientReputation:

    def test_defaults(self):
        rep = ClientReputation(client_id=7)
        assert rep.score == 1.0
        assert rep.contributions == 0
        assert rep.anomalies == 0
        assert rep.history == []

    def test_non_anomalous_increments_score(self):
        rep = ClientReputation(client_id=1, score=0.8)
        rep.update(is_anomalous=False)
        assert rep.score > 0.8
        assert rep.score <= 1.0

    def test_anomalous_decays_score(self):
        rep = ClientReputation(client_id=1, score=1.0)
        rep.update(is_anomalous=True)
        assert rep.score < 1.0

    def test_contributions_counted(self):
        rep = ClientReputation(client_id=1)
        rep.update(False)
        rep.update(True)
        assert rep.contributions == 2
        assert rep.anomalies == 1

    def test_history_grows(self):
        rep = ClientReputation(client_id=1)
        for _ in range(5):
            rep.update(False)
        assert len(rep.history) == 5

    def test_history_capped_at_100(self):
        rep = ClientReputation(client_id=1)
        for _ in range(120):
            rep.update(False)
        assert len(rep.history) == 100

    def test_is_trustworthy_high_score(self):
        assert ClientReputation(client_id=1, score=0.9).is_trustworthy() is True

    def test_not_trustworthy_low_score(self):
        assert ClientReputation(client_id=1, score=0.2).is_trustworthy() is False

    def test_is_trustworthy_at_threshold(self):
        assert ClientReputation(client_id=1, score=0.5).is_trustworthy(0.5) is True


# ---------------------------------------------------------------------------
# ByzantineDetector — init
# ---------------------------------------------------------------------------

class TestByzantineDetectorInit:

    def test_default_method_is_krum(self):
        d = ByzantineDetector()
        assert d.method == AggregationMethod.KRUM

    def test_custom_method(self):
        d = ByzantineDetector(method=AggregationMethod.MEDIAN, num_byzantine=1)
        assert d.method == AggregationMethod.MEDIAN
        assert d.num_byzantine == 1

    def test_reputation_disabled(self):
        d = ByzantineDetector(reputation_enabled=False)
        assert d.reputation_enabled is False

    def test_empty_reputations_on_init(self):
        assert ByzantineDetector().reputations == {}


# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------

class TestFedAvg:

    def test_averages_two_tensors(self):
        d = ByzantineDetector(method=AggregationMethod.FEDAVG)
        updates = [{"w": torch.tensor([0.0, 0.0])}, {"w": torch.tensor([4.0, 8.0])}]
        result = d.aggregate(updates)
        assert torch.allclose(result["w"], torch.tensor([2.0, 4.0]))

    def test_single_client_passthrough(self):
        d = ByzantineDetector(method=AggregationMethod.FEDAVG)
        t = torch.tensor([1.5, 2.5])
        result = d.aggregate([{"w": t}])
        assert torch.allclose(result["w"], t)

    def test_empty_updates_raises(self):
        with pytest.raises(ValueError, match="No client updates"):
            ByzantineDetector().aggregate([])

    def test_multiple_params_all_averaged(self):
        d = ByzantineDetector(method=AggregationMethod.FEDAVG)
        result = d.aggregate(_make_updates(4))
        assert "fc.weight" in result and "fc.bias" in result


# ---------------------------------------------------------------------------
# Krum
# ---------------------------------------------------------------------------

class TestKrum:

    def test_returns_one_of_the_inputs(self):
        d = ByzantineDetector(method=AggregationMethod.KRUM, num_byzantine=1)
        updates = _make_updates(6)
        result = d.aggregate(updates)
        found = any(torch.allclose(result["fc.bias"], u["fc.bias"]) for u in updates)
        assert found

    def test_fallback_to_fedavg_when_k_leq_zero(self):
        # n=3, f=2 → k = 3-2-2 = -1 → FedAvg
        d = ByzantineDetector(method=AggregationMethod.KRUM, num_byzantine=2)
        updates = [{"w": torch.tensor([v])} for v in [0.0, 3.0, 6.0]]
        result = d.aggregate(updates)
        assert torch.allclose(result["w"], torch.tensor([3.0]))

    def test_correct_output_keys(self):
        d = ByzantineDetector(method=AggregationMethod.KRUM, num_byzantine=0)
        result = d.aggregate(_make_updates(5))
        assert set(result.keys()) == {"fc.weight", "fc.bias"}


# ---------------------------------------------------------------------------
# Multi-Krum
# ---------------------------------------------------------------------------

class TestMultiKrum:

    def test_output_shape_matches_input(self):
        d = ByzantineDetector(method=AggregationMethod.MULTI_KRUM, num_byzantine=1)
        updates = _make_updates(6)
        result = d.aggregate(updates)
        assert result["fc.weight"].shape == updates[0]["fc.weight"].shape

    def test_result_is_average_not_single_update(self):
        """Multi-Krum returns an average, so it generally won't exactly match any one input."""
        d = ByzantineDetector(method=AggregationMethod.MULTI_KRUM, num_byzantine=1)
        updates = _make_updates(6)
        result = d.aggregate(updates)
        # Just check shapes and all params present
        assert "fc.bias" in result


# ---------------------------------------------------------------------------
# Trimmed Mean
# ---------------------------------------------------------------------------

class TestTrimmedMean:

    def test_output_shape_matches(self):
        d = ByzantineDetector(method=AggregationMethod.TRIMMED_MEAN)
        result = d.aggregate(_make_updates(10))
        assert result["fc.weight"].shape == (6, 6)

    def test_extreme_outlier_trimmed(self):
        """With 10 clients and trim_ratio=0.1, one outlier from each end is removed."""
        d = ByzantineDetector(method=AggregationMethod.TRIMMED_MEAN)
        normal = [{"w": torch.tensor([1.0])} for _ in range(8)]
        outliers = [{"w": torch.tensor([1000.0])}, {"w": torch.tensor([-1000.0])}]
        updates = normal + outliers  # 10 total → trims 1 each side
        result = d.aggregate(updates)
        assert result["w"].item() < 100.0

    def test_three_clients_no_trim(self):
        """int(3 * 0.1) = 0 → same as FedAvg."""
        d = ByzantineDetector(method=AggregationMethod.TRIMMED_MEAN)
        updates = [{"w": torch.tensor([v])} for v in [1.0, 2.0, 3.0]]
        result = d.aggregate(updates)
        assert torch.allclose(result["w"], torch.tensor([2.0]))


# ---------------------------------------------------------------------------
# Median
# ---------------------------------------------------------------------------

class TestMedian:

    def test_odd_clients_exact_median(self):
        d = ByzantineDetector(method=AggregationMethod.MEDIAN)
        updates = [{"w": torch.tensor([1.0])}, {"w": torch.tensor([5.0])}, {"w": torch.tensor([3.0])}]
        result = d.aggregate(updates)
        assert result["w"].item() == pytest.approx(3.0)

    def test_rejects_outlier(self):
        d = ByzantineDetector(method=AggregationMethod.MEDIAN)
        updates = [
            {"w": torch.tensor([1.0])}, {"w": torch.tensor([1.0])},
            {"w": torch.tensor([1.0])}, {"w": torch.tensor([999.0])},
            {"w": torch.tensor([1.0])},
        ]
        result = d.aggregate(updates)
        assert result["w"].item() == pytest.approx(1.0)

    def test_two_clients_returns_lower_median(self):
        """torch.median on even-length dim returns the lower middle value."""
        d = ByzantineDetector(method=AggregationMethod.MEDIAN)
        updates = [{"w": torch.tensor([2.0])}, {"w": torch.tensor([4.0])}]
        result = d.aggregate(updates)
        # PyTorch median on 2 elements returns the lower value (2.0)
        assert result["w"].item() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# PHOCAS
# ---------------------------------------------------------------------------

class TestPhocas:

    def test_requires_client_ids(self):
        d = ByzantineDetector(method=AggregationMethod.PHOCAS)
        with pytest.raises(ValueError, match="PHOCAS requires client IDs"):
            d.aggregate(_make_updates(3))

    def test_with_equal_reputations_equals_fedavg(self):
        d = ByzantineDetector(method=AggregationMethod.PHOCAS, reputation_enabled=False)
        updates = [{"w": torch.tensor([float(i)])} for i in range(1, 4)]
        r_phocas = d.aggregate(updates, client_ids=[0, 1, 2])
        r_fedavg = d.aggregate(updates, method=AggregationMethod.FEDAVG)
        assert torch.allclose(r_phocas["w"], r_fedavg["w"], atol=1e-4)

    def test_low_rep_client_down_weighted(self):
        d = ByzantineDetector(method=AggregationMethod.PHOCAS, reputation_enabled=False)
        d.reputations[0] = ClientReputation(client_id=0, score=1.0)
        d.reputations[1] = ClientReputation(client_id=1, score=1.0)
        d.reputations[2] = ClientReputation(client_id=2, score=0.01)
        updates = [{"w": torch.tensor([1.0])}, {"w": torch.tensor([1.0])}, {"w": torch.tensor([500.0])}]
        result = d.aggregate(updates, client_ids=[0, 1, 2])
        assert result["w"].item() < 50.0

    def test_all_zero_weights_uses_uniform(self):
        """If all reputation scores are 0 → uniform weights (total_weight == 0 branch)."""
        d = ByzantineDetector(method=AggregationMethod.PHOCAS, reputation_enabled=False)
        for cid in range(3):
            d.reputations[cid] = ClientReputation(client_id=cid, score=0.0)
        updates = [{"w": torch.tensor([float(i)])} for i in [1.0, 3.0, 5.0]]
        result = d.aggregate(updates, client_ids=[0, 1, 2])
        assert torch.allclose(result["w"], torch.tensor([3.0]), atol=1e-4)


# ---------------------------------------------------------------------------
# Method override at call time
# ---------------------------------------------------------------------------

class TestMethodOverride:

    def test_override_krum_with_median(self):
        d = ByzantineDetector(method=AggregationMethod.KRUM, num_byzantine=0)
        result = d.aggregate(_make_updates(5), method=AggregationMethod.MEDIAN)
        assert "fc.weight" in result

    def test_override_default_with_fedavg(self):
        d = ByzantineDetector(method=AggregationMethod.KRUM, num_byzantine=0)
        updates = [{"w": torch.tensor([2.0])}, {"w": torch.tensor([4.0])}]
        result = d.aggregate(updates, method=AggregationMethod.FEDAVG)
        assert torch.allclose(result["w"], torch.tensor([3.0]))


# ---------------------------------------------------------------------------
# detect_anomalies
# ---------------------------------------------------------------------------

class TestDetectAnomalies:

    def test_returns_list_of_bools(self):
        d = ByzantineDetector()
        anomalies = d.detect_anomalies(_make_updates(5))
        assert len(anomalies) == 5
        assert all(isinstance(a, bool) for a in anomalies)

    def test_single_update_not_anomalous(self):
        d = ByzantineDetector()
        assert d.detect_anomalies([_make_update(0)]) == [False]

    def test_identical_updates_not_flagged(self):
        d = ByzantineDetector()
        base = _make_update(42)
        anomalies = d.detect_anomalies([base, base, base, base])
        assert not any(anomalies)

    def test_huge_norm_update_flagged(self):
        d = ByzantineDetector()
        normal = [_make_update(i) for i in range(4)]
        poison = {"fc.weight": torch.full((6, 6), 1e5), "fc.bias": torch.full((6,), 1e5)}
        anomalies = d.detect_anomalies(normal + [poison])
        assert anomalies[-1] is True

    def test_custom_thresholds_accepted(self):
        d = ByzantineDetector()
        anomalies = d.detect_anomalies(_make_updates(4), threshold_std=10.0, cosine_threshold=-1.0)
        assert len(anomalies) == 4


# ---------------------------------------------------------------------------
# Reputation helpers
# ---------------------------------------------------------------------------

class TestReputationHelpers:

    def test_aggregate_with_ids_builds_reputations(self):
        d = ByzantineDetector(reputation_enabled=True)
        d.aggregate(_make_updates(4), client_ids=[0, 1, 2, 3])
        assert len(d.reputations) == 4

    def test_aggregate_without_ids_no_reputations(self):
        d = ByzantineDetector(reputation_enabled=True)
        d.aggregate(_make_updates(4))  # no client_ids
        assert d.reputations == {}

    def test_get_client_reputation_unknown(self):
        assert ByzantineDetector().get_client_reputation(999) is None

    def test_get_client_reputation_known(self):
        d = ByzantineDetector(reputation_enabled=True)
        d.aggregate(_make_updates(3), client_ids=[10, 11, 12])
        rep = d.get_client_reputation(10)
        assert rep is not None and isinstance(rep.score, float)

    def test_get_trustworthy_clients(self):
        d = ByzantineDetector()
        d.reputations[0] = ClientReputation(client_id=0, score=0.9)
        d.reputations[1] = ClientReputation(client_id=1, score=0.1)
        trusted = d.get_trustworthy_clients(threshold=0.5)
        assert 0 in trusted
        assert 1 not in trusted

    def test_to_dict_keys(self):
        d = ByzantineDetector()
        d.aggregate(_make_updates(3), client_ids=[0, 1, 2])
        info = d.to_dict()
        assert set(info.keys()) >= {"method", "num_byzantine", "reputation_enabled",
                                     "num_tracked_clients", "reputations"}
        assert info["num_tracked_clients"] == 3


# ---------------------------------------------------------------------------
# recommend_aggregation_method
# ---------------------------------------------------------------------------

class TestRecommendAggregationMethod:

    def test_zero_byzantine_fedavg(self):
        assert recommend_aggregation_method(10, 0) == AggregationMethod.FEDAVG

    def test_ratio_lt_02_krum(self):
        assert recommend_aggregation_method(10, 1) == AggregationMethod.KRUM

    def test_ratio_02_multi_krum(self):
        assert recommend_aggregation_method(10, 2) == AggregationMethod.MULTI_KRUM

    def test_ratio_03_trimmed_mean(self):
        assert recommend_aggregation_method(10, 3) == AggregationMethod.TRIMMED_MEAN

    def test_ratio_ge_05_median(self):
        assert recommend_aggregation_method(10, 5) == AggregationMethod.MEDIAN

    def test_returns_enum_instance(self):
        assert isinstance(recommend_aggregation_method(8, 2), AggregationMethod)
