"""
Tests for monitoring/ — Priority 2 operational layer.

Covers:
  - monitoring/metrics.py   → MetricType, Metric, MetricsCollector
  - monitoring/metrics_collector.py → InferenceMetricsCollector
"""

from __future__ import annotations

import csv
import time
from datetime import datetime, timedelta

import pytest

from monitoring.metrics import Metric, MetricsCollector, MetricType
from monitoring.metrics_collector import InferenceMetricsCollector

# ===========================================================================
# MetricType enum
# ===========================================================================


class TestMetricTypeEnum:

    def test_training_loss_value(self):
        assert MetricType.TRAINING_LOSS.value == "training_loss"

    def test_fitness_score_value(self):
        assert MetricType.FITNESS_SCORE.value == "fitness_score"

    def test_cpu_usage_value(self):
        assert MetricType.CPU_USAGE.value == "cpu_usage"

    def test_all_system_metrics_present(self):
        system = {
            MetricType.CPU_USAGE,
            MetricType.MEMORY_USAGE,
            MetricType.DISK_USAGE,
            MetricType.NETWORK_SENT,
            MetricType.NETWORK_RECV,
        }
        for mt in system:
            assert mt in MetricType


# ===========================================================================
# Metric dataclass
# ===========================================================================


class TestMetricDataclass:

    def test_create_with_defaults(self):
        m = Metric(name="test", value=1.0, timestamp=datetime.now())
        assert m.name == "test"
        assert m.value == 1.0
        assert m.labels == {}
        assert m.metadata == {}

    def test_create_with_labels(self):
        m = Metric(
            name="loss", value=0.5, timestamp=datetime.now(), labels={"epoch": "3"}
        )
        assert m.labels == {"epoch": "3"}

    def test_create_with_metadata(self):
        m = Metric(
            name="loss", value=0.5, timestamp=datetime.now(), metadata={"run_id": "abc"}
        )
        assert m.metadata["run_id"] == "abc"


# ===========================================================================
# MetricsCollector — init & basics
# ===========================================================================


@pytest.fixture
def collector() -> MetricsCollector:
    return MetricsCollector(buffer_size=100)


class TestMetricsCollectorInit:

    def test_default_init(self):
        mc = MetricsCollector()
        assert mc.buffer_size == 10000
        assert mc.metrics == []
        assert mc.history == {}

    def test_custom_buffer_size(self):
        mc = MetricsCollector(buffer_size=50)
        assert mc.buffer_size == 50


# ===========================================================================
# MetricsCollector — recording
# ===========================================================================


class TestMetricsCollectorRecord:

    def test_record_adds_to_metrics(self, collector):
        collector.record(MetricType.TRAINING_LOSS, 0.3)
        assert len(collector.metrics) == 1
        assert collector.metrics[0].value == 0.3
        assert collector.metrics[0].name == MetricType.TRAINING_LOSS.value

    def test_record_adds_to_history(self, collector):
        collector.record(MetricType.TRAINING_LOSS, 0.5)
        assert MetricType.TRAINING_LOSS.value in collector.history
        assert collector.history[MetricType.TRAINING_LOSS.value] == [0.5]

    def test_record_multiple_appends_history(self, collector):
        for v in [0.9, 0.8, 0.7]:
            collector.record(MetricType.TRAINING_LOSS, v)
        assert collector.history[MetricType.TRAINING_LOSS.value] == [0.9, 0.8, 0.7]

    def test_record_with_labels(self, collector):
        collector.record(MetricType.TRAINING_LOSS, 0.3, labels={"epoch": "1"})
        assert collector.metrics[0].labels == {"epoch": "1"}

    def test_record_with_metadata(self, collector):
        collector.record(MetricType.CPU_USAGE, 42.0, metadata={"host": "node1"})
        assert collector.metrics[0].metadata == {"host": "node1"}

    def test_buffer_trim_at_limit(self):
        mc = MetricsCollector(buffer_size=5)
        for i in range(10):
            mc.record(MetricType.CPU_USAGE, float(i))
        assert len(mc.metrics) == 5
        # Should keep the last 5
        assert mc.metrics[0].value == 5.0
        assert mc.metrics[-1].value == 9.0

    def test_thread_safety_no_crash(self, collector):
        """Multiple threads recording concurrently should not raise."""
        import threading

        def _record(n):
            for i in range(20):
                collector.record(MetricType.BATCH_TIME, float(n + i))

        threads = [threading.Thread(target=_record, args=(i * 100,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(collector.metrics) == 100  # 5 threads × 20 records


# ===========================================================================
# MetricsCollector — domain-specific helpers
# ===========================================================================


class TestMetricsCollectorDomainHelpers:

    def test_record_training_epoch_stores_4_metric_types(self, collector):
        before = len(collector.metrics)
        collector.record_training_epoch(
            epoch=1, train_loss=0.5, train_acc=0.8, val_loss=0.55, val_acc=0.78
        )
        # Should add 4 metrics (no epoch timing since start_epoch was not called)
        assert len(collector.metrics) - before == 4

    def test_record_training_epoch_includes_timing_when_started(self, collector):
        collector.start_epoch()
        time.sleep(0.01)
        before = len(collector.metrics)
        collector.record_training_epoch(
            epoch=2, train_loss=0.4, train_acc=0.85, val_loss=0.45, val_acc=0.83
        )
        assert len(collector.metrics) - before == 5  # 4 + epoch_time
        # _epoch_start should be cleared
        assert collector._epoch_start is None

    def test_end_batch_records_batch_time(self, collector):
        collector.start_batch()
        time.sleep(0.01)
        collector.end_batch()
        assert MetricType.BATCH_TIME.value in collector.history
        assert collector.history[MetricType.BATCH_TIME.value][0] > 0

    def test_end_batch_clears_start(self, collector):
        collector.start_batch()
        collector.end_batch()
        assert collector._batch_start is None

    def test_end_batch_without_start_is_noop(self, collector):
        collector.end_batch()  # should not raise
        assert MetricType.BATCH_TIME.value not in collector.history

    def test_record_fitness_stores_best_and_average(self, collector):
        collector.record_fitness(generation=5, best=0.92, average=0.75)
        assert MetricType.BEST_FITNESS.value in collector.history
        assert MetricType.AVERAGE_FITNESS.value in collector.history
        assert collector.history[MetricType.BEST_FITNESS.value] == [0.92]

    def test_record_fitness_with_individual(self, collector):
        collector.record_fitness(generation=1, best=0.9, average=0.7, individual=0.85)
        assert MetricType.FITNESS_SCORE.value in collector.history

    def test_record_federated_round_stores_3_metrics(self, collector):
        collector.start_federated_round()
        time.sleep(0.01)
        before = len(collector.metrics)
        collector.record_federated_round(
            round_num=3, num_clients=10, aggregation_time=0.5, communication_cost=1024.0
        )
        # CLIENT_COUNT, AGGREGATION_TIME, COMMUNICATION_COST, + ROUND_TIME = 4
        assert len(collector.metrics) - before == 4

    def test_record_federated_round_clears_round_start(self, collector):
        collector.start_federated_round()
        collector.record_federated_round(
            round_num=1, num_clients=5, aggregation_time=0.1, communication_cost=512.0
        )
        assert collector._round_start is None

    def test_record_blockchain_success(self, collector):
        collector.record_blockchain_transaction(success=True, tx_time=0.2)
        names = [m.name for m in collector.metrics]
        assert MetricType.TRANSACTION_SUCCESS.value in names
        assert MetricType.TRANSACTION_TIME.value in names

    def test_record_blockchain_failure(self, collector):
        collector.record_blockchain_transaction(success=False, tx_time=0.1)
        names = [m.name for m in collector.metrics]
        assert MetricType.TRANSACTION_FAILURE.value in names

    def test_record_blockchain_with_block_and_finalization(self, collector):
        before = len(collector.metrics)
        collector.record_blockchain_transaction(
            success=True, tx_time=0.3, block_number=12345, finalization_time=2.5
        )
        # success + tx_time + block_number + finalization_time = 4
        assert len(collector.metrics) - before == 4


# ===========================================================================
# MetricsCollector — retrieval
# ===========================================================================


class TestMetricsCollectorRetrieval:

    def test_get_metrics_returns_all(self, collector):
        collector.record(MetricType.TRAINING_LOSS, 0.5)
        collector.record(MetricType.CPU_USAGE, 30.0)
        assert len(collector.get_metrics()) == 2

    def test_get_metrics_filtered_by_type(self, collector):
        collector.record(MetricType.TRAINING_LOSS, 0.5)
        collector.record(MetricType.CPU_USAGE, 30.0)
        result = collector.get_metrics(MetricType.TRAINING_LOSS)
        assert len(result) == 1
        assert result[0].name == MetricType.TRAINING_LOSS.value

    def test_get_metrics_filtered_by_since(self, collector):
        # Record a metric "5 minutes ago"
        old_metric = Metric(
            name=MetricType.TRAINING_LOSS.value,
            value=0.9,
            timestamp=datetime.now() - timedelta(minutes=10),
        )
        collector.metrics.append(old_metric)
        # Record a recent metric now
        collector.record(MetricType.TRAINING_LOSS, 0.5)

        since = datetime.now() - timedelta(minutes=1)
        recent = collector.get_metrics(MetricType.TRAINING_LOSS, since=since)
        assert len(recent) == 1
        assert recent[0].value == 0.5

    def test_get_latest_returns_most_recent(self, collector):
        collector.record(MetricType.TRAINING_LOSS, 0.9)
        collector.record(MetricType.TRAINING_LOSS, 0.5)
        latest = collector.get_latest(MetricType.TRAINING_LOSS)
        assert latest.value == 0.5

    def test_get_latest_empty_returns_none(self, collector):
        assert collector.get_latest(MetricType.TRAINING_LOSS) is None

    def test_get_average_over_window(self, collector):
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            collector.record(MetricType.TRAINING_LOSS, v)
        avg = collector.get_average(MetricType.TRAINING_LOSS, window=3)
        assert abs(avg - 4.0) < 1e-9  # last 3: 3,4,5 → avg 4

    def test_get_average_no_history_returns_none(self, collector):
        assert collector.get_average(MetricType.CPU_USAGE, window=5) is None

    def test_get_summary_contains_required_keys(self, collector):
        summary = collector.get_summary()
        for key in ["uptime_seconds", "total_metrics", "metric_types", "buffer_usage"]:
            assert key in summary

    def test_get_summary_reflects_recorded_data(self, collector):
        collector.record(MetricType.TRAINING_LOSS, 0.4)
        summary = collector.get_summary()
        assert summary["total_metrics"] == 1


# ===========================================================================
# MetricsCollector — clear & export
# ===========================================================================


class TestMetricsCollectorClearExport:

    def test_clear_empties_metrics_and_history(self, collector):
        collector.record(MetricType.TRAINING_LOSS, 0.5)
        collector.record(MetricType.CPU_USAGE, 42.0)
        collector.clear()
        assert collector.metrics == []
        assert collector.history == {}

    def test_export_csv_creates_file(self, collector, tmp_path):
        collector.record(
            MetricType.TRAINING_LOSS,
            0.5,
            labels={"epoch": "1"},
            metadata={"note": "test"},
        )
        csv_path = tmp_path / "metrics.csv"
        collector.export_csv(csv_path)
        assert csv_path.exists()

    def test_export_csv_contains_header_and_row(self, collector, tmp_path):
        collector.record(MetricType.CPU_USAGE, 55.0)
        csv_path = tmp_path / "metrics.csv"
        collector.export_csv(csv_path)
        rows = list(csv.reader(open(csv_path)))
        assert rows[0] == ["timestamp", "name", "value", "labels", "metadata"]
        assert len(rows) == 2  # header + 1 data row
        assert "cpu_usage" in rows[1][1]
        assert "55.0" in rows[1][2]

    def test_export_csv_multiple_rows(self, collector, tmp_path):
        for v in [1.0, 2.0, 3.0]:
            collector.record(MetricType.BATCH_TIME, v)
        csv_path = tmp_path / "multi.csv"
        collector.export_csv(csv_path)
        rows = list(csv.reader(open(csv_path)))
        assert len(rows) == 4  # header + 3 data rows


# ===========================================================================
# MetricsCollector — system metrics (psutil integration)
# ===========================================================================


class TestMetricsCollectorSystemMetrics:

    def test_collect_system_metrics_records_cpu(self, collector):
        collector.collect_system_metrics()
        assert MetricType.CPU_USAGE.value in collector.history

    def test_collect_system_metrics_records_memory(self, collector):
        collector.collect_system_metrics()
        assert MetricType.MEMORY_USAGE.value in collector.history

    def test_collect_system_metrics_records_disk(self, collector):
        collector.collect_system_metrics()
        assert MetricType.DISK_USAGE.value in collector.history

    def test_collect_system_metrics_network_counters(self, collector):
        collector.collect_system_metrics()
        assert MetricType.NETWORK_SENT.value in collector.history
        assert MetricType.NETWORK_RECV.value in collector.history

    def test_collect_system_metrics_disabled_is_noop(self, collector):
        collector._system_metrics_enabled = False
        collector.collect_system_metrics()
        assert MetricType.CPU_USAGE.value not in collector.history

    def test_cpu_usage_value_range(self, collector):
        collector.collect_system_metrics()
        val = collector.history[MetricType.CPU_USAGE.value][0]
        assert 0.0 <= val <= 100.0


# ===========================================================================
# InferenceMetricsCollector
# ===========================================================================


class TestInferenceMetricsCollector:

    def test_init_creates_prometheus_registry(self):
        mc = InferenceMetricsCollector()
        assert mc.registry is not None

    def test_init_creates_all_counters(self):
        mc = InferenceMetricsCollector()
        assert mc.requests_total is not None
        assert mc.inference_duration is not None
        assert mc.tokens_generated is not None
        assert mc.model_info is not None

    @pytest.mark.asyncio
    async def test_log_inference_completes(self):
        mc = InferenceMetricsCollector()
        # Should not raise
        await mc.log_inference(
            user_id="user1", prompt_length=50, output_length=100, inference_time=500.0
        )

    @pytest.mark.asyncio
    async def test_log_inference_increments_request_counter(self):
        mc = InferenceMetricsCollector()
        await mc.log_inference(
            user_id="userA", prompt_length=10, output_length=20, inference_time=100.0
        )
        # Verify Prometheus metrics exported contain counter data
        exported = await mc.export_prometheus()
        assert b"nawal_inference_requests_total" in exported

    @pytest.mark.asyncio
    async def test_export_prometheus_returns_bytes(self):
        mc = InferenceMetricsCollector()
        result = await mc.export_prometheus()
        assert isinstance(result, bytes)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_export_prometheus_contains_token_counter(self):
        mc = InferenceMetricsCollector()
        await mc.log_inference("u1", 10, 50, 200.0)
        exported = await mc.export_prometheus()
        assert b"nawal_tokens_generated_total" in exported

    def test_two_collectors_have_independent_registries(self):
        """Each InferenceMetricsCollector has its own CollectorRegistry."""
        mc1 = InferenceMetricsCollector()
        mc2 = InferenceMetricsCollector()
        assert mc1.registry is not mc2.registry
