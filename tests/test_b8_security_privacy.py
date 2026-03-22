"""
B8 · Security & Privacy — Audit Tests

Covers C8.1–C8.7:
  C8.1  DP wrapper correctness (budget enforcement, noise, configurable params)
  C8.2  Byzantine aggregation defences (prod config, method correctness)
  C8.3  ZK-proof payroll generation (salted merkle, hashed totals, per-entry)
  C8.4  ZK-proof verification (dev-mode structural checks, prod raises)
  C8.5  KYC/AML compliance (ComplianceFilter, ComplianceDataFilter, verifier factory)
  C8.6  Content filtering scope (OutputFilter, ComplianceFilter coverage)
  C8.7  Azure secret handling (no hardcoded secrets, header-only API key)
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
import torch.nn as nn
import yaml

from security.differential_privacy import (
    DPOptimizer,
    DifferentialPrivacy,
    PrivacyAccountant,
    PrivacyBudget,
    PrivacyBudgetExhaustedError,
    create_dp_optimizer,
)
from security.byzantine_detection import (
    AggregationMethod,
    ByzantineDetector,
    ClientReputation,
)
from blockchain.payroll_connector import (
    EmployeeType,
    PayrollEntry,
    PayrollProof,
)
from client.nawal import ComplianceFilter
from maintenance.output_filter import OutputFilter
from maintenance.interfaces import RiskLevel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        return self.fc(x)


def _model_with_grads(scale: float = 1.0) -> _TinyModel:
    model = _TinyModel()
    x = torch.randn(2, 4) * scale
    loss = model(x).sum()
    loss.backward()
    return model


def _make_payroll_entry(**overrides) -> PayrollEntry:
    defaults = dict(
        employee_id="BZ-12345-0001",
        employee_name_hash=hashlib.sha256(b"John Doe").hexdigest(),
        gross_salary=5_000_000_000_000,
        tax_withholding=600_000_000_000,
        social_security=250_000_000_000,
        pension_contribution=250_000_000_000,
        net_salary=3_900_000_000_000,
        payment_period="2026-02",
        employee_type=EmployeeType.PRIVATE,
    )
    defaults.update(overrides)
    return PayrollEntry(**defaults)


def _make_update(seed: int = 0, size: int = 6):
    torch.manual_seed(seed)
    return {
        "fc.weight": torch.randn(size, size),
        "fc.bias": torch.randn(size),
    }


# ===========================================================================
# C8.1 — Differential Privacy Wrapper Correctness
# ===========================================================================


class TestC81_DPWrapperCorrectness:

    def test_all_params_configurable(self):
        """ε, δ, clip_norm, noise_multiplier all settable."""
        dp = DifferentialPrivacy(
            epsilon=2.0,
            delta=1e-6,
            clip_norm=0.5,
            noise_multiplier=0.8,
            target_steps=500,
            sampling_rate=0.01,
        )
        assert dp.budget.epsilon == 2.0
        assert dp.budget.delta == 1e-6
        assert dp.clip_norm == 0.5
        assert dp.noise_multiplier == 0.8

    def test_budget_cap_enforced(self):
        """Training must stop after budget exhaustion."""
        dp = DifferentialPrivacy(
            epsilon=0.1, delta=1e-5, clip_norm=1.0, noise_multiplier=1.0
        )
        model = _model_with_grads()
        # Exhaust the budget by spending many steps
        exhausted = False
        for _ in range(10_000):
            model = _model_with_grads()
            dp.clip_gradients(model)
            dp.add_noise(model)
            try:
                dp.update_privacy_budget()
            except PrivacyBudgetExhaustedError:
                exhausted = True
                break
        assert exhausted, "PrivacyBudgetExhaustedError was never raised"

    def test_get_privacy_spent_returns_tuple(self):
        """get_privacy_spent() should return (epsilon_spent, delta)."""
        dp = DifferentialPrivacy(epsilon=5.0, delta=1e-5, target_steps=100)
        result = dp.get_privacy_spent()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_can_continue_training_reflects_budget(self):
        dp = DifferentialPrivacy(epsilon=5.0, delta=1e-5, target_steps=100)
        assert dp.can_continue_training() is True

    def test_clip_gradients_bounds_norm(self):
        """Gradient L2 norm must be ≤ clip_norm after clipping."""
        clip = 0.5
        dp = DifferentialPrivacy(epsilon=1.0, clip_norm=clip)
        model = _model_with_grads(scale=100.0)
        dp.clip_gradients(model)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5
        assert total_norm <= clip + 1e-5

    def test_add_noise_changes_gradients(self):
        """Noise injection must actually modify gradients."""
        dp = DifferentialPrivacy(epsilon=1.0, noise_multiplier=1.0, clip_norm=1.0)
        model = _model_with_grads()
        dp.clip_gradients(model)
        before = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }
        dp.add_noise(model)
        changed = False
        for n, p in model.named_parameters():
            if p.grad is not None and not torch.equal(before[n], p.grad):
                changed = True
        assert changed, "add_noise() must inject non-zero noise"

    def test_privacy_accountant_rdp_orders(self):
        """PrivacyAccountant must use multiple RDP orders."""
        pa = PrivacyAccountant(epsilon=1.0, delta=1e-5, sampling_rate=0.01)
        assert len(pa.orders) >= 4

    def test_dp_optimizer_wraps_step(self):
        """DPOptimizer.step() should clip, noise, and advance budget."""
        model = _model_with_grads()
        base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
        dp = DifferentialPrivacy(epsilon=5.0, clip_norm=1.0, noise_multiplier=0.5)
        dp_opt = DPOptimizer(base_opt, dp)
        dp_opt.step(model)
        eps, delta = dp.get_privacy_spent()
        assert eps > 0


# ===========================================================================
# C8.2 — Byzantine Aggregation Defences
# ===========================================================================


class TestC82_ByzantineAggregation:

    def test_prod_config_uses_byzantine_robust_method(self):
        """config.prod.yaml must NOT use fedavg (F8.2a fix)."""
        with open("config.prod.yaml") as f:
            cfg = yaml.safe_load(f)
        strategy = cfg["federated"]["aggregation_strategy"]
        assert (
            strategy != "fedavg"
        ), f"Production config uses '{strategy}' — must be Byzantine-robust"
        assert strategy in {"krum", "multi_krum", "trimmed_mean", "median", "phocas"}

    def test_median_is_coordinate_wise(self):
        """_median() must use torch.median, not torch.mean."""
        det = ByzantineDetector(num_byzantine=1)
        updates = [_make_update(seed=i) for i in range(5)]
        result = det.aggregate(updates, method=AggregationMethod.MEDIAN)
        # True coordinate-wise median: stack and take median
        stacked = torch.stack([u["fc.weight"] for u in updates])
        expected = torch.median(stacked, dim=0).values
        assert torch.allclose(result["fc.weight"], expected, atol=1e-6)

    def test_trimmed_mean_trim_ratio_valid(self):
        """Trimmed mean with default trim_ratio=0.1 must produce valid result."""
        det = ByzantineDetector(num_byzantine=2)
        updates = [_make_update(seed=i) for i in range(10)]
        result = det.aggregate(updates, method=AggregationMethod.TRIMMED_MEAN)
        # Should not crash and should return valid parameter keys
        assert "fc.weight" in result
        assert "fc.bias" in result

    def test_krum_rejects_too_many_byzantine(self):
        """Krum must fall back to FedAvg when 2f+2 >= n."""
        det = ByzantineDetector(num_byzantine=2)
        updates = [_make_update(seed=i) for i in range(4)]
        # Falls back to FedAvg instead of raising
        result = det.aggregate(updates, method=AggregationMethod.KRUM)
        assert isinstance(result, dict)

    def test_fedavg_still_available(self):
        """FedAvg should still work (just not be default in production)."""
        det = ByzantineDetector(num_byzantine=0)
        updates = [_make_update(seed=i) for i in range(3)]
        result = det.aggregate(updates, method=AggregationMethod.FEDAVG)
        assert "fc.weight" in result

    def test_all_six_methods_exist(self):
        methods = {m.value for m in AggregationMethod}
        assert methods == {
            "fedavg",
            "krum",
            "multi_krum",
            "trimmed_mean",
            "median",
            "phocas",
        }


# ===========================================================================
# C8.3 — ZK-Proof Payroll Generation (Merkle + Proof Data)
# ===========================================================================


class TestC83_ZKProofGeneration:

    def test_merkle_root_uses_salt(self):
        """Merkle tree must include a per-entry salt (F8.3b fix)."""
        from blockchain.payroll_connector import PayrollConnector

        connector = PayrollConnector.__new__(PayrollConnector)
        entries = [_make_payroll_entry()]
        root = connector._compute_merkle_root(entries)
        # Without salt, a naive hash of employee_id + salary fields would differ
        # Re-derive the expected salted leaf
        entry = entries[0]
        salt = hashlib.sha256(
            f"{entry.employee_id}:{entry.payment_period}".encode()
        ).digest()[:16]
        leaf_hash = hashlib.sha256(
            salt
            + hashlib.sha256(entry.employee_id.encode()).digest()
            + hashlib.sha256(str(entry.gross_salary).encode()).digest()
            + hashlib.sha256(str(entry.net_salary).encode()).digest()
        ).digest()
        assert root == leaf_hash.hex()

    def test_merkle_root_different_with_different_periods(self):
        """Same salary, different period must produce different merkle roots."""
        from blockchain.payroll_connector import PayrollConnector

        connector = PayrollConnector.__new__(PayrollConnector)
        e1 = _make_payroll_entry(payment_period="2026-01")
        e2 = _make_payroll_entry(payment_period="2026-02")
        root1 = connector._compute_merkle_root([e1])
        root2 = connector._compute_merkle_root([e2])
        assert (
            root1 != root2
        ), "Different periods must produce different roots (salt varies)"

    def test_proof_data_no_plaintext_totals(self):
        """Proof JSON must NOT contain plaintext salary totals."""
        from blockchain.payroll_connector import PayrollConnector
        import json

        connector = PayrollConnector.__new__(PayrollConnector)
        entries = [
            _make_payroll_entry(),
            _make_payroll_entry(employee_id="BZ-12345-0002"),
        ]
        merkle_root = connector._compute_merkle_root(entries)
        proof_json = connector._generate_zk_proof(entries, merkle_root)
        proof_data = json.loads(proof_json)
        # No plaintext salary totals anywhere in the proof
        assert "total_gross" not in proof_data
        assert "total_net" not in proof_data
        assert "gross_salary" not in proof_json
        assert "net_salary" not in str(proof_data.get("entry_commitments", []))

    def test_per_entry_commitment_not_aggregate(self):
        """Each entry must produce its own leaf hash (not one aggregate)."""
        from blockchain.payroll_connector import PayrollConnector

        connector = PayrollConnector.__new__(PayrollConnector)
        entries = [
            _make_payroll_entry(employee_id="BZ-00001"),
            _make_payroll_entry(employee_id="BZ-00002"),
        ]
        root_both = connector._compute_merkle_root(entries)
        root_single = connector._compute_merkle_root([entries[0]])
        assert root_both != root_single, "Multiple entries must combine individually"

    def test_zk_proof_works_in_production(self):
        """_generate_zk_proof must produce a valid JSON proof in any environment."""
        import json
        from blockchain.payroll_connector import PayrollConnector

        connector = PayrollConnector.__new__(PayrollConnector)
        entries = [_make_payroll_entry()]
        merkle_root = connector._compute_merkle_root(entries)
        with patch.dict(os.environ, {"NAWAL_ENV": "production"}, clear=False):
            proof_json = connector._generate_zk_proof(entries, merkle_root)
        proof = json.loads(proof_json)
        assert proof["version"] == 1
        assert proof["scheme"] == "hmac-sha256-commitment"
        assert len(proof["entry_commitments"]) == 1
        assert proof["entry_count"] == 1


# ===========================================================================
# C8.4 — ZK-Proof Verification
# ===========================================================================


class TestC84_ZKProofVerification:

    def test_verify_works_in_production(self):
        """PayrollProof.verify() must work in production with valid commitment proof."""
        import json
        from blockchain.payroll_connector import PayrollConnector

        connector = PayrollConnector.__new__(PayrollConnector)
        entries = [_make_payroll_entry()]
        merkle_root = connector._compute_merkle_root(entries)
        proof_json = connector._generate_zk_proof(entries, merkle_root)
        proof = PayrollProof(
            proof_type="zk-snark",
            proof_data=proof_json,
            public_inputs=[merkle_root],
            commitment=merkle_root,
        )
        assert proof.verify() is True

    def test_verify_valid_commitment_proof(self):
        """verify() accepts a correctly generated commitment proof."""
        import json
        from blockchain.payroll_connector import PayrollConnector

        connector = PayrollConnector.__new__(PayrollConnector)
        entries = [_make_payroll_entry(), _make_payroll_entry(employee_id="BZ-00002")]
        merkle_root = connector._compute_merkle_root(entries)
        proof_json = connector._generate_zk_proof(entries, merkle_root)
        proof = PayrollProof(
            proof_type="zk-snark",
            proof_data=proof_json,
            public_inputs=[merkle_root],
            commitment=merkle_root,
        )
        assert proof.verify() is True

    def test_verify_rejects_unknown_proof_type(self):
        """verify() must reject unknown proof_type."""
        bad_proof = PayrollProof(
            proof_type="unknown_type",
            proof_data='{"version":1}',
            public_inputs=["input1"],
            commitment="some_commitment",
        )
        assert bad_proof.verify() is False

    def test_verify_rejects_empty_proof_data(self):
        """verify() must reject empty proof_data."""
        bad_proof = PayrollProof(
            proof_type="zk-snark",
            proof_data="",
            public_inputs=["input1"],
            commitment="some_commitment",
        )
        assert bad_proof.verify() is False

    def test_verify_rejects_empty_public_inputs(self):
        """verify() must reject empty public_inputs."""
        bad_proof = PayrollProof(
            proof_type="zk-snark",
            proof_data='{"version":1}',
            public_inputs=[],
            commitment="some_commitment",
        )
        assert bad_proof.verify() is False

    def test_verify_rejects_tampered_commitment(self):
        """verify() must reject proof where commitment doesn't match merkle_binding."""
        import json
        from blockchain.payroll_connector import PayrollConnector

        connector = PayrollConnector.__new__(PayrollConnector)
        entries = [_make_payroll_entry()]
        merkle_root = connector._compute_merkle_root(entries)
        proof_json = connector._generate_zk_proof(entries, merkle_root)
        # Tamper: use wrong commitment
        proof = PayrollProof(
            proof_type="zk-snark",
            proof_data=proof_json,
            public_inputs=[merkle_root],
            commitment="tampered_root",
        )
        assert proof.verify() is False


# ===========================================================================
# C8.5 — KYC/AML Compliance Coverage
# ===========================================================================


class TestC85_ComplianceCoverage:

    def test_verifier_factory_blocks_dev_in_production(self):
        """create_verifier must raise when mode='development' and NAWAL_ENV=production."""
        from blockchain.identity_verifier import create_verifier

        with patch.dict(os.environ, {"NAWAL_ENV": "production"}, clear=False):
            with pytest.raises(RuntimeError, match="cannot be used"):
                create_verifier(mode="development")

    def test_verifier_factory_dev_mode_returns_dummy(self):
        """create_verifier mode='development' returns DummyBelizeIDVerifier."""
        from blockchain.identity_verifier import DummyBelizeIDVerifier, create_verifier

        with patch.dict(os.environ, {"NAWAL_ENV": "development"}, clear=False):
            verifier = create_verifier(mode="development")
            assert isinstance(verifier, DummyBelizeIDVerifier)

    def test_compliance_config_kyc_default_true(self):
        """ComplianceConfig defaults to require_kyc=True."""
        from config import ComplianceConfig

        cfg = ComplianceConfig()
        assert cfg.require_kyc is True

    def test_compliance_config_audit_logging_default_true(self):
        from config import ComplianceConfig

        cfg = ComplianceConfig()
        assert cfg.audit_logging is True

    def test_compliance_config_encryption_default_true(self):
        from config import ComplianceConfig

        cfg = ComplianceConfig()
        assert cfg.data_encryption is True

    def test_prod_config_kyc_required(self):
        """config.prod.yaml must set require_kyc: true."""
        with open("config.prod.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg.get("compliance", {}).get("require_kyc", True) is True


# ===========================================================================
# C8.6 — Content Filtering Scope
# ===========================================================================


class TestC86_ContentFiltering:

    def test_compliance_filter_redacts_ssn(self):
        """ComplianceFilter must redact SSN-like patterns."""
        cf = ComplianceFilter()
        text = "My SSN is 123-45-6789 and I need help"
        filtered = cf.filter(text)
        assert "123-45-6789" not in filtered
        assert "[REDACTED]" in filtered

    def test_compliance_filter_redacts_credit_card(self):
        """ComplianceFilter must redact 16-digit credit card numbers."""
        cf = ComplianceFilter()
        text = "Card number 1234567890123456"
        filtered = cf.filter(text)
        assert "1234567890123456" not in filtered

    def test_compliance_filter_redacts_password(self):
        """ComplianceFilter must redact password patterns."""
        cf = ComplianceFilter()
        text = "pwd: mysecretpassword123"
        filtered = cf.filter(text)
        assert "mysecretpassword123" not in filtered

    def test_output_filter_blocks_violence_instructions(self):
        """OutputFilter must block violence instructions."""
        of = OutputFilter()
        result = of.filter("", "Here are step by step instructions to murder someone")
        assert result.is_safe is False
        assert result.risk_level in (RiskLevel.HIGH, RiskLevel.BLOCKED)

    def test_output_filter_blocks_credential_leak(self):
        """OutputFilter must flag credential leaks."""
        of = OutputFilter()
        result = of.filter("", "The api_key = sk_live_abcdef123456789")
        assert result.is_safe is False
        assert "credential_leak" in result.flags

    def test_output_filter_blocks_private_key_leak(self):
        """OutputFilter must block private key exposure."""
        of = OutputFilter()
        # Regex matches: BEGIN (RSA|EC|PRIVATE) KEY
        result = of.filter("", "-----BEGIN PRIVATE KEY----- data here")
        assert result.is_safe is False

    def test_output_filter_flags_pii_ssn(self):
        """OutputFilter must flag SSN patterns in output."""
        of = OutputFilter()
        result = of.filter("", "The SSN is 123-45-6789")
        assert "ssn_in_output" in result.flags

    def test_output_filter_allows_safe_text(self):
        """OutputFilter must pass safe text."""
        of = OutputFilter()
        result = of.filter(
            "What is Bitcoin?", "Bitcoin is a decentralized cryptocurrency."
        )
        assert result.is_safe is True
        assert result.risk_level == RiskLevel.NONE or result.risk_level == RiskLevel.LOW

    def test_output_filter_imported_by_api_server(self):
        """api_server.py must import OutputFilter (C8.6 gap fix)."""
        import ast

        with open("api_server.py") as f:
            source = f.read()
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
        assert any(
            "output_filter" in imp for imp in imports
        ), "OutputFilter must be imported by api_server.py"

    def test_input_screener_imported_by_api_server(self):
        """api_server.py must import InputScreener (C8.6 gap fix)."""
        import ast

        with open("api_server.py") as f:
            source = f.read()
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
        assert any(
            "input_screener" in imp for imp in imports
        ), "InputScreener must be imported by api_server.py"


# ===========================================================================
# C8.7 — Azure / Secret Handling
# ===========================================================================


class TestC87_SecretHandling:

    def test_no_hardcoded_secrets_in_security_module(self):
        """No hardcoded API keys, passwords, or seeds in security/ files."""
        import glob

        secret_patterns = [
            re.compile(
                r"""(?:api_key|secret_key|password|seed_phrase)\s*=\s*['"][^'"]{8,}['"]""",
                re.I,
            ),
            re.compile(r"""sk-[a-zA-Z0-9]{20,}"""),
            re.compile(r"""-----BEGIN\s+(RSA|EC|PRIVATE)\s+KEY-----"""),
        ]
        for filepath in glob.glob("security/*.py"):
            with open(filepath) as f:
                content = f.read()
            for pattern in secret_patterns:
                match = pattern.search(content)
                assert (
                    match is None
                ), f"Hardcoded secret in {filepath}: {match.group()[:40]}..."

    def test_no_hardcoded_secrets_in_blockchain_module(self):
        """No hardcoded secrets in blockchain/ files."""
        import glob

        secret_patterns = [
            re.compile(
                r"""(?:api_key|secret_key|password|seed_phrase)\s*=\s*['"][^'"]{8,}['"]""",
                re.I,
            ),
            re.compile(r"""sk-[a-zA-Z0-9]{20,}"""),
        ]
        for filepath in glob.glob("blockchain/*.py"):
            with open(filepath) as f:
                content = f.read()
            for pattern in secret_patterns:
                match = pattern.search(content)
                assert (
                    match is None
                ), f"Hardcoded secret in {filepath}: {match.group()[:40]}..."

    def test_api_key_header_only(self):
        """API key must only be accepted from X-API-Key header, not query params (F8.7a fix)."""
        with open("api_server.py") as f:
            source = f.read()
        # Ensure no query_params.get("api_key") in auth middleware
        assert (
            'query_params.get("api_key")' not in source
        ), "API key should not be accepted from query params (leaks in logs)"

    def test_env_example_no_real_seeds(self):
        """The .env.example must not contain real seed phrases or keys."""
        with open(".env.example") as f:
            content = f.read()
        # BLOCKCHAIN_ACCOUNT_SEED should be blank
        for line in content.splitlines():
            if line.startswith("BLOCKCHAIN_ACCOUNT_SEED"):
                value = line.split("=", 1)[1].strip()
                assert value == "" or value.startswith(
                    "#"
                ), "BLOCKCHAIN_ACCOUNT_SEED must be blank in .env.example"

    def test_identity_verifier_blocks_dummy_in_production(self):
        """DummyBelizeIDVerifier must not be usable when NAWAL_ENV=production."""
        from blockchain.identity_verifier import create_verifier

        with patch.dict(os.environ, {"NAWAL_ENV": "production"}, clear=False):
            with pytest.raises(RuntimeError):
                create_verifier(mode="development")

    def test_prod_config_docs_disabled(self):
        """Production must disable /docs and /redoc endpoints."""
        with open("api_server.py") as f:
            source = f.read()
        assert "docs_url=None if _is_production else" in source
        assert "redoc_url=None if _is_production else" in source
