"""
Tests for Payroll Connector functionality.

Tests payroll submission, ZK-proof generation, tax calculations,
and employee paystub queries.

Author: BelizeChain AI Team
Date: February 2026
"""

import hashlib
import pytest
from unittest.mock import Mock, patch

from blockchain.payroll_connector import (
    PayrollConnector,
    PayrollEntry,
    PayrollSubmission,
    PayrollStatus,
    EmployeeType,
    PayrollProof,
)


class TestPayrollEntry:
    """Test suite for PayrollEntry."""
    
    def test_valid_entry(self):
        """Test creating a valid payroll entry."""
        entry = PayrollEntry(
            employee_id="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            employee_name_hash=hashlib.sha256(b"John Doe").hexdigest(),
            gross_salary=500000000000,  # 5000 DALLA
            tax_withholding=100000000000,
            social_security=40000000000,
            pension_contribution=25000000000,
            net_salary=335000000000,
            payment_period="2026-02",
            employee_type=EmployeeType.GOVERNMENT,
        )
        
        assert entry.employee_id == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        assert entry.gross_salary == 500000000000
        assert entry.net_salary == 335000000000
        assert entry.employee_type == EmployeeType.GOVERNMENT
    
    def test_negative_salary(self):
        """Test that negative salaries are rejected."""
        with pytest.raises(ValueError, match="Gross salary cannot be negative"):
            PayrollEntry(
                employee_id="5GrwvaEF...",
                employee_name_hash="abc123",
                gross_salary=-1000,
                tax_withholding=0,
                social_security=0,
                pension_contribution=0,
                net_salary=0,
                payment_period="2026-02",
                employee_type=EmployeeType.GOVERNMENT,
            )
    
    def test_net_salary_mismatch(self):
        """Test that net salary calculation is validated."""
        with pytest.raises(ValueError, match="Net salary mismatch"):
            PayrollEntry(
                employee_id="5GrwvaEF...",
                employee_name_hash="abc123",
                gross_salary=500000000000,
                tax_withholding=100000000000,
                social_security=40000000000,
                pension_contribution=25000000000,
                net_salary=999999999999,  # Wrong!
                payment_period="2026-02",
                employee_type=EmployeeType.GOVERNMENT,
            )


class TestPayrollSubmission:
    """Test suite for PayrollSubmission."""
    
    def test_valid_submission(self):
        """Test creating a valid payroll submission."""
        entries = [
            PayrollEntry(
                employee_id="5GrwvaEF...",
                employee_name_hash=hashlib.sha256(b"Employee 1").hexdigest(),
                gross_salary=500000000000,
                tax_withholding=100000000000,
                social_security=40000000000,
                pension_contribution=25000000000,
                net_salary=335000000000,
                payment_period="2026-02",
                employee_type=EmployeeType.GOVERNMENT,
            ),
        ]
        
        submission = PayrollSubmission(
            submission_id="sub_001",
            employer_id="5FHneW46...",
            employer_name="Ministry of Health",
            payment_period="2026-02",
            total_gross=500000000000,
            total_tax=100000000000,
            total_net=335000000000,
            employee_count=1,
            entries=entries,
            zk_proof="proof123",
            merkle_root="root123",
        )
        
        errors = submission.validate()
        assert len(errors) == 0
    
    def test_employee_count_mismatch(self):
        """Test validation catches employee count mismatch."""
        entries = [
            PayrollEntry(
                employee_id="5GrwvaEF...",
                employee_name_hash="abc",
                gross_salary=500000000000,
                tax_withholding=100000000000,
                social_security=40000000000,
                pension_contribution=25000000000,
                net_salary=335000000000,
                payment_period="2026-02",
                employee_type=EmployeeType.GOVERNMENT,
            ),
        ]
        
        submission = PayrollSubmission(
            submission_id="sub_002",
            employer_id="5FHneW46...",
            employer_name="Ministry of Health",
            payment_period="2026-02",
            total_gross=500000000000,
            total_tax=100000000000,
            total_net=335000000000,
            employee_count=5,  # Wrong! Should be 1
            entries=entries,
            zk_proof="proof",
            merkle_root="root",
        )
        
        errors = submission.validate()
        assert len(errors) > 0
        assert any("Employee count mismatch" in err for err in errors)


class TestPayrollConnector:
    """Test suite for PayrollConnector."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test payroll connector initialization."""
        connector = PayrollConnector(
            websocket_url="ws://localhost:9944",
            mock_mode=True,
        )
        
        assert connector.websocket_url == "ws://localhost:9944"
        assert connector.mock_mode is True
        assert connector._connected is False
    
    @pytest.mark.asyncio
    async def test_connect_mock_mode(self):
        """Test connection in mock mode."""
        connector = PayrollConnector(mock_mode=True)
        
        result = await connector.connect()
        assert result is True
        assert connector._connected is True
    
    @pytest.mark.asyncio
    async def test_submit_payroll_mock(self):
        """Test payroll submission in mock mode."""
        connector = PayrollConnector(mock_mode=True)
        await connector.connect()
        
        # Create mock keypair
        mock_keypair = Mock()
        mock_keypair.ss58_address = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        connector.keypair = mock_keypair
        
        entries = [
            PayrollEntry(
                employee_id="5FHneW46...",
                employee_name_hash=hashlib.sha256(b"Test Employee").hexdigest(),
                gross_salary=500000000000,
                tax_withholding=100000000000,
                social_security=40000000000,
                pension_contribution=25000000000,
                net_salary=335000000000,
                payment_period="2026-02",
                employee_type=EmployeeType.GOVERNMENT,
            ),
        ]
        
        submission = await connector.submit_payroll(
            entries=entries,
            payment_period="2026-02",
            employer_name="Test Ministry",
        )
        
        assert submission.submission_id is not None
        assert submission.status == PayrollStatus.VERIFIED
        assert submission.employee_count == 1
        assert len(submission.zk_proof) > 0
        assert len(submission.merkle_root) > 0
    
    @pytest.mark.asyncio
    async def test_get_employee_paystub_mock(self):
        """Test employee paystub retrieval in mock mode."""
        connector = PayrollConnector(mock_mode=True)
        await connector.connect()
        
        paystub = await connector.get_employee_paystub(
            employee_id="5GrwvaEF...",
            payment_period="2026-02",
        )
        
        assert paystub is not None
        assert paystub.employee_id == "5GrwvaEF..."
        assert paystub.payment_period == "2026-02"
        assert paystub.gross_salary > 0
        assert paystub.net_salary > 0
    
    @pytest.mark.asyncio
    async def test_get_payroll_stats_mock(self):
        """Test payroll statistics query in mock mode."""
        connector = PayrollConnector(mock_mode=True)
        await connector.connect()
        
        stats = await connector.get_payroll_stats(payment_period="2026-02")
        
        assert stats is not None
        assert "total_employees" in stats
        assert "total_gross" in stats
        assert "total_tax_collected" in stats
    
    def test_merkle_root_computation(self):
        """Test Merkle root computation."""
        connector = PayrollConnector(mock_mode=True)
        
        entries = [
            PayrollEntry(
                employee_id="emp1",
                employee_name_hash="hash1",
                gross_salary=100000000000,
                tax_withholding=20000000000,
                social_security=8000000000,
                pension_contribution=5000000000,
                net_salary=67000000000,
                payment_period="2026-02",
                employee_type=EmployeeType.GOVERNMENT,
            ),
            PayrollEntry(
                employee_id="emp2",
                employee_name_hash="hash2",
                gross_salary=200000000000,
                tax_withholding=40000000000,
                social_security=16000000000,
                pension_contribution=10000000000,
                net_salary=134000000000,
                payment_period="2026-02",
                employee_type=EmployeeType.GOVERNMENT,
            ),
        ]
        
        merkle_root = connector._compute_merkle_root(entries)
        
        assert merkle_root is not None
        assert len(merkle_root) == 64  # SHA-256 hex string
    
    def test_zk_proof_generation(self):
        """Test ZK proof generation."""
        connector = PayrollConnector(mock_mode=True)
        
        entries = [
            PayrollEntry(
                employee_id="emp1",
                employee_name_hash="hash1",
                gross_salary=100000000000,
                tax_withholding=20000000000,
                social_security=8000000000,
                pension_contribution=5000000000,
                net_salary=67000000000,
                payment_period="2026-02",
                employee_type=EmployeeType.GOVERNMENT,
            ),
        ]
        
        merkle_root = connector._compute_merkle_root(entries)
        zk_proof = connector._generate_zk_proof(entries, merkle_root)
        
        assert zk_proof is not None
        assert len(zk_proof) > 0
    
    def test_tax_calculation_zero_bracket(self):
        """Test tax calculation for income below threshold."""
        connector = PayrollConnector(mock_mode=True)
        
        # Annual salary: 20,000 DALLA (below 26,000 threshold)
        gross_salary = 2000000000000  # 20,000 DALLA in Planck
        tax = connector.calculate_tax_withholding(gross_salary)
        
        assert tax == 0  # No tax in 0% bracket
    
    def test_tax_calculation_first_bracket(self):
        """Test tax calculation in first taxable bracket."""
        connector = PayrollConnector(mock_mode=True)
        
        # Annual salary: 26,500 DALLA (in 20% bracket)
        gross_salary = 2650000000000
        tax = connector.calculate_tax_withholding(gross_salary)
        
        # Should be 20% of (26,500 - 26,000) = 20% of 500 = 100 DALLA
        expected_tax = 10000000000  # 100 DALLA in Planck
        assert abs(tax - expected_tax) < 100000000  # Allow small rounding error
    
    def test_tax_calculation_top_bracket(self):
        """Test tax calculation in top bracket."""
        connector = PayrollConnector(mock_mode=True)
        
        # Annual salary: 50,000 DALLA (in 25% bracket)
        gross_salary = 5000000000000
        tax = connector.calculate_tax_withholding(gross_salary)
        
        # 0-26,000: 0%
        # 26,000-27,000: 20% = 200 DALLA
        # 27,000-50,000: 25% = 5,750 DALLA
        # Total: ~5,950 DALLA
        expected_tax_approx = 595000000000  # ~5,950 DALLA in Planck
        
        # Allow 10% margin due to simplified calculation
        assert abs(tax - expected_tax_approx) < expected_tax_approx * 0.1


class TestPayrollProof:
    """Test suite for PayrollProof."""
    
    def test_proof_verification(self):
        """Test ZK proof verification (simplified)."""
        proof = PayrollProof(
            proof_type="zk-snark",
            proof_data="abc123def456",
            public_inputs=["1000", "200"],
            commitment="root_hash_123",
        )
        
        # Current implementation just checks non-empty
        assert proof.verify() is True
    
    def test_empty_proof_fails(self):
        """Test that empty proof fails verification."""
        proof = PayrollProof(
            proof_type="zk-snark",
            proof_data="",
            public_inputs=[],
            commitment="",
        )
        
        assert proof.verify() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
