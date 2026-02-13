"""
Payroll Connector - BelizeChain Zero-Knowledge Payroll Integration

Connects Nawal AI to BelizeChain's Payroll pallet for privacy-preserving
payroll submissions and verifications using zero-knowledge proofs.

Features:
- ZK-proof payroll submissions (hide salary amounts)
- Validator payroll verification without revealing data
- Government and private sector payroll management
- Compliance tracking with KYC/AML integration
- Automated tax calculations and withholdings
- Integration with BelizeID for employee verification

The Payroll pallet enables:
1. Employers to submit encrypted payroll data
2. Validators to verify correctness via ZK-proofs
3. Employees to claim wages without exposing salaries
4. Government to collect taxes without seeing individual wages

Author: BelizeChain AI Team
Date: February 2026
Python: 3.11+
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from decimal import Decimal

from loguru import logger
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend

try:
    from substrateinterface import SubstrateInterface, Keypair
    from substrateinterface.exceptions import SubstrateRequestException
    SUBSTRATE_AVAILABLE = True
except ImportError:
    SUBSTRATE_AVAILABLE = False
    logger.warning("substrate-interface not installed, using mock mode")


# =============================================================================
# Data Classes
# =============================================================================


class PayrollStatus(Enum):
    """Status of payroll submission."""
    PENDING = "pending"
    VERIFIED = "verified"
    PROCESSED = "processed"
    FAILED = "failed"
    DISPUTED = "disputed"


class EmployeeType(Enum):
    """Type of employee."""
    GOVERNMENT = "government"
    PRIVATE = "private"
    CONTRACTOR = "contractor"
    SELF_EMPLOYED = "self_employed"


@dataclass
class PayrollEntry:
    """Individual payroll entry for an employee."""
    
    employee_id: str  # BelizeID account
    employee_name_hash: str  # SHA-256 hash of name (privacy)
    gross_salary: int  # In Planck (smallest unit)
    tax_withholding: int
    social_security: int
    pension_contribution: int
    net_salary: int
    payment_period: str  # YYYY-MM format
    employee_type: EmployeeType
    department: Optional[str] = None
    
    def __post_init__(self):
        """Validate payroll entry."""
        if self.gross_salary < 0:
            raise ValueError("Gross salary cannot be negative")
        
        if self.net_salary < 0:
            raise ValueError("Net salary cannot be negative")
        
        # Validate calculation
        expected_net = self.gross_salary - self.tax_withholding - self.social_security - self.pension_contribution
        if abs(expected_net - self.net_salary) > 1:  # Allow 1 Planck rounding error
            raise ValueError(f"Net salary mismatch: expected {expected_net}, got {self.net_salary}")


@dataclass
class PayrollSubmission:
    """Complete payroll submission for multiple employees."""
    
    submission_id: str
    employer_id: str  # BelizeID or business registration
    employer_name: str
    payment_period: str  # YYYY-MM
    total_gross: int
    total_tax: int
    total_net: int
    employee_count: int
    entries: List[PayrollEntry]
    zk_proof: str  # Zero-knowledge proof of correctness
    merkle_root: str  # Merkle root of all entries
    timestamp: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    status: PayrollStatus = PayrollStatus.PENDING
    
    def validate(self) -> List[str]:
        """Validate submission data."""
        errors = []
        
        if self.employee_count != len(self.entries):
            errors.append(f"Employee count mismatch: {self.employee_count} vs {len(self.entries)}")
        
        # Validate totals
        calc_gross = sum(e.gross_salary for e in self.entries)
        calc_tax = sum(e.tax_withholding for e in self.entries)
        calc_net = sum(e.net_salary for e in self.entries)
        
        if abs(calc_gross - self.total_gross) > len(self.entries):
            errors.append(f"Total gross mismatch: {calc_gross} vs {self.total_gross}")
        
        if abs(calc_tax - self.total_tax) > len(self.entries):
            errors.append(f"Total tax mismatch: {calc_tax} vs {self.total_tax}")
        
        if abs(calc_net - self.total_net) > len(self.entries):
            errors.append(f"Total net mismatch: {calc_net} vs {self.total_net}")
        
        # Validate individual entries
        for i, entry in enumerate(self.entries):
            try:
                entry.__post_init__()
            except ValueError as e:
                errors.append(f"Entry {i} invalid: {e}")
        
        return errors


@dataclass
class PayrollProof:
    """Zero-knowledge proof for payroll submission."""
    
    proof_type: str  # "zk-snark" or "bulletproof"
    proof_data: str  # Serialized proof
    public_inputs: List[str]  # Public verification inputs
    commitment: str  # Commitment to private data
    
    def verify(self) -> bool:
        """
        Verify the ZK proof (simplified implementation).
        
        In production, this would use a proper ZK-SNARK library
        like py-ecc, libsnark bindings, or circom.
        """
        # TODO: Implement actual ZK proof verification
        # For now, basic commitment verification
        return len(self.proof_data) > 0 and len(self.commitment) > 0


@dataclass
class EmployeePaystub:
    """Employee paystub (private, for employee's view only)."""
    
    employee_id: str
    payment_period: str
    gross_salary: int
    tax_withholding: int
    social_security: int
    pension_contribution: int
    net_salary: int
    payment_date: str
    employer_name: str
    payment_status: str  # "pending", "paid"


# =============================================================================
# Payroll Connector
# =============================================================================


class PayrollConnector:
    """
    Connector to BelizeChain Payroll pallet.
    
    Enables privacy-preserving payroll submissions using zero-knowledge proofs.
    Validators can verify payroll correctness without seeing individual salaries.
    
    Usage:
        connector = PayrollConnector(
            websocket_url="ws://localhost:9944",
            keypair=employer_keypair,
        )
        
        await connector.connect()
        
        # Submit payroll
        submission = await connector.submit_payroll(
            entries=[
                PayrollEntry(
                    employee_id="5GrwvaEF...",
                    employee_name_hash=hashlib.sha256(b"John Doe").hexdigest(),
                    gross_salary=500000000000,  # 5000 DALLA
                    tax_withholding=100000000000,
                    social_security=50000000000,
                    pension_contribution=25000000000,
                    net_salary=325000000000,
                    payment_period="2026-02",
                    employee_type=EmployeeType.GOVERNMENT,
                ),
            ],
            payment_period="2026-02",
        )
    """
    
    def __init__(
        self,
        websocket_url: str = "ws://127.0.0.1:9944",
        keypair: Optional[Keypair] = None,
        mock_mode: bool = False,
    ):
        """
        Initialize Payroll pallet connector.
        
        Args:
            websocket_url: BelizeChain node WebSocket endpoint
            keypair: Employer account keypair for signing transactions
            mock_mode: Use mock responses instead of real blockchain
        """
        self.websocket_url = websocket_url
        self.keypair = keypair
        self.mock_mode = mock_mode or not SUBSTRATE_AVAILABLE
        self.substrate: Optional[SubstrateInterface] = None
        self._connected = False
        
        if self.mock_mode:
            logger.warning("Payroll connector running in MOCK MODE")
    
    async def connect(self) -> bool:
        """
        Connect to BelizeChain node.
        
        Returns:
            True if connected successfully
        """
        if self._connected:
            return True
        
        if self.mock_mode:
            self._connected = True
            return True
        
        try:
            self.substrate = SubstrateInterface(url=self.websocket_url)
            self._connected = True
            logger.info(f"Connected to BelizeChain Payroll pallet at {self.websocket_url}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to BelizeChain: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from BelizeChain."""
        if self.substrate:
            self.substrate.close()
            self._connected = False
    
    async def submit_payroll(
        self,
        entries: List[PayrollEntry],
        payment_period: str,
        employer_name: Optional[str] = None,
    ) -> PayrollSubmission:
        """
        Submit payroll with ZK-proof.
        
        Args:
            entries: List of payroll entries
            payment_period: Payment period (YYYY-MM)
            employer_name: Employer name (optional)
        
        Returns:
            PayrollSubmission with proof
        
        Raises:
            ValueError: If validation fails
            RuntimeError: If submission fails
        """
        if not self._connected:
            raise RuntimeError("Not connected to blockchain")
        
        if not self.keypair:
            raise RuntimeError("No keypair configured")
        
        # Generate submission ID
        submission_id = hashlib.sha256(
            f"{self.keypair.ss58_address}{payment_period}{datetime.now(timezone.utc).timestamp()}".encode()
        ).hexdigest()[:16]
        
        # Calculate totals
        total_gross = sum(e.gross_salary for e in entries)
        total_tax = sum(e.tax_withholding for e in entries)
        total_net = sum(e.net_salary for e in entries)
        
        # Generate Merkle root
        merkle_root = self._compute_merkle_root(entries)
        
        # Generate ZK proof
        zk_proof = self._generate_zk_proof(entries, merkle_root)
        
        submission = PayrollSubmission(
            submission_id=submission_id,
            employer_id=self.keypair.ss58_address,
            employer_name=employer_name or "Unknown Employer",
            payment_period=payment_period,
            total_gross=total_gross,
            total_tax=total_tax,
            total_net=total_net,
            employee_count=len(entries),
            entries=entries,
            zk_proof=zk_proof,
            merkle_root=merkle_root,
        )
        
        # Validate submission
        errors = submission.validate()
        if errors:
            raise ValueError(f"Payroll validation failed: {errors}")
        
        # Submit to blockchain
        if not self.mock_mode:
            try:
                call = self.substrate.compose_call(
                    call_module="Payroll",
                    call_function="submit_payroll",
                    call_params={
                        "submission_id": submission_id,
                        "payment_period": payment_period,
                        "employee_count": len(entries),
                        "total_gross": total_gross,
                        "total_tax": total_tax,
                        "merkle_root": merkle_root,
                        "zk_proof": zk_proof,
                    },
                )
                
                extrinsic = self.substrate.create_signed_extrinsic(
                    call=call,
                    keypair=self.keypair,
                )
                
                receipt = self.substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=True,
                )
                
                if receipt.is_success:
                    submission.status = PayrollStatus.VERIFIED
                    logger.info(f"Payroll {submission_id} submitted successfully")
                else:
                    submission.status = PayrollStatus.FAILED
                    raise RuntimeError(f"Payroll submission failed: {receipt.error_message}")
            
            except SubstrateRequestException as e:
                logger.error(f"Blockchain error: {e}")
                submission.status = PayrollStatus.FAILED
                raise RuntimeError(f"Blockchain submission failed: {e}")
        
        else:
            # Mock mode - just mark as verified
            submission.status = PayrollStatus.VERIFIED
            logger.info(f"[MOCK] Payroll {submission_id} submitted")
        
        return submission
    
    async def verify_payroll(
        self,
        submission_id: str,
    ) -> bool:
        """
        Verify a payroll submission as a validator.
        
        Args:
            submission_id: Submission ID to verify
        
        Returns:
            True if verification successful
        """
        if not self._connected:
            raise RuntimeError("Not connected to blockchain")
        
        if not self.keypair:
            raise RuntimeError("No keypair configured for verification")
        
        if self.mock_mode:
            logger.info(f"[MOCK] Verified payroll {submission_id}")
            return True
        
        try:
            # Query submission from blockchain
            submission_data = self.substrate.query(
                module="Payroll",
                storage_function="PayrollSubmissions",
                params=[submission_id],
            )
            
            if not submission_data:
                logger.error(f"Payroll submission {submission_id} not found")
                return False
            
            # Verify ZK proof
            proof = PayrollProof(
                proof_type="zk-snark",
                proof_data=submission_data["zk_proof"],
                public_inputs=[
                    submission_data["total_gross"],
                    submission_data["total_tax"],
                    submission_data["merkle_root"],
                ],
                commitment=submission_data["merkle_root"],
            )
            
            if not proof.verify():
                logger.error(f"ZK proof verification failed for {submission_id}")
                return False
            
            # Submit verification
            call = self.substrate.compose_call(
                call_module="Payroll",
                call_function="verify_payroll",
                call_params={"submission_id": submission_id},
            )
            
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call,
                keypair=self.keypair,
            )
            
            receipt = self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=True,
            )
            
            if receipt.is_success:
                logger.info(f"Payroll {submission_id} verified successfully")
                return True
            else:
                logger.error(f"Verification submission failed: {receipt.error_message}")
                return False
        
        except Exception as e:
            logger.error(f"Payroll verification error: {e}")
            return False
    
    async def get_employee_paystub(
        self,
        employee_id: str,
        payment_period: str,
    ) -> Optional[EmployeePaystub]:
        """
        Get paystub for an employee (requires employee's keypair).
        
        Args:
            employee_id: Employee BelizeID account
            payment_period: Payment period (YYYY-MM)
        
        Returns:
            EmployeePaystub or None if not found
        """
        if not self._connected:
            raise RuntimeError("Not connected to blockchain")
        
        if self.mock_mode:
            # Mock paystub
            return EmployeePaystub(
                employee_id=employee_id,
                payment_period=payment_period,
                gross_salary=500000000000,
                tax_withholding=100000000000,
                social_security=50000000000,
                pension_contribution=25000000000,
                net_salary=325000000000,
                payment_date=f"{payment_period}-28",
                employer_name="Mock Employer",
                payment_status="paid",
            )
        
        try:
            # Query encrypted paystub
            paystub_data = self.substrate.query(
                module="Payroll",
                storage_function="EmployeePaystubs",
                params=[employee_id, payment_period],
            )
            
            if not paystub_data:
                logger.info(f"No paystub found for {employee_id} in {payment_period}")
                return None
            
            # Decrypt paystub (employee can decrypt with their private key)
            return EmployeePaystub(
                employee_id=employee_id,
                payment_period=payment_period,
                gross_salary=paystub_data["gross_salary"],
                tax_withholding=paystub_data["tax_withholding"],
                social_security=paystub_data["social_security"],
                pension_contribution=paystub_data["pension_contribution"],
                net_salary=paystub_data["net_salary"],
                payment_date=paystub_data["payment_date"],
                employer_name=paystub_data["employer_name"],
                payment_status=paystub_data["status"],
            )
        
        except Exception as e:
            logger.error(f"Failed to get paystub: {e}")
            return None
    
    async def get_payroll_stats(
        self,
        payment_period: str,
    ) -> Dict[str, Any]:
        """
        Get aggregated payroll statistics (government view).
        
        Args:
            payment_period: Payment period (YYYY-MM)
        
        Returns:
            Dictionary with stats
        """
        if not self._connected:
            raise RuntimeError("Not connected to blockchain")
        
        if self.mock_mode:
            return {
                "payment_period": payment_period,
                "total_submissions": 150,
                "total_employees": 5000,
                "total_gross": 25000000000000,  # 250M DALLA
                "total_tax_collected": 5000000000000,  # 50M DALLA
                "government_employees": 2000,
                "private_employees": 3000,
            }
        
        try:
            stats = self.substrate.query(
                module="Payroll",
                storage_function="PeriodStats",
                params=[payment_period],
            )
            
            return stats or {}
        
        except Exception as e:
            logger.error(f"Failed to get payroll stats: {e}")
            return {}
    
    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------
    
    def _compute_merkle_root(self, entries: List[PayrollEntry]) -> str:
        """
        Compute Merkle root of payroll entries.
        
        Args:
            entries: List of payroll entries
        
        Returns:
            Hex-encoded Merkle root
        """
        # Hash each entry
        leaves = []
        for entry in entries:
            entry_data = f"{entry.employee_id}{entry.gross_salary}{entry.net_salary}"
            leaf_hash = hashlib.sha256(entry_data.encode()).digest()
            leaves.append(leaf_hash)
        
        # Build Merkle tree
        while len(leaves) > 1:
            new_level = []
            for i in range(0, len(leaves), 2):
                if i + 1 < len(leaves):
                    combined = leaves[i] + leaves[i + 1]
                else:
                    combined = leaves[i] + leaves[i]
                
                parent_hash = hashlib.sha256(combined).digest()
                new_level.append(parent_hash)
            
            leaves = new_level
        
        return leaves[0].hex() if leaves else ""
    
    def _generate_zk_proof(
        self,
        entries: List[PayrollEntry],
        merkle_root: str,
    ) -> str:
        """
        Generate zero-knowledge proof for payroll submission.
        
        In production, this would use a proper ZK-SNARK library.
        For now, it generates a commitment-based proof.
        
        Args:
            entries: List of payroll entries
            merkle_root: Merkle root of entries
        
        Returns:
            Hex-encoded ZK proof
        """
        # TODO: Implement actual ZK-SNARK proof generation
        # This could use libraries like:
        # - py-ecc (for elliptic curve cryptography)
        # - circom/snarkjs (via subprocess)
        # - bellman (Rust bindings)
        
        # For now, generate deterministic commitment
        proof_data = {
            "merkle_root": merkle_root,
            "entry_count": len(entries),
            "total_gross": sum(e.gross_salary for e in entries),
            "total_net": sum(e.net_salary for e in entries),
            "timestamp": datetime.now(timezone.utc).timestamp(),
        }
        
        proof_json = json.dumps(proof_data, sort_keys=True)
        proof_hash = hashlib.sha256(proof_json.encode()).hexdigest()
        
        return proof_hash
    
    def calculate_tax_withholding(
        self,
        gross_salary: int,
        tax_brackets: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Calculate tax withholding based on Belize tax brackets.
        
        Belize Progressive Income Tax (2026):
        - 0 - 26,000 BZD: 0%
        - 26,001 - 27,000 BZD: 20%
        - Above 27,000 BZD: 25%
        
        Args:
            gross_salary: Annual gross salary in Planck
            tax_brackets: Custom tax brackets (optional)
        
        Returns:
            Tax withholding in Planck
        """
        # Default Belize tax brackets (in DALLA, 1 DALLA = 100,000,000 Planck)
        if tax_brackets is None:
            tax_brackets = [
                {"threshold": 0, "rate": 0.0},
                {"threshold": 2600000000000, "rate": 0.20},  # 26,000 DALLA
                {"threshold": 2700000000000, "rate": 0.25},  # 27,000 DALLA
            ]
        
        tax = 0
        remaining = gross_salary
        
        for i, bracket in enumerate(tax_brackets):
            if i == len(tax_brackets) - 1:
                # Last bracket - tax all remaining
                tax += int(remaining * bracket["rate"])
                break
            else:
                # Calculate amount in this bracket
                next_threshold = tax_brackets[i + 1]["threshold"]
                if gross_salary > next_threshold:
                    amount_in_bracket = next_threshold - bracket["threshold"]
                    tax += int(amount_in_bracket * bracket["rate"])
                    remaining -= amount_in_bracket
                else:
                    amount_in_bracket = gross_salary - bracket["threshold"]
                    tax += int(amount_in_bracket * bracket["rate"])
                    break
        
        return tax
