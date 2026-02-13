"""
Payroll Integration Example - Government Payroll Submission

Demonstrates how to submit government payroll with zero-knowledge proofs
to BelizeChain's Payroll pallet.

Author: BelizeChain AI Team
Date: February 2026
"""

import asyncio
import hashlib
import sys
from pathlib import Path
from decimal import Decimal
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from blockchain.payroll_connector import (
    PayrollConnector,
    PayrollEntry,
    PayrollSubmission,
    EmployeeType,
)

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO")


# Sample employee data (in production, load from secure database)
SAMPLE_EMPLOYEES = [
    {
        "name": "Maria Garcia",
        "belizeid": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        "gross_salary_monthly": 5000.00,  # DALLA
        "department": "Ministry of Health",
    },
    {
        "name": "John Smith",
        "belizeid": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        "gross_salary_monthly": 6500.00,
        "department": "Ministry of Health",
    },
    {
        "name": "Carlos Mendez",
        "belizeid": "5FLSigC9HGRKVhB9FiEo4Y3koPsNmBmLJbpXg2mp1hXcS59Y",
        "gross_salary_monthly": 4200.00,
        "department": "Ministry of Health",
    },
    {
        "name": "Sarah Johnson",
        "belizeid": "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy",
        "gross_salary_monthly": 7800.00,
        "department": "Ministry of Health",
    },
    {
        "name": "Roberto Cruz",
        "belizeid": "5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw",
        "gross_salary_monthly": 5500.00,
        "department": "Ministry of Health",
    },
]


async def submit_government_payroll():
    """Submit Ministry of Health payroll for February 2026."""
    
    logger.info("💼 Government Payroll Submission System")
    logger.info("=" * 70)
    logger.info("Ministry of Health - February 2026 Payroll")
    logger.info("")
    
    # Initialize payroll connector (mock mode for this example)
    payroll = PayrollConnector(
        websocket_url="ws://127.0.0.1:9944",
        keypair=None,  # In production, use ministry keypair
        mock_mode=True,  # Set to False for real blockchain
    )
    
    logger.info("Connecting to BelizeChain Payroll pallet...")
    connected = await payroll.connect()
    
    if connected:
        logger.info("✅ Connected to blockchain")
    else:
        logger.error("❌ Failed to connect to blockchain")
        return
    
    logger.info("")
    
    # Prepare payroll entries
    logger.info("Preparing payroll entries...")
    entries = []
    
    for emp in SAMPLE_EMPLOYEES:
        gross_salary_planck = int(emp["gross_salary_monthly"] * 100000000)  # Convert to Planck
        
        # Calculate deductions
        tax_withholding = payroll.calculate_tax_withholding(gross_salary_planck)
        social_security = int(gross_salary_planck * 0.08)  # 8% SSB contribution
        pension_contribution = int(gross_salary_planck * 0.05)  # 5% pension
        
        # Calculate net salary
        net_salary = gross_salary_planck - tax_withholding - social_security - pension_contribution
        
        # Create entry
        entry = PayrollEntry(
            employee_id=emp["belizeid"],
            employee_name_hash=hashlib.sha256(emp["name"].encode()).hexdigest(),
            gross_salary=gross_salary_planck,
            tax_withholding=tax_withholding,
            social_security=social_security,
            pension_contribution=pension_contribution,
            net_salary=net_salary,
            payment_period="2026-02",
            employee_type=EmployeeType.GOVERNMENT,
            department=emp["department"],
        )
        
        entries.append(entry)
        
        # Display entry details
        logger.info(f"Employee: {emp['name'][:20]:20} | "
                   f"Gross: {emp['gross_salary_monthly']:8.2f} | "
                   f"Tax: {tax_withholding / 100000000:7.2f} | "
                   f"Net: {net_salary / 100000000:8.2f} DALLA")
    
    logger.info("")
    logger.info(f"Total employees: {len(entries)}")
    
    # Calculate totals
    total_gross = sum(e.gross_salary for e in entries) / 100000000
    total_tax = sum(e.tax_withholding for e in entries) / 100000000
    total_net = sum(e.net_salary for e in entries) / 100000000
    
    logger.info(f"Total gross payroll: {total_gross:,.2f} DALLA")
    logger.info(f"Total tax withheld: {total_tax:,.2f} DALLA")
    logger.info(f"Total net payroll: {total_net:,.2f} DALLA")
    logger.info("")
    
    # Submit payroll with ZK-proof
    logger.info("Generating zero-knowledge proof...")
    logger.info("Computing Merkle tree...")
    logger.info("Submitting to blockchain...")
    logger.info("")
    
    try:
        submission = await payroll.submit_payroll(
            entries=entries,
            payment_period="2026-02",
            employer_name="Ministry of Health",
        )
        
        # Display submission details
        logger.info("✅ PAYROLL SUBMITTED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"Submission ID: {submission.submission_id}")
        logger.info(f"Payment Period: {submission.payment_period}")
        logger.info(f"Employer: {submission.employer_name}")
        logger.info(f"Employee Count: {submission.employee_count}")
        logger.info(f"Status: {submission.status.value}")
        logger.info("")
        logger.info("Zero-Knowledge Proof:")
        logger.info(f"  ZK Proof: {submission.zk_proof[:64]}...")
        logger.info(f"  Merkle Root: {submission.merkle_root}")
        logger.info("")
        logger.info("Financial Summary:")
        logger.info(f"  Total Gross: {submission.total_gross / 100000000:,.2f} DALLA")
        logger.info(f"  Total Tax: {submission.total_tax / 100000000:,.2f} DALLA")
        logger.info(f"  Total Net: {submission.total_net / 100000000:,.2f} DALLA")
        logger.info("")
        logger.info("Privacy Guarantee:")
        logger.info("  ✅ Individual salaries are NOT visible on-chain")
        logger.info("  ✅ Only aggregated totals and ZK-proof are stored")
        logger.info("  ✅ Validators can verify correctness without seeing amounts")
        logger.info("")
        
    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        return
    
    except RuntimeError as e:
        logger.error(f"❌ Submission error: {e}")
        return
    
    # Demonstrate getting payroll stats
    logger.info("Querying payroll statistics...")
    stats = await payroll.get_payroll_stats(payment_period="2026-02")
    
    if stats:
        logger.info("📊 FEBRUARY 2026 PAYROLL STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total Submissions: {stats.get('total_submissions', 0)}")
        logger.info(f"Total Employees: {stats.get('total_employees', 0):,}")
        logger.info(f"Government Employees: {stats.get('government_employees', 0):,}")
        logger.info(f"Private Employees: {stats.get('private_employees', 0):,}")
        logger.info(f"Total Gross Payroll: {stats.get('total_gross', 0) / 100000000:,.2f} DALLA")
        logger.info(f"Total Tax Collected: {stats.get('total_tax_collected', 0) / 100000000:,.2f} DALLA")
        logger.info("")
    
    # Disconnect
    await payroll.disconnect()
    logger.info("✅ Disconnected from blockchain")


async def get_employee_paystub():
    """Demonstrate employee paystub query."""
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("📄 EMPLOYEE PAYSTUB QUERY EXAMPLE")
    logger.info("=" * 70)
    logger.info("")
    
    # Initialize connector (mock mode)
    payroll = PayrollConnector(
        websocket_url="ws://127.0.0.1:9944",
        keypair=None,  # In production, use employee's keypair
        mock_mode=True,
    )
    
    await payroll.connect()
    
    # Query paystub for first sample employee
    employee = SAMPLE_EMPLOYEES[0]
    logger.info(f"Querying paystub for: {employee['name']}")
    logger.info(f"BelizeID: {employee['belizeid'][:20]}...")
    logger.info("")
    
    paystub = await payroll.get_employee_paystub(
        employee_id=employee["belizeid"],
        payment_period="2026-02",
    )
    
    if paystub:
        logger.info("📄 PAYSTUB - FEBRUARY 2026")
        logger.info("=" * 70)
        logger.info(f"Employer: {paystub.employer_name}")
        logger.info(f"Payment Period: {paystub.payment_period}")
        logger.info(f"Payment Date: {paystub.payment_date}")
        logger.info("")
        logger.info("Earnings:")
        logger.info(f"  Gross Salary:        {paystub.gross_salary / 100000000:>10.2f} DALLA")
        logger.info("")
        logger.info("Deductions:")
        logger.info(f"  Tax Withholding:     {paystub.tax_withholding / 100000000:>10.2f} DALLA")
        logger.info(f"  Social Security:     {paystub.social_security / 100000000:>10.2f} DALLA")
        logger.info(f"  Pension:             {paystub.pension_contribution / 100000000:>10.2f} DALLA")
        logger.info("  " + "-" * 40)
        logger.info(f"  NET PAY:             {paystub.net_salary / 100000000:>10.2f} DALLA")
        logger.info("")
        logger.info(f"Payment Status: {paystub.payment_status.upper()}")
        logger.info("=" * 70)
    else:
        logger.info("❌ No paystub found for this period")
    
    await payroll.disconnect()


async def main():
    """Run payroll examples."""
    
    # Submit government payroll
    await submit_government_payroll()
    
    # Query employee paystub
    await get_employee_paystub()
    
    logger.info("")
    logger.info("✅ Payroll example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
