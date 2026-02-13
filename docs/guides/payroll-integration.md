# 💰 Payroll Integration Guide

**Version**: 1.1.0  
**Component**: `blockchain/payroll_connector.py`  
**Status**: Production Ready

---

## Overview

The ZK-proof payroll system enables government entities to submit employee payroll to BelizeChain while preserving salary privacy. Validators verify payroll correctness and earn Proof-of-Useful-Work (PoUW) rewards.

### Key Features

- **Privacy-Preserving**: Salaries are hidden using Merkle tree commitments
- **Zero-Knowledge Proofs**: Validators verify without seeing individual salaries
- **Automated Tax Calculation**: Belize tax brackets (0%, 25%, 40%)
- **Paystub Queries**: Employees retrieve their own paystubs privately
- **PoUW Rewards**: Validators earn DALLA for payroll verification

---

## Architecture

```
┌──────────────────┐
│  Government      │
│  (Employer)      │
└────────┬─────────┘
         │
         │ 1. Submit Payroll
         ▼
┌────────────────────────┐
│  PayrollConnector      │
│  ├─ Calculate taxes    │
│  ├─ Build Merkle tree  │
│  └─ Generate ZK-proofs │
└────────┬───────────────┘
         │
         │ 2. Submit to BelizeChain
         ▼
┌────────────────────────┐
│  BelizeChain           │
│  Payroll Pallet        │
│  ├─ Verify proofs      │
│  ├─ Store commitment   │
│  └─ Reward validators  │
└────────┬───────────────┘
         │
         │ 3. Validator Verification
         ▼
┌────────────────────────┐
│  Validators            │
│  ├─ Verify Merkle tree │
│  ├─ Check tax calcs    │
│  └─ Earn PoUW rewards  │
└────────────────────────┘
```

---

## Quick Start

### Installation

```bash
# Already included in Nawal AI
pip install -r requirements.txt
```

### Basic Usage

```python
import asyncio
from blockchain import PayrollConnector, PayrollEntry

async def main():
    # Initialize connector
    payroll = PayrollConnector(
        blockchain_rpc="ws://localhost:9944",
        account_seed="//Alice",  # Government account
    )
    
    # Define employees
    employees = [
        PayrollEntry(
            employee_id="emp_001",
            account_id="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            gross_salary=500000,  # In cents (BZD $5,000)
            deductions=50000,     # In cents (BZD $500)
            pay_period_start=1707782400,
            pay_period_end=1708387200,
        ),
        PayrollEntry(
            employee_id="emp_002",
            account_id="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            gross_salary=800000,  # BZD $8,000
            deductions=100000,    # BZD $1,000
            pay_period_start=1707782400,
            pay_period_end=1708387200,
        ),
    ]
    
    # Submit payroll
    submission_id = await payroll.submit_payroll(
        employees=employees,
        pay_period_id="2026-02",
        employer_name="Ministry of Health",
    )
    
    print(f"✅ Payroll submitted: {submission_id}")
    
    # Employees can query their paystubs
    paystub = await payroll.get_employee_paystub(
        submission_id=submission_id,
        employee_id="emp_001",
    )
    
    print(f"Employee paystub: {paystub}")

asyncio.run(main())
```

---

## Configuration

### Environment Variables

```bash
# Payroll Configuration
BLOCKCHAIN_RPC="ws://localhost:9944"           # BelizeChain RPC endpoint
PAYROLL_ACCOUNT_SEED="//Alice"                 # Government account seed
PAYROLL_TAX_YEAR=2026                          # Tax year for calculations

# Optional
PAYROLL_ENABLE_VALIDATION=true                 # Enable local validation
PAYROLL_MERKLE_TREE_DEPTH=16                   # Merkle tree depth (default: 16)
```

### Python Configuration

```python
from blockchain import PayrollConnector

connector = PayrollConnector(
    blockchain_rpc="ws://localhost:9944",   # Required
    account_seed="//Alice",                 # Required
)
```

---

## Belize Tax Calculation

### Tax Brackets (2026)

| Annual Income (BZD) | Tax Rate | Threshold |
|---------------------|----------|-----------|
| $0 - $26,000        | 0%       | Exempt    |
| $26,001 - $47,000   | 25%      | On excess |
| $47,001+            | 40%      | On excess |

### Example Calculation

```python
# Gross salary: BZD $8,000/month = $96,000/year
gross_annual = 96000.00

# Calculation:
# First $26,000: $0 (0%)
# Next $21,000 ($26,001 to $47,000): $5,250 (25%)
# Remaining $49,000 ($47,001+): $19,600 (40%)
# Total tax: $24,850/year = $2,070.83/month

monthly_tax = 2070.83
net_salary = 8000.00 - 2070.83  # = $5,929.17
```

### Implementation

Taxes are automatically calculated:

```python
from blockchain.payroll_connector import calculate_belize_tax

# Calculate monthly tax
income_bzd = 8000.00  # Monthly gross
tax = calculate_belize_tax(income_bzd)
print(f"Monthly tax: ${tax:.2f}")  # $2,070.83
```

---

## Privacy & Zero-Knowledge Proofs

### How It Works

1. **Merkle Tree Construction**:
   Each employee's data is hashed and placed in a Merkle tree
   
2. **Root Commitment**:
   Only the Merkle root is published to the blockchain
   
3. **ZK-Proofs**:
   Proofs demonstrate payroll correctness without revealing individual salaries
   
4. **Private Paystub Retrieval**:
   Employees query with their ID to get their specific data

### Example ZK-Proof

```python
# Government submits payroll
submission = await payroll.submit_payroll(employees=employees)

# ZK-proof components:
proof = {
    "merkle_root": "0xabc123...",           # Merkle tree root
    "total_employees": 150,                 # Public
    "total_gross": "hidden",                # Private
    "total_net": "hidden",                  # Private
    "tax_year": 2026,                       # Public
}

# Validators verify:
# ✅ Merkle root is correct
# ✅ Tax calculations follow Belize law
# ✅ No salary information leaked
```

---

## Payroll Submission

### Creating PayrollEntry

```python
from blockchain import PayrollEntry

entry = PayrollEntry(
    employee_id="emp_12345",              # Required: Unique employee ID
    account_id="5GrwvaEF...",             # Required: BelizeChain account
    gross_salary=500000,                  # Required: In cents (BZD $5,000)
    deductions=50000,                     # Required: In cents (BZD $500)
    pay_period_start=1707782400,          # Required: Unix timestamp
    pay_period_end=1708387200,            # Required: Unix timestamp
)

# Automatically populated:
print(f"Tax amount: {entry.tax_amount}")        # Calculated
print(f"Net salary: {entry.net_salary}")        # gross - tax - deductions
```

### Submitting to Blockchain

```python
# Submit payroll batch
submission_id = await payroll.submit_payroll(
    employees=[entry1, entry2, ...],
    pay_period_id="2026-02",              # Format: YYYY-MM
    employer_name="Ministry of Health",
    metadata={                             # Optional
        "department": "Nursing",
        "location": "Belize City Hospital",
    },
)

print(f"Submission ID: {submission_id}")
# Output: "payroll_20260213_001"
```

### Batch Processing

For large payrolls (500+ employees):

```python
# Process in batches of 100
batch_size = 100
for i in range(0, len(employees), batch_size):
    batch = employees[i:i + batch_size]
    
    submission_id = await payroll.submit_payroll(
        employees=batch,
        pay_period_id=f"2026-02-batch-{i // batch_size}",
        employer_name="Ministry of Health",
    )
    
    print(f"Batch {i // batch_size} submitted: {submission_id}")
```

---

## Validator Verification

### Earning PoUW Rewards

Validators verify payroll and earn rewards:

```python
# Validator checks new payroll submissions
submissions = await payroll.get_pending_submissions()

for submission in submissions:
    # Verify Merkle tree
    is_valid = await payroll.verify_payroll_submission(
        submission_id=submission["id"],
    )
    
    if is_valid:
        # Submit verification
        reward = await payroll.submit_verification(
            submission_id=submission["id"],
            is_valid=True,
        )
        print(f"✅ Verified! Earned {reward} DALLA")
    else:
        # Report invalid payroll
        await payroll.report_invalid_payroll(
            submission_id=submission["id"],
            reason="Tax calculation error",
        )
```

### Verification Process

1. **Fetch submission** from blockchain
2. **Rebuild Merkle tree** from payroll data
3. **Compare roots** - must match
4. **Verify tax calculations** for each employee
5. **Submit verification** to earn reward

### Reward Structure

| Action | Reward (DALLA) |
|--------|----------------|
| Correct verification | 10 DALLA |
| First verifier bonus | +5 DALLA |
| Invalid detection | 20 DALLA |
| False report penalty | -50 DALLA |

---

## Employee Paystub Queries

### Retrieving Paystubs

Employees can privately query their paystubs:

```python
# Employee queries their paystub
paystub = await payroll.get_employee_paystub(
    submission_id="payroll_20260213_001",
    employee_id="emp_12345",
)

print(f"Gross: ${paystub['gross_salary'] / 100:.2f}")
print(f"Tax: ${paystub['tax_amount'] / 100:.2f}")
print(f"Deductions: ${paystub['deductions'] / 100:.2f}")
print(f"Net: ${paystub['net_salary'] / 100:.2f}")
```

### Privacy Guarantees

- ✅ Only the employee can query their own paystub
- ✅ Queries do not reveal employee ID on-chain
- ✅ No other employee salaries are exposed
- ✅ Government cannot see query activity

### Paystub Format

```json
{
  "employee_id": "emp_12345",
  "account_id": "5GrwvaEF...",
  "gross_salary": 500000,
  "tax_amount": 87500,
  "deductions": 50000,
  "net_salary": 362500,
  "pay_period_start": 1707782400,
  "pay_period_end": 1708387200,
  "employer_name": "Ministry of Health",
  "merkle_proof": ["0xabc...", "0xdef..."]
}
```

---

## Advanced Usage

### Custom Tax Calculations

Override default Belize tax:

```python
def custom_tax_calculator(gross_monthly: float) -> float:
    """Custom tax calculation."""
    if gross_monthly < 2000:
        return 0.0
    elif gross_monthly < 5000:
        return (gross_monthly - 2000) * 0.20
    else:
        return 600 + (gross_monthly - 5000) * 0.35

# Use custom calculator
payroll.tax_calculator = custom_tax_calculator
```

### Merkle Proof Generation

Generate Merkle proofs for specific employees:

```python
# Get Merkle proof for employee
proof = await payroll.generate_merkle_proof(
    submission_id="payroll_001",
    employee_id="emp_12345",
)

# Proof contains sibling hashes for verification
print(f"Proof path: {proof['path']}")
print(f"Indices: {proof['indices']}")
```

### Audit Trail

Query payroll history:

```python
# Get all submissions for employer
history = await payroll.get_employer_history(
    employer_name="Ministry of Health",
    start_date="2026-01-01",
    end_date="2026-12-31",
)

for submission in history:
    print(f"{submission['pay_period_id']}: {submission['total_employees']} employees")
```

---

## Deployment

### Docker Compose

```yaml
services:
  payroll-processor:
    image: belizechainregistry.azurecr.io/nawal-ai:1.1.0
    environment:
      - BLOCKCHAIN_RPC=ws://blockchain:9944
      - PAYROLL_ACCOUNT_SEED=${PAYROLL_SEED}
      - PAYROLL_TAX_YEAR=2026
    volumes:
      - ./payroll_data:/data
```

### Kubernetes

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: payroll-processor
spec:
  schedule: "0 0 1 * *"  # Monthly on 1st
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: processor
              image: belizechainregistry.azurecr.io/nawal-ai:1.1.0
              env:
                - name: BLOCKCHAIN_RPC
                  value: "ws://blockchain:9944"
                - name: PAYROLL_ACCOUNT_SEED
                  valueFrom:
                    secretKeyRef:
                      name: payroll-secrets
                      key: account-seed
```

---

## Security Best Practices

### Account Security

```python
# ❌ DON'T: Hardcode seeds
payroll = PayrollConnector(account_seed="//Alice")

# ✅ DO: Use environment variables
import os
payroll = PayrollConnector(
    account_seed=os.environ["PAYROLL_ACCOUNT_SEED"]
)

# ✅ BEST: Use Azure Key Vault
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
client = SecretClient(vault_url="https://...", credential=credential)
seed = client.get_secret("payroll-seed").value

payroll = PayrollConnector(account_seed=seed)
```

### Data Validation

```python
# Validate employee data before submission
def validate_employee(entry: PayrollEntry) -> bool:
    if entry.gross_salary <= 0:
        raise ValueError("Gross salary must be positive")
    
    if entry.deductions < 0:
        raise ValueError("Deductions cannot be negative")
    
    if entry.pay_period_end <= entry.pay_period_start:
        raise ValueError("Invalid pay period")
    
    return True

# Validate before submission
for entry in employees:
    validate_employee(entry)
```

### Audit Logging

```python
import logging

# Configure audit logger
audit_logger = logging.getLogger("payroll.audit")
audit_logger.setLevel(logging.INFO)

# Log all submissions
audit_logger.info(f"Payroll submitted: {submission_id}, employees: {len(employees)}")
```

---

## Troubleshooting

### Failed Submission

```python
try:
    submission_id = await payroll.submit_payroll(employees)
except Exception as e:
    print(f"Submission failed: {e}")
    
    # Check blockchain connection
    health = await payroll.client.get_system_health()
    print(f"Blockchain health: {health}")
    
    # Validate individual entries
    for i, entry in enumerate(employees):
        try:
            validate_employee(entry)
        except ValueError as ve:
            print(f"Employee {i} invalid: {ve}")
```

### Tax Calculation Mismatch

```python
# Debug tax calculation
from blockchain.payroll_connector import calculate_belize_tax

gross = 8000.00
expected_tax = 2070.83

calculated = calculate_belize_tax(gross)
if abs(calculated - expected_tax) > 0.01:
    print(f"Tax mismatch: expected {expected_tax}, got {calculated}")
    
    # Check tax year
    print(f"Using {payroll.tax_year} tax brackets")
```

### Paystub Not Found

```python
# Check submission exists
submission = await payroll.get_submission(submission_id)
if not submission:
    print("Submission not found")
else:
    # Check employee ID
    employee_ids = [e["employee_id"] for e in submission["employees"]]
    if employee_id not in employee_ids:
        print(f"Employee ID not in submission. Available: {employee_ids}")
```

---

## Performance

### Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Single submission (10 employees) | ~500ms | Including Merkle tree |
| Large batch (500 employees) | ~3s | Parallel processing |
| Paystub query | ~100ms | O(log n) lookup |
| Verification | ~200ms | Merkle proof verification |

### Optimization Tips

1. **Batch submissions**: Group 100-500 employees per batch
2. **Parallel processing**: Use `asyncio.gather()` for multiple batches
3. **Cache blockchain client**: Reuse connection for multiple submissions
4. **Precompute taxes**: Calculate taxes before submission

---

## Example: Complete Payroll Flow

See [examples/payroll_example.py](../../examples/payroll_example.py) for a complete working example.

---

**Next Steps**:
- [Mesh Networking Guide](mesh-networking.md)
- [API Reference](../reference/api-reference.md)
- [Deployment Guide](deployment.md)
