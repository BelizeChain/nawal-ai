# Nawal Federated Server

**Coordinate federated learning across BelizeChain validators**

The federated server manages the distributed training of Nawal's evolving AI genomes across multiple BelizeChain validator nodes. It implements secure aggregation, participant management, and real-time metrics tracking.

## 📦 Components

### 1. **FederatedAggregator** - Model Aggregation
Aggregates model updates from participants using multiple strategies:

- **FedAvg**: Classic Federated Averaging (McMahan et al., 2017)
  - Sample-based weighting
  - Fitness-based weighting  
  - Hybrid weighting (70% samples + 30% fitness)

- **Byzantine-Robust**: Trimmed median aggregation
  - Removes extreme outliers
  - Protects against malicious participants
  - More robust but slower

```python
from nawal.server import FederatedAggregator, FedAvgStrategy

# Initialize with FedAvg strategy
aggregator = FederatedAggregator(
    strategy=FedAvgStrategy(weighting="hybrid"),
    min_participants=3,
    max_wait_time=300.0,
)

# Set current genome
aggregator.set_genome(genome, initial_weights)

# Submit update from participant
await aggregator.submit_update(model_update)

# Get aggregated global model
global_weights = aggregator.get_global_weights()
```

### 2. **ParticipantManager** - Validator Tracking
Manages validator participants in federated learning:

- Participant enrollment and verification
- Status tracking (active, idle, offline, Byzantine, slashed)
- Contribution tracking (rounds, samples, training time)
- Performance metrics (quality, timeliness, honesty, fitness)
- Reward calculation and distribution
- Byzantine behavior detection
- Reputation scoring

```python
from nawal.server import ParticipantManager, ParticipantStatus

# Initialize manager
manager = ParticipantManager(
    min_reputation=50.0,
    byzantine_threshold=3,
    activity_timeout=600,
)

# Enroll validator
participant = manager.enroll_participant(
    participant_id="validator_001",
    validator_address="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
    staking_account="staking_001",
)

# Record contribution
manager.record_contribution(
    participant_id="validator_001",
    samples=1000,
    training_time=120.5,
    quality=85.0,
    timeliness=90.0,
    honesty=95.0,
)

# Distribute rewards
rewards = manager.distribute_rewards(
    base_reward=100.0,  # 100 DALLA per round
    round_number=42,
)
```

### 3. **MetricsTracker** - Training Analytics
Tracks and aggregates training metrics across the population:

- Individual training metrics (loss, accuracy, throughput)
- Population-wide aggregation
- Time-series tracking for visualization
- Export formats (JSON, Prometheus)
- Resource usage tracking (memory, GPU)
- Fitness score tracking

```python
from nawal.server import MetricsTracker, TrainingMetrics

# Initialize tracker
tracker = MetricsTracker(max_history=1000)

# Record metrics from participant
metrics = TrainingMetrics(
    participant_id="validator_001",
    genome_id="genome_42",
    round_number=10,
    train_loss=0.234,
    train_accuracy=92.5,
    samples_trained=1000,
    training_time=120.5,
    quality_score=85.0,
    fitness_score=87.5,
)
tracker.record_metrics(metrics)

# Aggregate round metrics
aggregated = tracker.aggregate_round_metrics(round_number=10)
print(f"Average loss: {aggregated.avg_train_loss:.4f}")
print(f"Average fitness: {aggregated.avg_fitness:.2f}")

# Export to JSON
tracker.export_to_json("metrics_round_10.json", round_number=10)

# Export to Prometheus
prometheus_metrics = tracker.export_to_prometheus()
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Federated Server                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────┐ │
│  │ Federated        │  │ Participant      │  │ Metrics   │ │
│  │ Aggregator       │  │ Manager          │  │ Tracker   │ │
│  ├──────────────────┤  ├──────────────────┤  ├───────────┤ │
│  │ - FedAvg         │  │ - Enrollment     │  │ - Loss    │ │
│  │ - Byzantine-     │  │ - Status         │  │ - Accuracy│ │
│  │   robust         │  │ - Contributions  │  │ - Fitness │ │
│  │ - Weighting      │  │ - Rewards        │  │ - Export  │ │
│  │ - History        │  │ - Reputation     │  │ - Time    │ │
│  └──────────────────┘  └──────────────────┘  │   series  │ │
│           │                     │             └───────────┘ │
│           │                     │                    │       │
│           └─────────────────────┴────────────────────┘       │
│                              │                                │
└──────────────────────────────┼────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
         ┌──────▼─────┐              ┌───────▼──────┐
         │ Validator  │              │  Validator   │
         │ Node 1     │     ...      │  Node N      │
         │            │              │              │
         │ - Train    │              │  - Train     │
         │ - Submit   │              │  - Submit    │
         └────────────┘              └──────────────┘
```

## 🔄 Federated Learning Flow

### 1. **Initialization**
```python
# Server setup
aggregator = FederatedAggregator(
    strategy=FedAvgStrategy(weighting="hybrid"),
    min_participants=3,
)
manager = ParticipantManager()
tracker = MetricsTracker()

# Set genome
aggregator.set_genome(genome, initial_weights)

# Enroll validators
for validator in validators:
    manager.enroll_participant(
        participant_id=validator.id,
        validator_address=validator.address,
        staking_account=validator.staking_id,
    )
```

### 2. **Training Round**
```python
# Validators train locally
for participant in active_participants:
    # Train on local data
    weights, metrics = await participant.train_local(
        global_weights=aggregator.get_global_weights(),
        genome=genome,
    )
    
    # Submit update
    update = ModelUpdate(
        participant_id=participant.id,
        genome_id=genome.genome_id,
        round_number=round_num,
        weights=weights,
        samples_trained=metrics.samples_trained,
        training_time=metrics.training_time,
        quality_score=metrics.quality_score,
        timeliness_score=metrics.timeliness_score,
        honesty_score=metrics.honesty_score,
    )
    
    await aggregator.submit_update(update)
    tracker.record_metrics(metrics)
```

### 3. **Aggregation** (Automatic)
```python
# Triggered automatically when min_participants reached
# Server aggregates updates using configured strategy
# Updates global model weights
# Records aggregation history
```

### 4. **Metrics & Rewards**
```python
# Aggregate metrics
aggregated = tracker.aggregate_round_metrics(round_num)

# Distribute rewards based on fitness
rewards = manager.distribute_rewards(
    base_reward=100.0,  # 100 DALLA per round
    round_number=round_num,
)

# Validators claim rewards on blockchain
for participant_id, reward in rewards.items():
    participant = manager.get_participant(participant_id)
    claimed = participant.claim_rewards()
    # Submit reward claim to blockchain
```

## 🎯 PoUW Integration

The federated server implements **Proof of Useful Work (PoUW)** scoring:

### Fitness Calculation
```
Fitness = 0.4 × Quality + 0.3 × Timeliness + 0.3 × Honesty
```

- **Quality (40%)**: Model improvement accuracy
  - Measured by validation loss/accuracy
  - Higher quality = better model updates

- **Timeliness (30%)**: Submission deadline adherence
  - Measured by time from round start
  - Earlier submissions = higher score

- **Honesty (30%)**: Privacy compliance
  - Gradient verification
  - Data poisoning detection
  - Byzantine behavior checks

### Reward Multipliers
```python
# Fitness-based multiplier
if fitness >= 90:    multiplier = 2.0x  # Excellent
elif fitness >= 80:  multiplier = 1.5x  # Good
elif fitness >= 70:  multiplier = 1.0x  # Average
elif fitness >= 60:  multiplier = 0.5x  # Below average
else:                multiplier = 0.0x  # Poor (no reward)

# Reputation multiplier (0.5 - 1.5x)
reputation_multiplier = 0.5 + (reputation_score / 100)

# Total reward
reward = base_reward × fitness_multiplier × reputation_multiplier
```

### Slashing Conditions
Participants are slashed if:
- Fitness score < 50% for a round
- Byzantine behavior detected ≥ 3 times
- Honesty score < 50%
- Offline for > activity_timeout

## 📊 Metrics & Monitoring

### Time-Series Tracking
```python
# Loss over time
loss_history = tracker.get_loss_history(last_n=50)
for round_num, avg_loss in loss_history:
    print(f"Round {round_num}: {avg_loss:.4f}")

# Fitness over time  
fitness_history = tracker.get_fitness_history(last_n=50)
```

### Export Formats

**JSON Export**:
```python
tracker.export_to_json("metrics.json")  # All rounds
tracker.export_to_json("round_10.json", round_number=10)  # Specific round
```

**Prometheus Export**:
```python
metrics = tracker.export_to_prometheus()
# Use with Prometheus scraper
```

### Statistics
```python
# Aggregator stats
agg_stats = aggregator.get_statistics()
print(f"Total rounds: {agg_stats['total_rounds']}")
print(f"Avg fitness: {agg_stats['avg_fitness']:.2f}")

# Participant stats
part_stats = manager.get_statistics()
print(f"Active participants: {part_stats['active_participants']}")
print(f"Total rewards: {part_stats['total_rewards_distributed']:.2f} DALLA")

# Tracker stats
tracker_stats = tracker.get_statistics()
print(f"Total samples: {tracker_stats['total_samples']}")
```

## 🔐 Security Features

### Byzantine Fault Tolerance
- **Trimmed Median Aggregation**: Removes outliers
- **Reputation Scoring**: Tracks participant behavior
- **Automatic Suspension**: Detects malicious patterns
- **Gradient Verification**: Validates update integrity

### Privacy Preservation
- **Secure Aggregation**: Only aggregated model shared
- **Differential Privacy**: Optional noise injection
- **Local Training**: Data never leaves validator node
- **Encrypted Communication**: TLS for all transfers

## 🚀 Performance

### Aggregation Strategies
| Strategy | Speed | Robustness | Use Case |
|----------|-------|------------|----------|
| FedAvg (samples) | Fast | Medium | Trusted environment |
| FedAvg (fitness) | Fast | Medium | Quality-focused |
| FedAvg (hybrid) | Fast | Medium | Balanced (recommended) |
| Byzantine-robust | Slower | High | Adversarial environment |

### Scalability
- **Async Architecture**: Non-blocking operations
- **Concurrent Updates**: Handle 100+ participants
- **Memory Efficient**: Streaming aggregation
- **Horizontal Scaling**: Multi-server deployment

## 🔧 Configuration

### Server Settings
```python
# Aggregator
aggregator = FederatedAggregator(
    strategy=FedAvgStrategy(weighting="hybrid"),
    min_participants=3,          # Min for aggregation
    max_wait_time=300.0,        # 5 minutes max wait
)

# Participant Manager
manager = ParticipantManager(
    min_reputation=50.0,        # Min reputation to participate
    byzantine_threshold=3,      # Detections before suspension
    activity_timeout=600,       # 10 minutes before offline
)

# Metrics Tracker
tracker = MetricsTracker(
    max_history=1000,          # Max metrics per round
)
```

## 📚 References

- **FedAvg**: McMahan et al. (2017) - "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- **Byzantine-Robust**: Yin et al. (2018) - "Byzantine-Robust Distributed Learning"
- **PoUW**: BelizeChain Whitepaper - "Proof of Useful Work Consensus"

## 🧪 Testing

```bash
# Test aggregator
pytest nawal/tests/test_aggregator.py -v

# Test participant manager
pytest nawal/tests/test_participant_manager.py -v

# Test metrics tracker
pytest nawal/tests/test_metrics_tracker.py -v

# Integration tests
pytest nawal/tests/test_server_integration.py -v

# Full test suite
pytest nawal/tests/ -v --cov=nawal.server
```

## 📝 Next Steps

1. **Blockchain Integration**: Connect to Staking pallet
2. **API Server**: REST/WebSocket interface
3. **Dashboard**: Real-time monitoring UI
4. **Advanced Aggregation**: Personalized FL, hierarchical FL
5. **Auto-scaling**: Dynamic participant management

---

**Status**: ✅ Complete - Ready for integration testing  
**Version**: 1.0.0  
**Date**: October 2025
