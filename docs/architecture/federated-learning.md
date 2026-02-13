# 🤝 Federated Learning Architecture

**Version**: 1.1.0  
**Framework**: Flower (flwr) 1.6+  
**Last Updated**: February 13, 2026

---

## Overview

Nawal AI uses federated learning to train models across distributed validators without centralizing sensitive Belize corpus data.

```
┌────────────────────────────────────────────┐
│        Federated Learning Round            │
├────────────────────────────────────────────┤
│                                            │
│  1. Coordinator Announces Round            │
│     │                                      │
│     ▼                                      │
│  2. Validators Join (Stake DALLA)          │
│     │                                      │
│     ▼                                      │
│  3. Download Global Model                  │
│     │                                      │
│     ▼                                      │
│  4. Local Training on Private Data         │
│     │                                      │
│     ▼                                      │
│  5. Upload Model Deltas (Pakit)            │
│     │                                      │
│     ▼                                      │
│  6. Aggregation (FedAvg)                   │
│     │                                      │
│     ▼                                      │
│  7. Reward Distribution (PoUW)             │
│                                            │
└────────────────────────────────────────────┘
```

---

## Flower Framework Integration

### Client Implementation

```python
# client/genome_trainer.py

import flwr as fl
from client.model import NawalTransformer
from client.train import train, test

class GenomeTrainer(fl.client.NumPyClient):
    """Flower client for federated learning."""
    
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
    
    def get_parameters(self, config):
        """Return current model parameters."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters):
        """Update model with aggregated parameters."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train model on local data."""
        # Update model
        self.set_parameters(parameters)
        
        # Train
        train_loss, train_acc = train(
            self.model,
            self.trainloader,
            epochs=config["local_epochs"],
            learning_rate=config["learning_rate"],
        )
        
        # Return updated parameters and metrics
        return (
            self.get_parameters(config={}),
            len(self.trainloader.dataset),
            {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
            }
        )
    
    def evaluate(self, parameters, config):
        """Evaluate model on local test set."""
        # Update model
        self.set_parameters(parameters)
        
        # Evaluate
        loss, accuracy = test(self.model, self.testloader)
        
        # Return metrics
        return (
            float(loss),
            len(self.testloader.dataset),
            {
                "accuracy": accuracy,
            }
        )

# Start client
def start_client(server_address: str):
    """Start Flower client."""
    model = NawalTransformer.load_from_checkpoint("checkpoints/final_checkpoint.pt")
    trainloader, testloader = load_data()
    
    client = GenomeTrainer(model, trainloader, testloader)
    
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
    )

if __name__ == "__main__":
    start_client("localhost:8080")
```

---

## Aggregation Strategy

### FedAvg Implementation

**Federated Averaging (FedAvg)** - weighted average of client models

```python
# server/aggregator.py

from typing import List, Tuple
import numpy as np

def fedavg_aggregate(
    results: List[Tuple[np.ndarray, int]],
) -> np.ndarray:
    """
    Aggregate parameters using FedAvg.
    
    Args:
        results: List of (parameters, num_examples) tuples
    
    Returns:
        Aggregated parameters
    """
    # Calculate total examples
    total_examples = sum([num_examples for _, num_examples in results])
    
    # Weighted average
    aggregated = [
        np.sum([
            parameters[i] * num_examples / total_examples
            for parameters, num_examples in results
        ])
        for i in range(len(results[0][0]))
    ]
    
    return aggregated
```

### Custom Aggregation Strategy

```python
from flwr.server.strategy import Strategy
from typing import Dict, List, Optional, Tuple

class NawalStrategy(Strategy):
    """Custom aggregation strategy for Nawal AI."""
    
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results."""
        
        # Filter out low-quality updates
        quality_threshold = 0.7
        filtered_results = [
            (client, fit_res)
            for client, fit_res in results
            if fit_res.metrics.get("train_accuracy", 0) > quality_threshold
        ]
        
        # Byzantine-resistant aggregation
        aggregated = self.byzantine_robust_aggregate(filtered_results)
        
        # Compute aggregate metrics
        metrics = {
            "round": rnd,
            "num_clients": len(filtered_results),
            "avg_accuracy": np.mean([
                fit_res.metrics["train_accuracy"]
                for _, fit_res in filtered_results
            ]),
        }
        
        return aggregated, metrics
    
    def byzantine_robust_aggregate(self, results):
        """Byzantine-resistant aggregation using coordinate-wise median."""
        parameters_list = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]
        
        # Coordinate-wise median
        aggregated = [
            np.median([params[i] for params in parameters_list], axis=0)
            for i in range(len(parameters_list[0]))
        ]
        
        return ndarrays_to_parameters(aggregated)
```

---

## Proof-of-Useful-Work (PoUW)

### Reward Calculation

```python
# server/rewards.py

def calculate_pouw_reward(
    validator_id: str,
    model_quality: float,
    training_time: float,
    stake_amount: float,
    total_pool: float,
) -> float:
    """
    Calculate PoUW reward for validator.
    
    Args:
        validator_id: Validator account ID
        model_quality: Model accuracy (0-1)
        training_time: Training duration (seconds)
        stake_amount: Staked DALLA
        total_pool: Total reward pool for round
    
    Returns:
        Reward amount (DALLA)
    """
    # Base reward
    base_reward = total_pool * 0.6  # 60% of pool
    
    # Quality bonus
    quality_bonus = total_pool * 0.3 * model_quality
    
    # Stake bonus
    stake_bonus = total_pool * 0.1 * (stake_amount / 10000)
    
    # Time penalty (late submissions)
    time_penalty = max(0, 1 - (training_time - 3600) / 3600)
    
    total_reward = (base_reward + quality_bonus + stake_bonus) * time_penalty
    
    return total_reward
```

### Reward Distribution

```python
from blockchain import SubstrateClient

async def distribute_rewards(
    client: SubstrateClient,
    round_id: str,
    rewards: Dict[str, float],
):
    """Distribute PoUW rewards to validators."""
    
    for validator_id, reward_amount in rewards.items():
        # Convert to smallest unit (12 decimals)
        amount_planck = int(reward_amount * 1e12)
        
        # Submit reward transaction
        call = client.compose_call(
            call_module="Rewards",
            call_function="distribute_pouw_reward",
            call_params={
                "validator": validator_id,
                "round_id": round_id,
                "amount": amount_planck,
            }
        )
        
        receipt = await client.submit_extrinsic(call)
        
        if receipt.is_success:
            print(f"✅ Rewarded {validator_id}: {reward_amount:.2f} DALLA")
        else:
            print(f"❌ Failed to reward {validator_id}")
```

---

## Data Management

### Local Dataset

```python
# data/data_manager.py

from torch.utils.data import Dataset, DataLoader
import json

class BelizeCorpusDataset(Dataset):
    """Dataset for Belize legislative and cultural texts."""
    
    def __init__(self, data_path: str, tokenizer):
        self.tokenizer = tokenizer
        
        # Load dataset
        with open(data_path, 'r') as f:
            self.data = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        
        # Tokenize
        tokens = self.tokenizer.encode(text, max_length=2048)
        
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:],
        }

def load_data(batch_size=32):
    """Load train and test data loaders."""
    tokenizer = BelizeTokenizer.from_pretrained("tokenizers/belize_bpe_50k")
    
    train_dataset = BelizeCorpusDataset("data/belize_corpus/train.jsonl", tokenizer)
    test_dataset = BelizeCorpusDataset("data/belize_corpus/test.jsonl", tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader
```

### Data Privacy

**Differential Privacy** during training:

```python
from security.differential_privacy import DPMechanism

# Initialize DP mechanism
dp = DPMechanism(
    epsilon=1.0,      # Privacy budget
    delta=1e-5,       # Privacy parameter
    max_grad_norm=1.0 # Gradient clipping
)

# Training with DP
def train_with_dp(model, trainloader, epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        for batch in trainloader:
            # Forward pass
            loss = model(batch['input_ids'], labels=batch['labels']).loss
            
            # Backward pass
            loss.backward()
            
            # Add DP noise to gradients
            dp.add_noise_to_gradients(model)
            
            # Update
            optimizer.step()
            optimizer.zero_grad()
```

---

## Model Versioning

### Checkpoint Management

```python
# client/genome_trainer.py

import torch
from datetime import datetime

class CheckpointManager:
    """Manage model checkpoints during FL rounds."""
    
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
    
    def save_checkpoint(
        self,
        model,
        round_id: str,
        epoch: int,
        metrics: dict,
    ):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'round_id': round_id,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }
        
        filename = f"{self.checkpoint_dir}/checkpoint_round_{round_id}_epoch_{epoch}.pt"
        torch.save(checkpoint, filename)
        
        return filename
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        checkpoint = torch.load(filename)
        return checkpoint
    
    def get_latest_checkpoint(self):
        """Get the latest checkpoint."""
        import glob
        checkpoints = glob.glob(f"{self.checkpoint_dir}/checkpoint_*.pt")
        
        if not checkpoints:
            return None
        
        # Sort by modification time
        latest = max(checkpoints, key=os.path.getmtime)
        return latest
```

### Pakit Integration

```python
from storage.pakit_client import PakitClient

async def upload_model_delta(
    model_before,
    model_after,
    round_id: str,
    validator_id: str,
):
    """Upload model delta to Pakit."""
    
    # Compute delta
    delta = {}
    for name, param_before in model_before.named_parameters():
        param_after = dict(model_after.named_parameters())[name]
        delta[name] = param_after - param_before
    
    # Serialize
    import pickle
    delta_bytes = pickle.dumps(delta)
    
    # Upload to Pakit
    pakit = PakitClient()
    cid = await pakit.store(delta_bytes)
    
    print(f"✅ Model delta uploaded: {cid}")
    
    # Record on blockchain
    from blockchain import GenomeRegistry
    registry = GenomeRegistry()
    await registry.record_model_delta(
        round_id=round_id,
        validator_id=validator_id,
        cid=cid,
    )
    
    return cid
```

---

## Round Coordination

### Mesh Network Integration

```python
from blockchain import MeshNetworkClient

async def coordinate_fl_round(
    mesh: MeshNetworkClient,
    round_id: str,
    dataset: str,
    target_participants: int,
):
    """Coordinate federated learning round."""
    
    # Announce round
    await mesh.announce_fl_round(
        round_id=round_id,
        dataset_name=dataset,
        target_participants=target_participants,
        deadline=3600,  # 1 hour
        reward_pool=50000,  # 50,000 DALLA
    )
    
    # Wait for validators to join
    participants = []
    timeout = time.time() + 600  # 10 minutes
    
    while len(participants) < target_participants and time.time() < timeout:
        # Check for join messages
        await asyncio.sleep(1)
        # (handled by mesh message handler)
    
    print(f"✅ {len(participants)} validators joined")
    
    # Start training
    await start_training_round(participants, round_id)
```

---

## Performance Optimization

### Gradient Compression

```python
from typing import List
import numpy as np

def compress_gradients(gradients: List[np.ndarray], compression_ratio=0.1):
    """Compress gradients using top-k sparsification."""
    
    compressed = []
    
    for grad in gradients:
        # Flatten
        flat = grad.flatten()
        
        # Select top-k values
        k = int(len(flat) * compression_ratio)
        indices = np.argpartition(np.abs(flat), -k)[-k:]
        values = flat[indices]
        
        # Store sparse representation
        compressed.append({
            'indices': indices,
            'values': values,
            'shape': grad.shape,
        })
    
    return compressed

def decompress_gradients(compressed: List[dict]) -> List[np.ndarray]:
    """Decompress gradients."""
    
    gradients = []
    
    for comp in compressed:
        # Reconstruct sparse array
        flat = np.zeros(np.prod(comp['shape']))
        flat[comp['indices']] = comp['values']
        
        # Reshape
        grad = flat.reshape(comp['shape'])
        gradients.append(grad)
    
    return gradients
```

### Asynchronous Aggregation

```python
import asyncio
from collections import deque

class AsyncAggregator:
    """Asynchronous federated aggregation."""
    
    def __init__(self, staleness_threshold=5):
        self.global_model = None
        self.update_queue = deque()
        self.staleness_threshold = staleness_threshold
        self.round = 0
    
    async def aggregate_async(self):
        """Continuously aggregate updates."""
        
        while True:
            if len(self.update_queue) > 0:
                # Get update
                validator_id, model_delta, update_round = self.update_queue.popleft()
                
                # Check staleness
                staleness = self.round - update_round
                if staleness > self.staleness_threshold:
                    print(f"⚠️ Stale update from {validator_id}, skipping")
                    continue
                
                # Apply update with staleness penalty
                weight = 1.0 / (1 + staleness)
                self.apply_update(model_delta, weight)
                
                self.round += 1
            
            await asyncio.sleep(0.1)
    
    def apply_update(self, delta, weight):
        """Apply model update to global model."""
        for name, param in self.global_model.named_parameters():
            param.data += weight * delta[name]
```

---

## Monitoring & Metrics

### Training Metrics

```python
from monitoring.metrics_collector import MetricsCollector

metrics = MetricsCollector()

# Record training metrics
metrics.record("fl_round_duration", round_duration)
metrics.record("fl_participants", num_participants)
metrics.record("fl_average_accuracy", avg_accuracy)
metrics.record("fl_model_size_mb", model_size / 1024 / 1024)

# Export to Prometheus
from prometheus_client import Gauge

fl_participants_gauge = Gauge('nawal_fl_participants', 'Number of FL participants')
fl_participants_gauge.set(num_participants)
```

---

## Example: Complete FL Round

See [examples/complete_training.py](../../examples/complete_training.py) for a complete federated learning round example.

---

**Next Steps**:
- [Architecture Overview](overview.md)
- [Blockchain Integration](blockchain-integration.md)
- [API Reference](../reference/api-reference.md)
