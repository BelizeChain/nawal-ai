"""
Oracle Pipeline: Bridge between Oracle pallet IoT data and Nawal domain models.

This module implements the data flow:
1. Fetch IoT data from blockchain (Oracle pallet storage)
2. Route data to appropriate domain model based on device type
3. Run inference to generate predictions
4. Submit results back to Oracle pallet with quality metrics
5. Track PoUW contributions for reward distribution

Author: BelizeChain Core Team
Phase: 3 (Integration Layer)
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import torch
import numpy as np
from substrateinterface import SubstrateInterface, Keypair

from nawal.client.domain_models import (
    DomainModelFactory,
    ModelDomain,
    calculate_quality_score,
    prepare_oracle_submission,
)


class DeviceType(Enum):
    """
    Device types matching Oracle pallet Rust enum.
    Must stay in sync with pallets/oracle/src/types.rs
    """
    DRONE = 0
    PHONE = 1
    SENSOR = 2
    WEATHER_STATION = 3
    BUOY = 4
    CAMERA = 5


@dataclass
class IoTDeviceInfo:
    """Information about registered IoT device"""
    device_id: bytes
    device_type: DeviceType
    domain: ModelDomain
    operator: str  # AccountId (SS58 format)
    location: Optional[Tuple[float, float]]
    reputation_score: int
    total_submissions: int
    is_verified: bool
    registration_block: int


@dataclass
class IoTDataSubmission:
    """Raw data submission from IoT device"""
    device_id: bytes
    data: bytes
    feed_type: str
    location: Optional[Tuple[float, float]]
    timestamp: int
    quality_metrics: Dict[str, int]
    metadata: Optional[Dict[str, Any]]


class OracleDataFetcher:
    """
    Fetches IoT device data from Oracle pallet on-chain storage.
    
    Queries:
    - IoTDevices: Get registered device information
    - PendingSubmissions: Get unprocessed data submissions
    - OperatorStats: Get operator performance metrics
    """
    
    def __init__(self, substrate_url: str = "ws://127.0.0.1:9944"):
        """
        Initialize connection to BelizeChain node.
        
        Args:
            substrate_url: WebSocket URL of BelizeChain node
        """
        self.substrate = SubstrateInterface(
            url=substrate_url,
            ss58_format=42,  # BelizeChain SS58 format
            type_registry_preset='substrate-node-template',
        )
    
    def get_device_info(self, device_id: bytes) -> Optional[IoTDeviceInfo]:
        """
        Query IoTDevices storage for device information.
        
        Args:
            device_id: 32-byte device identifier
            
        Returns:
            Device info if registered, None otherwise
        """
        try:
            # Query Oracle::IoTDevices(device_id)
            result = self.substrate.query(
                module='Oracle',
                storage_function='IoTDevices',
                params=[device_id.hex()],
            )
            
            if result.value is None:
                return None
            
            # Parse storage result
            device_data = result.value
            return IoTDeviceInfo(
                device_id=device_id,
                device_type=DeviceType(device_data['device_type']),
                domain=ModelDomain(device_data['domain_index']),
                operator=device_data['operator'],
                location=(device_data['location']['lat'], device_data['location']['lon']) 
                    if device_data.get('location') else None,
                reputation_score=device_data['reputation_score'],
                total_submissions=device_data['total_submissions'],
                is_verified=device_data['is_verified'],
                registration_block=device_data['registration_block'],
            )
        
        except Exception as e:
            print(f"Error fetching device {device_id.hex()[:16]}...: {e}")
            return None
    
    def get_pending_submissions(
        self, 
        domain: Optional[ModelDomain] = None,
        limit: int = 100,
    ) -> List[IoTDataSubmission]:
        """
        Query PendingSubmissions storage for unprocessed data.
        
        Args:
            domain: Filter by domain (None = all domains)
            limit: Maximum submissions to fetch
            
        Returns:
            List of pending data submissions
        """
        submissions = []
        
        try:
            # Query Oracle::PendingSubmissions() - returns Vec<(DeviceId, SubmissionData)>
            result = self.substrate.query(
                module='Oracle',
                storage_function='PendingSubmissions',
            )
            
            if result.value is None:
                return submissions
            
            # Parse submissions
            for item in result.value[:limit]:
                device_id_hex, submission_data = item
                device_id = bytes.fromhex(device_id_hex)
                
                # Get device info to check domain
                device_info = self.get_device_info(device_id)
                if device_info is None:
                    continue
                
                # Filter by domain if specified
                if domain is not None and device_info.domain != domain:
                    continue
                
                # Parse submission data
                submissions.append(IoTDataSubmission(
                    device_id=device_id,
                    data=bytes(submission_data['data']),
                    feed_type=submission_data['feed_type'],
                    location=(submission_data['location']['lat'], submission_data['location']['lon'])
                        if submission_data.get('location') else None,
                    timestamp=submission_data['timestamp'],
                    quality_metrics=submission_data.get('quality_metrics', {}),
                    metadata=submission_data.get('metadata'),
                ))
        
        except Exception as e:
            print(f"Error fetching pending submissions: {e}")
        
        return submissions
    
    def get_operator_stats(self, operator: str) -> Dict[str, Any]:
        """
        Query OperatorStats storage for operator performance.
        
        Args:
            operator: Operator account (SS58 format)
            
        Returns:
            Dictionary with operator statistics
        """
        try:
            result = self.substrate.query(
                module='Oracle',
                storage_function='OperatorStats',
                params=[operator],
            )
            
            if result.value is None:
                return {
                    'total_devices': 0,
                    'active_devices': 0,
                    'total_submissions': 0,
                    'average_quality': 0,
                    'domain_breakdown': {},
                }
            
            return result.value
        
        except Exception as e:
            print(f"Error fetching operator stats for {operator}: {e}")
            return {}


class DataPreprocessor:
    """
    Routes IoT data to appropriate domain model and preprocesses it.
    
    Handles domain-specific data transformations:
    - AgriTech: Drone imagery → RGB tensors, sensor data → time series
    - Marine: Underwater imagery → color-corrected tensors, water quality sensors
    - Education: Student interactions → JSON → embedding
    - Tech: Infrastructure metrics → time series → normalized tensors
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize preprocessor.
        
        Args:
            device: PyTorch device ('cpu' or 'cuda')
        """
        self.device = device
        self.models: Dict[ModelDomain, Any] = {}
    
    def get_model(self, domain: ModelDomain):
        """Get or create domain model (cached)."""
        if domain not in self.models:
            self.models[domain] = DomainModelFactory.create_model(
                domain=domain,
                device=self.device,
            )
        return self.models[domain]
    
    def preprocess(
        self,
        submission: IoTDataSubmission,
        device_info: IoTDeviceInfo,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Preprocess IoT data for domain-specific model.
        
        Args:
            submission: Raw data submission
            device_info: Device metadata
            
        Returns:
            (input_tensor, model) - Ready for inference
        """
        # Get appropriate domain model
        model = self.get_model(device_info.domain)
        
        # Build raw_data dict for model preprocessing
        raw_data = {
            'data': submission.data,
            'feed_type': submission.feed_type,
            'location': submission.location,
            'timestamp': submission.timestamp,
            'metadata': submission.metadata or {},
        }
        
        # Domain-specific preprocessing
        input_tensor = model.preprocess_data(raw_data)
        
        return input_tensor, model


class ModelInferenceRunner:
    """
    Runs domain-specific model inference on preprocessed data.
    
    Tracks:
    - Inference latency
    - Model confidence
    - Feature extraction
    - Error handling
    """
    
    def __init__(self):
        self.inference_stats = {
            'total_inferences': 0,
            'total_time_ms': 0,
            'domain_breakdown': {},
        }
    
    def run_inference(
        self,
        model: Any,
        input_tensor: torch.Tensor,
        domain: ModelDomain,
    ) -> Dict[str, torch.Tensor]:
        """
        Run model inference and track performance.
        
        Args:
            model: Domain-specific model
            input_tensor: Preprocessed input
            domain: Model domain
            
        Returns:
            Predictions dictionary with domain-specific outputs
        """
        start_time = time.time()
        
        try:
            # Run inference
            with torch.no_grad():
                predictions = model.forward(input_tensor)
            
            # Track stats
            inference_time_ms = (time.time() - start_time) * 1000
            self.inference_stats['total_inferences'] += 1
            self.inference_stats['total_time_ms'] += inference_time_ms
            
            if domain.name not in self.inference_stats['domain_breakdown']:
                self.inference_stats['domain_breakdown'][domain.name] = {
                    'count': 0,
                    'total_time_ms': 0,
                }
            
            self.inference_stats['domain_breakdown'][domain.name]['count'] += 1
            self.inference_stats['domain_breakdown'][domain.name]['total_time_ms'] += inference_time_ms
            
            return predictions
        
        except Exception as e:
            print(f"Inference error for {domain.name}: {e}")
            # Return dummy predictions on error
            return {
                'predictions': torch.zeros(1),
                'confidence': torch.zeros(1),
                'features': torch.zeros(1, 128),
                'error': str(e),
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        stats = self.inference_stats.copy()
        
        if stats['total_inferences'] > 0:
            stats['average_time_ms'] = stats['total_time_ms'] / stats['total_inferences']
        
        return stats


class ResultSubmitter:
    """
    Submits model predictions back to Oracle pallet via extrinsics.
    
    Extrinsics used:
    - submit_iot_data: Submit predictions with quality metrics
    - claim_oracle_rewards: Claim accumulated PoUW rewards
    """
    
    def __init__(
        self,
        substrate_url: str = "ws://127.0.0.1:9944",
        keypair: Optional[Keypair] = None,
    ):
        """
        Initialize submitter.
        
        Args:
            substrate_url: WebSocket URL of BelizeChain node
            keypair: Operator keypair for signing transactions
        """
        self.substrate = SubstrateInterface(
            url=substrate_url,
            ss58_format=42,
            type_registry_preset='substrate-node-template',
        )
        self.keypair = keypair or Keypair.create_from_uri('//Alice')  # Dev default
    
    def submit_prediction(
        self,
        submission: IoTDataSubmission,
        predictions: Dict[str, torch.Tensor],
        device_info: IoTDeviceInfo,
    ) -> Optional[str]:
        """
        Submit model predictions to Oracle pallet.
        
        Args:
            submission: Original data submission
            predictions: Model predictions
            device_info: Device metadata
            
        Returns:
            Transaction hash if successful, None otherwise
        """
        try:
            # Prepare Oracle submission
            oracle_data = prepare_oracle_submission(
                domain=device_info.domain,
                device_id=submission.device_id,
                data=submission.data,
                predictions=predictions,
                quality_metrics=submission.quality_metrics,
            )
            
            # Build extrinsic call
            call = self.substrate.compose_call(
                call_module='Oracle',
                call_function='submit_iot_data',
                call_params={
                    'device_id': oracle_data['device_id'],
                    'data': oracle_data['data'],
                    'feed_type': oracle_data['feed_type'],
                    'quality_score': oracle_data['quality_score'],
                    'location': oracle_data.get('location'),
                },
            )
            
            # Create signed extrinsic
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call,
                keypair=self.keypair,
            )
            
            # Submit to blockchain
            receipt = self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=True,
            )
            
            if receipt.is_success:
                tx_hash = receipt.extrinsic_hash
                print(f"✅ Submitted prediction for device {submission.device_id.hex()[:16]}... (tx: {tx_hash})")
                return tx_hash
            else:
                print(f"❌ Submission failed: {receipt.error_message}")
                return None
        
        except Exception as e:
            print(f"Error submitting prediction: {e}")
            return None
    
    def claim_rewards(self, operator: Optional[str] = None) -> Optional[str]:
        """
        Claim accumulated Oracle rewards for operator.
        
        Args:
            operator: Operator account (defaults to keypair account)
            
        Returns:
            Transaction hash if successful
        """
        try:
            operator_account = operator or self.keypair.ss58_address
            
            call = self.substrate.compose_call(
                call_module='Oracle',
                call_function='claim_oracle_rewards',
                call_params={},
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
                print(f"✅ Claimed rewards for operator {operator_account}")
                return receipt.extrinsic_hash
            else:
                print(f"❌ Reward claim failed: {receipt.error_message}")
                return None
        
        except Exception as e:
            print(f"Error claiming rewards: {e}")
            return None


class OraclePipeline:
    """
    End-to-end pipeline: Oracle data → Nawal inference → Oracle submission.
    
    Usage:
        pipeline = OraclePipeline()
        await pipeline.process_pending_submissions(domain=ModelDomain.AGRITECH)
    """
    
    def __init__(
        self,
        substrate_url: str = "ws://127.0.0.1:9944",
        device: str = 'cpu',
        keypair: Optional[Keypair] = None,
    ):
        """
        Initialize complete pipeline.
        
        Args:
            substrate_url: BelizeChain node URL
            device: PyTorch device for inference
            keypair: Operator keypair for signing
        """
        self.fetcher = OracleDataFetcher(substrate_url)
        self.preprocessor = DataPreprocessor(device)
        self.runner = ModelInferenceRunner()
        self.submitter = ResultSubmitter(substrate_url, keypair)
    
    async def process_submission(
        self,
        submission: IoTDataSubmission,
    ) -> bool:
        """
        Process single IoT data submission.
        
        Args:
            submission: Data submission from Oracle pallet
            
        Returns:
            True if successful, False otherwise
        """
        # 1. Get device info
        device_info = self.fetcher.get_device_info(submission.device_id)
        if device_info is None:
            print(f"Device {submission.device_id.hex()[:16]}... not registered")
            return False
        
        # 2. Preprocess data
        try:
            input_tensor, model = self.preprocessor.preprocess(submission, device_info)
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return False
        
        # 3. Run inference
        predictions = self.runner.run_inference(model, input_tensor, device_info.domain)
        
        # 4. Submit results
        tx_hash = self.submitter.submit_prediction(submission, predictions, device_info)
        
        return tx_hash is not None
    
    async def process_pending_submissions(
        self,
        domain: Optional[ModelDomain] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Process all pending submissions for a domain.
        
        Args:
            domain: Filter by domain (None = all)
            limit: Maximum submissions to process
            
        Returns:
            Processing statistics
        """
        print(f"🔍 Fetching pending submissions (domain={domain.name if domain else 'ALL'})...")
        
        # Fetch pending submissions
        submissions = self.fetcher.get_pending_submissions(domain, limit)
        print(f"📦 Found {len(submissions)} pending submissions")
        
        # Process each submission
        results = {
            'total': len(submissions),
            'success': 0,
            'failed': 0,
            'by_domain': {},
        }
        
        for submission in submissions:
            success = await self.process_submission(submission)
            
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
        
        # Add inference stats
        results['inference_stats'] = self.runner.get_stats()
        
        print(f"\n✅ Processing complete: {results['success']}/{results['total']} successful")
        
        return results
    
    async def process_loop(
        self,
        domain: Optional[ModelDomain] = None,
        interval_seconds: int = 10,
    ):
        """
        Continuously process pending submissions.
        
        Args:
            domain: Filter by domain (None = all)
            interval_seconds: Polling interval
        """
        print(f"🔄 Starting Oracle pipeline loop (interval={interval_seconds}s)...")
        
        while True:
            try:
                await self.process_pending_submissions(domain)
                await asyncio.sleep(interval_seconds)
            except KeyboardInterrupt:
                print("\n⏹️ Pipeline stopped by user")
                break
            except Exception as e:
                print(f"Pipeline error: {e}")
                await asyncio.sleep(interval_seconds)


# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='BelizeChain Oracle Pipeline')
    parser.add_argument('--url', default='ws://127.0.0.1:9944', help='Substrate node URL')
    parser.add_argument('--domain', choices=['agritech', 'marine', 'education', 'tech', 'general'], 
                       help='Filter by domain')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='PyTorch device')
    parser.add_argument('--loop', action='store_true', help='Run continuous processing loop')
    parser.add_argument('--interval', type=int, default=10, help='Loop interval (seconds)')
    
    args = parser.parse_args()
    
    # Parse domain
    domain_map = {
        'agritech': ModelDomain.AGRITECH,
        'marine': ModelDomain.MARINE,
        'education': ModelDomain.EDUCATION,
        'tech': ModelDomain.TECH,
        'general': ModelDomain.GENERAL,
    }
    domain = domain_map.get(args.domain) if args.domain else None
    
    # Create pipeline
    pipeline = OraclePipeline(
        substrate_url=args.url,
        device=args.device,
    )
    
    # Run pipeline
    if args.loop:
        asyncio.run(pipeline.process_loop(domain, args.interval))
    else:
        asyncio.run(pipeline.process_pending_submissions(domain))
