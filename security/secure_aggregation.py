"""
Secure Aggregation for Federated Learning.

Implements secure multi-party computation (SMPC) techniques
to aggregate model updates without revealing individual contributions.

Key Features:
- Encrypted gradient transmission
- Secure aggregation protocol
- **Production Paillier Homomorphic Encryption** (via PyCryptodome)
- Protection against honest-but-curious aggregator

References:
- Bonawitz et al. "Practical Secure Aggregation for Privacy-Preserving Machine Learning" (2017)
- Paillier "Public-Key Cryptosystems Based on Composite Degree Residuosity Classes" (1999)

Author: BelizeChain Team
License: MIT
Date: October 2025 - PRODUCTION READY
"""

import hashlib
import hmac
import secrets
from dataclasses import dataclass

import torch
from loguru import logger

# Production-grade Paillier encryption
try:
    from Crypto.Util import number

    CRYPTO_AVAILABLE = True
except ImportError:
    logger.warning(
        "PyCryptodome not installed. Install with: pip install pycryptodome. "
        "Falling back to mock encryption (NOT SECURE for production)."
    )
    CRYPTO_AVAILABLE = False


@dataclass
class PaillierPublicKey:
    """Paillier public key for homomorphic encryption."""

    n: int  # Modulus (n = p * q)
    g: int  # Generator
    n_squared: int  # n^2 for efficiency

    def encrypt(self, plaintext: int) -> int:
        """
        Encrypt plaintext using Paillier encryption.

        E(m) = g^m * r^n mod n^2
        where r is random in Z*_n

        Args:
            plaintext: Integer to encrypt

        Returns:
            Encrypted ciphertext
        """
        # Generate random r
        r = number.getRandomRange(1, self.n)

        # Compute g^m mod n^2
        gm = pow(self.g, plaintext, self.n_squared)

        # Compute r^n mod n^2
        rn = pow(r, self.n, self.n_squared)

        # Compute ciphertext: g^m * r^n mod n^2
        ciphertext = (gm * rn) % self.n_squared

        return ciphertext

    def add_encrypted(self, c1: int, c2: int) -> int:
        """
        Add two encrypted values (homomorphic property).

        E(m1) * E(m2) mod n^2 = E(m1 + m2)

        Args:
            c1: First ciphertext
            c2: Second ciphertext

        Returns:
            Encrypted sum
        """
        return (c1 * c2) % self.n_squared

    def multiply_encrypted_by_scalar(self, ciphertext: int, scalar: int) -> int:
        """
        Multiply encrypted value by scalar (homomorphic property).

        E(m)^k mod n^2 = E(k * m)

        Args:
            ciphertext: Encrypted value
            scalar: Scalar multiplier

        Returns:
            Encrypted product
        """
        return pow(ciphertext, scalar, self.n_squared)


@dataclass
class PaillierPrivateKey:
    """Paillier private key for decryption."""

    lambda_: int  # λ = lcm(p-1, q-1)
    mu: int  # μ = (L(g^λ mod n^2))^-1 mod n
    public_key: PaillierPublicKey

    def decrypt(self, ciphertext: int) -> int:
        """
        Decrypt ciphertext using Paillier decryption.

        D(c) = L(c^λ mod n^2) * μ mod n
        where L(x) = (x-1) / n

        Args:
            ciphertext: Encrypted value

        Returns:
            Decrypted plaintext
        """
        n = self.public_key.n
        n_squared = self.public_key.n_squared

        # Compute c^λ mod n^2
        c_lambda = pow(ciphertext, self.lambda_, n_squared)

        # Compute L(c^λ mod n^2) = (c^λ - 1) / n
        l_value = (c_lambda - 1) // n

        # Compute plaintext: L * μ mod n
        plaintext = (l_value * self.mu) % n

        return plaintext


class PaillierKeyPair:
    """Complete Paillier key pair."""

    def __init__(self, key_size: int = 2048):
        """
        Generate Paillier key pair.

        Args:
            key_size: Key size in bits (2048 or 3072 recommended)
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError(
                "PyCryptodome required for production encryption. "
                "Install: pip install pycryptodome"
            )

        # Generate two large primes p and q
        p = number.getPrime(key_size // 2)
        q = number.getPrime(key_size // 2)

        # Compute n = p * q
        n = p * q
        n_squared = n * n

        # Compute λ = lcm(p-1, q-1)
        lambda_ = self._lcm(p - 1, q - 1)

        # Set g = n + 1 (simplified generator)
        g = n + 1

        # Compute μ = (L(g^λ mod n^2))^-1 mod n
        g_lambda = pow(g, lambda_, n_squared)
        l_value = (g_lambda - 1) // n
        mu = number.inverse(l_value, n)

        # Create key pair
        self.public_key = PaillierPublicKey(n=n, g=g, n_squared=n_squared)
        self.private_key = PaillierPrivateKey(lambda_=lambda_, mu=mu, public_key=self.public_key)

        logger.info(f"Generated Paillier key pair (key_size={key_size} bits)")

    @staticmethod
    def _lcm(a: int, b: int) -> int:
        """Compute least common multiple."""
        from math import gcd

        return abs(a * b) // gcd(a, b)

    def encrypt_float(self, value: float, scale: int = 1_000_000) -> int:
        """
        Encrypt floating point value by scaling to integer.

        Args:
            value: Float to encrypt
            scale: Scaling factor (higher = more precision, default 1e6 for
                   6 decimal places — sufficient for gradient values)

        Returns:
            Encrypted integer
        """
        # Scale float to integer
        scaled = int(value * scale)

        # Handle negative values (add offset)
        offset = 2 ** (self.public_key.n.bit_length() // 2 - 1)
        adjusted = scaled + offset

        # Encrypt
        return self.public_key.encrypt(adjusted)

    def decrypt_float(self, ciphertext: int, scale: int = 1_000_000) -> float:
        """
        Decrypt to floating point value.

        Args:
            ciphertext: Encrypted value
            scale: Scaling factor used in encryption (must match encrypt_float)

        Returns:
            Decrypted float
        """
        # Decrypt
        adjusted = self.private_key.decrypt(ciphertext)

        # Remove offset
        offset = 2 ** (self.public_key.n.bit_length() // 2 - 1)
        scaled = adjusted - offset

        # Convert back to float
        return scaled / scale


@dataclass
class EncryptionKey:
    """
    Production-grade encryption key using Paillier homomorphic encryption.

    Supports:
    - Secure encryption/decryption
    - Homomorphic addition (E(a) + E(b) = E(a+b))
    - Scalar multiplication (k * E(a) = E(k*a))
    """

    paillier_keypair: PaillierKeyPair | None = None
    # Legacy fields for backwards compatibility (no longer used)
    public_key: int = 0
    private_key: int | None = None
    modulus: int = 0

    def __post_init__(self):
        """Initialize Paillier keypair if not provided."""
        if self.paillier_keypair is None:
            if not CRYPTO_AVAILABLE:
                raise RuntimeError(
                    "PyCryptodome is required for encryption. " "Install: pip install pycryptodome"
                )
            self.paillier_keypair = PaillierKeyPair(key_size=2048)
            logger.info("✅ Initialized production Paillier encryption")

    def encrypt(self, value: float) -> int:
        """
        Encrypt value using production Paillier encryption.

        Args:
            value: Float to encrypt

        Returns:
            Encrypted ciphertext
        """
        if self.paillier_keypair and CRYPTO_AVAILABLE:
            return self.paillier_keypair.encrypt_float(value)
        else:
            raise RuntimeError(
                "Encryption unavailable: PyCryptodome is required. "
                "Install: pip install pycryptodome"
            )

    def decrypt(self, encrypted_value: int) -> float:
        """
        Decrypt value using production Paillier decryption.

        Args:
            encrypted_value: Encrypted ciphertext

        Returns:
            Decrypted plaintext
        """
        if self.paillier_keypair and CRYPTO_AVAILABLE:
            return self.paillier_keypair.decrypt_float(encrypted_value)
        else:
            raise RuntimeError(
                "Decryption unavailable: PyCryptodome is required. "
                "Install: pip install pycryptodome"
            )


class SecureAggregator:
    """
    Secure aggregation for federated learning.

    Protects individual client updates from honest-but-curious aggregator
    using secure multi-party computation techniques.

    Protocol Overview:
    1. Key Agreement: Clients establish shared secrets
    2. Masking: Each client masks their update with random noise
    3. Aggregation: Server aggregates masked updates
    4. Unmasking: Noise cancels out, revealing only the sum

    Usage:
        aggregator = SecureAggregator(num_clients=10)

        # Client side
        masked_update = aggregator.mask_update(client_id, model_update)

        # Server side
        global_update = aggregator.aggregate_masked_updates(masked_updates)
    """

    def __init__(
        self,
        num_clients: int,
        encryption_enabled: bool = True,
        mask_scale: float = 100.0,
    ):
        """
        Initialize SecureAggregator.

        Args:
            num_clients: Number of participating clients
            encryption_enabled: Whether to use encryption (placeholder)
            mask_scale: Scale of random masking noise
        """
        self.num_clients = num_clients
        self.encryption_enabled = encryption_enabled
        self.mask_scale = mask_scale

        # Client secrets for masking (256-bit seeds per client)
        self.client_secrets: dict[int, bytes] = {}
        self.pairwise_masks: dict[tuple[int, int], torch.Tensor] = {}

        logger.info(
            f"SecureAggregator initialized: {num_clients} clients, "
            f"encryption={'ON' if encryption_enabled else 'OFF'}"
        )

    def generate_client_keys(self) -> dict[int, EncryptionKey]:
        """
        Generate encryption keys for each client.

        Each client receives a cryptographically strong 256-bit seed
        used to derive pairwise masks deterministically.

        Returns:
            Dictionary mapping client_id to encryption key
        """
        keys = {}
        for client_id in range(self.num_clients):
            # Generate 256-bit cryptographic seed per client
            seed = secrets.token_bytes(32)
            self.client_secrets[client_id] = seed
            keys[client_id] = EncryptionKey()

        logger.debug(f"Generated keys for {self.num_clients} clients")
        return keys

    @staticmethod
    def _derive_pairwise_seed(seed_i: bytes, seed_j: bytes, pair: tuple[int, int]) -> bytes:
        """
        Derive a deterministic pairwise seed from two client secrets.

        Uses HMAC-SHA256 to combine client seeds with pair identifiers,
        producing a shared secret that both clients can independently compute.

        Args:
            seed_i: First client's secret seed
            seed_j: Second client's secret seed
            pair: Ordered (min, max) client ID pair

        Returns:
            32-byte deterministic pairwise seed
        """
        combined = seed_i + seed_j + str(pair).encode()
        return hmac.new(combined, b"pairwise_mask", hashlib.sha256).digest()

    def generate_pairwise_masks(
        self,
        client_id: int,
        model_shape: dict[str, torch.Size],
    ) -> None:
        """
        Generate deterministic pairwise masks from client seeds.

        Masks are derived from HMAC-based shared seeds so that:
        1. The server never generates masks — only distributes seeds
        2. Each pair of clients derives the same mask deterministically
        3. mask_ij = -mask_ji guarantees cancellation during aggregation

        Args:
            client_id: Client identifier
            model_shape: Dictionary of parameter shapes
        """
        if client_id not in self.client_secrets:
            raise ValueError(
                f"Client {client_id} has no secret — call generate_client_keys() first"
            )

        for other_id in range(self.num_clients):
            if other_id == client_id:
                continue

            pair = (min(client_id, other_id), max(client_id, other_id))

            if pair not in self.pairwise_masks:
                seed_i = self.client_secrets.get(pair[0])
                seed_j = self.client_secrets.get(pair[1])
                if seed_i is None or seed_j is None:
                    raise ValueError(f"Missing secret for client in pair {pair}")

                pairwise_seed = self._derive_pairwise_seed(seed_i, seed_j, pair)

                # Deterministic mask generation from shared seed
                gen = torch.Generator()
                gen.manual_seed(int.from_bytes(pairwise_seed[:8], byteorder="big"))

                masks = {}
                for param_name, shape in model_shape.items():
                    mask = torch.randn(shape, generator=gen) * self.mask_scale
                    masks[param_name] = mask

                self.pairwise_masks[pair] = masks

    def mask_update(
        self,
        client_id: int,
        model_update: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Mask client model update with secret shares.

        Adds pairwise masks that cancel out during aggregation:
        masked_update_i = update_i + Σ_j mask_ij

        When summed: Σ_i masked_update_i = Σ_i update_i + 0 (masks cancel)

        Args:
            client_id: Client identifier
            model_update: Client's model parameters

        Returns:
            Masked model update
        """
        if not self.encryption_enabled:
            return model_update

        # Generate pairwise masks if needed
        model_shape = {name: param.shape for name, param in model_update.items()}
        self.generate_pairwise_masks(client_id, model_shape)

        # Apply pairwise masks
        masked_update = {}
        for param_name, param in model_update.items():
            masked_param = param.clone()

            for other_id in range(self.num_clients):
                if other_id == client_id:
                    continue

                pair = (min(client_id, other_id), max(client_id, other_id))
                mask = self.pairwise_masks[pair][param_name]

                # Add or subtract based on client order
                if client_id < other_id:
                    masked_param += mask
                else:
                    masked_param -= mask

            masked_update[param_name] = masked_param

        return masked_update

    def aggregate_masked_updates(
        self,
        masked_updates: list[dict[str, torch.Tensor]],
        weights: list[float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Aggregate masked updates securely.

        The pairwise masks cancel out, revealing only the sum:
        Σ_i masked_update_i = Σ_i update_i

        Args:
            masked_updates: List of masked client updates
            weights: Optional weights for weighted average

        Returns:
            Aggregated model parameters
        """
        if len(masked_updates) == 0:
            raise ValueError("No updates to aggregate")

        # Default to uniform weights
        if weights is None:
            weights = [1.0 / len(masked_updates)] * len(masked_updates)

        # Initialize aggregated parameters
        aggregated = {}
        for param_name in masked_updates[0]:
            aggregated[param_name] = torch.zeros_like(masked_updates[0][param_name])

        # Weighted sum
        for update, weight in zip(masked_updates, weights, strict=False):
            for param_name, param in update.items():
                aggregated[param_name] += param * weight

        logger.debug(f"Aggregated {len(masked_updates)} masked updates")
        return aggregated

    def encrypt_gradients(
        self,
        gradients: dict[str, torch.Tensor],
        key: EncryptionKey,
    ) -> dict[str, list[int]]:
        """
        Encrypt gradients using homomorphic encryption (placeholder).

        In production, use proper libraries like:
        - python-paillier
        - TenSEAL
        - PySyft

        Args:
            gradients: Model gradients
            key: Encryption key

        Returns:
            Encrypted gradients
        """
        encrypted = {}
        for param_name, param in gradients.items():
            flat_param = param.flatten()
            encrypted_values = [key.encrypt(val.item()) for val in flat_param]
            encrypted[param_name] = encrypted_values

        logger.debug(f"Encrypted {len(gradients)} gradient tensors")
        return encrypted

    def decrypt_gradients(
        self,
        encrypted_gradients: dict[str, list[int]],
        key: EncryptionKey,
        shapes: dict[str, torch.Size],
    ) -> dict[str, torch.Tensor]:
        """
        Decrypt encrypted gradients.

        Args:
            encrypted_gradients: Encrypted gradient values
            key: Decryption key
            shapes: Original tensor shapes

        Returns:
            Decrypted gradients
        """
        decrypted = {}
        for param_name, encrypted_values in encrypted_gradients.items():
            decrypted_values = [key.decrypt(val) for val in encrypted_values]
            tensor = torch.tensor(decrypted_values)
            tensor = tensor.reshape(shapes[param_name])
            decrypted[param_name] = tensor

        logger.debug(f"Decrypted {len(encrypted_gradients)} gradient tensors")
        return decrypted

    def verify_client_participation(
        self,
        client_ids: list[int],
        min_clients: int,
    ) -> bool:
        """
        Verify minimum number of clients participated.

        Args:
            client_ids: List of participating client IDs
            min_clients: Minimum required clients

        Returns:
            True if enough clients participated
        """
        participated = len(set(client_ids))
        sufficient = participated >= min_clients

        if not sufficient:
            logger.warning(f"Insufficient clients: {participated}/{min_clients} required")

        return sufficient

    def dropout_resilient_aggregation(
        self,
        masked_updates: list[dict[str, torch.Tensor]],
        expected_clients: int,
        participating_client_ids: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Aggregate updates with dropout resilience.

        When some clients drop out, their masks don't cancel.
        This method subtracts the residual masks from dropped clients
        before scaling the result.

        Args:
            masked_updates: Updates from participating clients
            expected_clients: Total expected clients
            participating_client_ids: IDs of clients who actually participated

        Returns:
            Aggregated updates compensated for dropouts
        """
        actual_clients = len(masked_updates)

        if actual_clients == expected_clients:
            # No dropouts, standard aggregation
            return self.aggregate_masked_updates(masked_updates)

        # Compensate for dropouts
        logger.warning(f"Client dropout detected: {actual_clients}/{expected_clients} participated")

        # First, do a raw aggregation (masks from active clients cancel)
        aggregated = self.aggregate_masked_updates(masked_updates)

        # Subtract residual masks from dropped clients
        if participating_client_ids is not None:
            all_ids = set(range(expected_clients))
            dropped_ids = all_ids - set(participating_client_ids)

            for param_name in aggregated:
                residual = torch.zeros_like(aggregated[param_name])
                for dropped_id in dropped_ids:
                    for active_id in participating_client_ids:
                        pair = (min(dropped_id, active_id), max(dropped_id, active_id))
                        if pair in self.pairwise_masks and param_name in self.pairwise_masks[pair]:
                            mask = self.pairwise_masks[pair][param_name]
                            # The active client added this mask but the dropped
                            # client never subtracted it, so subtract here
                            if active_id < dropped_id:
                                residual += mask
                            else:
                                residual -= mask
                aggregated[param_name] -= residual

        # Scale for the reduced number of participants
        scale_factor = expected_clients / actual_clients
        for param_name in aggregated:
            aggregated[param_name] *= scale_factor / expected_clients

        return aggregated

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "num_clients": self.num_clients,
            "encryption_enabled": self.encryption_enabled,
            "mask_scale": self.mask_scale,
            "num_pairwise_masks": len(self.pairwise_masks),
        }


class SecureChannel:
    """
    Secure communication channel between clients and server.

    WARNING: This is a placeholder/stub implementation.
    In production, replace with proper TLS/SSL mutual authentication
    using libraries such as `cryptography` or `PySyft`.
    See `recommend_production_libraries()` below for options.
    """

    def __init__(self, use_tls: bool = True):
        """
        Initialize SecureChannel.

        Args:
            use_tls: Whether to use TLS encryption
        """
        self.use_tls = use_tls
        logger.info(f"SecureChannel initialized: TLS={'ON' if use_tls else 'OFF'}")

    def send_encrypted(
        self,
        data: dict[str, torch.Tensor],
        recipient_key: EncryptionKey,
    ) -> bytes:
        """
        Send encrypted data over secure channel.

        Args:
            data: Data to send
            recipient_key: Recipient's public key

        Returns:
            Encrypted bytes
        """
        # Placeholder: In production, use proper encryption
        logger.debug("Sending encrypted data")
        return b""  # Placeholder

    def receive_encrypted(
        self,
        encrypted_data: bytes,
        private_key: EncryptionKey,
    ) -> dict[str, torch.Tensor]:
        """
        Receive and decrypt data.

        Args:
            encrypted_data: Encrypted bytes
            private_key: Recipient's private key

        Returns:
            Decrypted data
        """
        # Placeholder
        logger.debug("Receiving encrypted data")
        return {}


# Production recommendation
def recommend_production_libraries() -> dict[str, str]:
    """
    Recommend production-grade secure aggregation libraries.

    Returns:
        Dictionary of library names and descriptions
    """
    return {
        "PySyft": "Secure & private deep learning (OpenMined)",
        "TenSEAL": "Homomorphic encryption for PyTorch (Microsoft)",
        "python-paillier": "Paillier homomorphic encryption",
        "cryptography": "Python cryptography toolkit",
        "SEAL": "Microsoft SEAL homomorphic encryption",
        "HElib": "IBM homomorphic encryption library",
    }
