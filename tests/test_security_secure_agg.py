"""
Direct tests for security/secure_aggregation.py classes.

Covers:
  - PaillierKeyPair   — key generation, encrypt/decrypt roundtrip, float support
  - PaillierPublicKey — encrypt, add_encrypted, multiply_encrypted_by_scalar
  - PaillierPrivateKey — decrypt
  - EncryptionKey     — encrypt/decrypt with injected keypair
  - SecureAggregator  — masking protocol, aggregation, verification, dropout
  - SecureChannel     — stub send/receive
  - recommend_production_libraries — utility

NOTE: Paillier tests use 512-bit keys for speed. Production uses 2048-bit.
"""
from __future__ import annotations

from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
import torch

from security.secure_aggregation import (
    CRYPTO_AVAILABLE,
    EncryptionKey,
    PaillierKeyPair,
    SecureAggregator,
    SecureChannel,
    recommend_production_libraries,
)


# ---------------------------------------------------------------------------
# Session-scoped small Paillier keypair (512-bit, generated once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def small_keypair():
    """512-bit Paillier keypair — fast to generate, reused across all tests."""
    if not CRYPTO_AVAILABLE:
        pytest.skip("pycryptodome not installed")
    return PaillierKeyPair(key_size=512)


@pytest.fixture
def enc_key(small_keypair):
    """EncryptionKey backed by the 512-bit keypair (no auto-generation)."""
    return EncryptionKey(paillier_keypair=small_keypair)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_update(seed: int = 0) -> Dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {"w": torch.randn(3, 3), "b": torch.randn(3)}


def _make_aggregator(n: int = 3, encrypt: bool = False) -> SecureAggregator:
    return SecureAggregator(num_clients=n, encryption_enabled=encrypt)


# ---------------------------------------------------------------------------
# PaillierKeyPair
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="pycryptodome required")
class TestPaillierKeyPair:

    def test_generates_public_and_private_keys(self, small_keypair):
        assert small_keypair.public_key is not None
        assert small_keypair.private_key is not None

    def test_public_key_has_n_and_g(self, small_keypair):
        pk = small_keypair.public_key
        assert pk.n > 0
        assert pk.g > 0
        assert pk.n_squared == pk.n ** 2

    def test_private_key_links_to_public(self, small_keypair):
        sk = small_keypair.private_key
        assert sk.public_key is small_keypair.public_key

    def test_encrypt_decrypt_small_int(self, small_keypair):
        plaintext = 42
        ct = small_keypair.public_key.encrypt(plaintext)
        pt = small_keypair.private_key.decrypt(ct)
        assert pt == plaintext

    def test_encrypt_decrypt_zero(self, small_keypair):
        ct = small_keypair.public_key.encrypt(0)
        pt = small_keypair.private_key.decrypt(ct)
        assert pt == 0

    def test_encrypt_decrypt_roundtrip_float(self, small_keypair):
        value = 3.14159
        ct = small_keypair.encrypt_float(value)
        recovered = small_keypair.decrypt_float(ct)
        assert abs(recovered - value) < 0.01  # ~4 decimal places of precision

    def test_encrypt_float_negative(self, small_keypair):
        value = -1.5
        ct = small_keypair.encrypt_float(value)
        recovered = small_keypair.decrypt_float(ct)
        assert abs(recovered - value) < 0.01

    def test_encrypt_float_near_zero(self, small_keypair):
        value = 0.001
        ct = small_keypair.encrypt_float(value)
        recovered = small_keypair.decrypt_float(ct)
        assert abs(recovered - value) < 0.001


# ---------------------------------------------------------------------------
# PaillierPublicKey homomorphic operations
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="pycryptodome required")
class TestPaillierHomomorphic:

    def test_add_encrypted_homomorphism(self, small_keypair):
        """E(a) * E(b) mod n^2 = E(a+b)"""
        pk = small_keypair.public_key
        sk = small_keypair.private_key
        a, b = 5, 7
        ea = pk.encrypt(a)
        eb = pk.encrypt(b)
        eab = pk.add_encrypted(ea, eb)
        assert sk.decrypt(eab) == a + b

    def test_multiply_by_scalar(self, small_keypair):
        """E(m)^k = E(k*m)"""
        pk = small_keypair.public_key
        sk = small_keypair.private_key
        m, k = 3, 4
        em = pk.encrypt(m)
        ekm = pk.multiply_encrypted_by_scalar(em, k)
        assert sk.decrypt(ekm) == m * k

    def test_add_encrypted_commutative(self, small_keypair):
        pk = small_keypair.public_key
        sk = small_keypair.private_key
        a, b = 11, 13
        ea, eb = pk.encrypt(a), pk.encrypt(b)
        assert sk.decrypt(pk.add_encrypted(ea, eb)) == sk.decrypt(pk.add_encrypted(eb, ea))


# ---------------------------------------------------------------------------
# EncryptionKey
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="pycryptodome required")
class TestEncryptionKey:

    def test_encrypt_returns_int(self, enc_key):
        result = enc_key.encrypt(0.5)
        assert isinstance(result, int)

    def test_decrypt_roundtrip(self, enc_key):
        value = 0.123
        ct = enc_key.encrypt(value)
        pt = enc_key.decrypt(ct)
        assert abs(pt - value) < 0.01

    def test_encrypt_decrypt_negative(self, enc_key):
        value = -0.75
        ct = enc_key.encrypt(value)
        pt = enc_key.decrypt(ct)
        assert abs(pt - value) < 0.01


# ---------------------------------------------------------------------------
# SecureAggregator — init
# ---------------------------------------------------------------------------

class TestSecureAggregatorInit:

    def test_stores_num_clients(self):
        ag = SecureAggregator(num_clients=5)
        assert ag.num_clients == 5

    def test_encryption_enabled_flag(self):
        ag = SecureAggregator(num_clients=3, encryption_enabled=False)
        assert ag.encryption_enabled is False

    def test_mask_scale_stored(self):
        ag = SecureAggregator(num_clients=2, mask_scale=50.0)
        assert ag.mask_scale == 50.0

    def test_empty_secrets_on_init(self):
        assert SecureAggregator(num_clients=3).client_secrets == {}

    def test_empty_pairwise_masks_on_init(self):
        assert SecureAggregator(num_clients=3).pairwise_masks == {}


# ---------------------------------------------------------------------------
# SecureAggregator — generate_client_keys (mocked to avoid slow Paillier)
# ---------------------------------------------------------------------------

class TestGenerateClientKeys:

    def test_generates_one_secret_per_client(self):
        ag = SecureAggregator(num_clients=4)
        with patch("security.secure_aggregation.EncryptionKey", MagicMock):
            keys = ag.generate_client_keys()
        assert len(ag.client_secrets) == 4

    def test_returns_dict_keyed_by_client_id(self):
        ag = SecureAggregator(num_clients=3)
        with patch("security.secure_aggregation.EncryptionKey", MagicMock):
            keys = ag.generate_client_keys()
        assert set(keys.keys()) == {0, 1, 2}

    def test_secrets_are_32_bytes(self):
        ag = SecureAggregator(num_clients=2)
        with patch("security.secure_aggregation.EncryptionKey", MagicMock):
            ag.generate_client_keys()
        for secret in ag.client_secrets.values():
            assert len(secret) == 32


# ---------------------------------------------------------------------------
# SecureAggregator — masking protocol (encryption_enabled=True)
# ---------------------------------------------------------------------------

class TestMaskingProtocol:

    def _aggregator_with_secrets(self, n: int = 2) -> SecureAggregator:
        """Return an aggregator with manually injected secrets (no slow key gen)."""
        import secrets as _secrets
        ag = SecureAggregator(num_clients=n, encryption_enabled=True)
        for i in range(n):
            ag.client_secrets[i] = _secrets.token_bytes(32)
        return ag

    def test_mask_update_returns_same_keys(self):
        ag = self._aggregator_with_secrets(2)
        u = _make_update(0)
        masked = ag.mask_update(0, u)
        assert set(masked.keys()) == set(u.keys())

    def test_mask_update_changes_values(self):
        ag = self._aggregator_with_secrets(2)
        u = _make_update(0)
        masked = ag.mask_update(0, u)
        assert not torch.allclose(masked["w"], u["w"])

    def test_masks_cancel_on_aggregation(self):
        """
        With 2 clients, mask_ij = -mask_ji, so the sum of masked updates
        equals the sum of original updates.
        """
        ag = self._aggregator_with_secrets(2)
        u0 = _make_update(0)
        u1 = _make_update(1)
        m0 = ag.mask_update(0, u0)
        m1 = ag.mask_update(1, u1)

        # Aggregate masked → uniform weights sum = (m0+m1)/2
        agg_masked = ag.aggregate_masked_updates([m0, m1])
        # Expected: (u0+u1)/2
        agg_plain = ag.aggregate_masked_updates([u0, u1],
                                                 weights=[0.5, 0.5])
        # Numerically they should be equal because masks cancel
        assert torch.allclose(agg_masked["w"], agg_plain["w"], atol=1e-4)

    def test_mask_update_disabled_passthrough(self):
        ag = SecureAggregator(num_clients=2, encryption_enabled=False)
        u = _make_update(0)
        masked = ag.mask_update(0, u)
        assert masked is u  # same object returned directly

    def test_mask_update_no_secret_raises(self):
        ag = SecureAggregator(num_clients=2, encryption_enabled=True)
        # client_secrets is empty
        with pytest.raises((ValueError, KeyError)):
            ag.mask_update(0, _make_update(0))


# ---------------------------------------------------------------------------
# SecureAggregator — aggregate_masked_updates
# ---------------------------------------------------------------------------

class TestAggregateUpdates:

    def test_empty_list_raises(self):
        ag = _make_aggregator()
        with pytest.raises(ValueError, match="No updates to aggregate"):
            ag.aggregate_masked_updates([])

    def test_single_update_passthrough(self):
        ag = _make_aggregator()
        u = _make_update(0)
        result = ag.aggregate_masked_updates([u])
        assert torch.allclose(result["w"], u["w"])

    def test_average_uniform_weights(self):
        ag = _make_aggregator()
        updates = [{"w": torch.tensor([0.0])}, {"w": torch.tensor([4.0])}]
        result = ag.aggregate_masked_updates(updates)
        assert torch.allclose(result["w"], torch.tensor([2.0]))

    def test_custom_weights_applied(self):
        ag = _make_aggregator()
        updates = [{"w": torch.tensor([0.0])}, {"w": torch.tensor([10.0])}]
        result = ag.aggregate_masked_updates(updates, weights=[0.9, 0.1])
        assert torch.allclose(result["w"], torch.tensor([1.0]))

    def test_output_shape_matches_input(self):
        ag = _make_aggregator()
        updates = [_make_update(i) for i in range(3)]
        result = ag.aggregate_masked_updates(updates)
        assert result["w"].shape == updates[0]["w"].shape


# ---------------------------------------------------------------------------
# SecureAggregator — verify_client_participation
# ---------------------------------------------------------------------------

class TestVerifyParticipation:

    def test_enough_clients_returns_true(self):
        ag = _make_aggregator(5)
        assert ag.verify_client_participation([0, 1, 2, 3, 4], min_clients=5) is True

    def test_too_few_clients_returns_false(self):
        ag = _make_aggregator(5)
        assert ag.verify_client_participation([0, 1], min_clients=3) is False

    def test_exactly_min_clients_returns_true(self):
        ag = _make_aggregator(5)
        assert ag.verify_client_participation([0, 1, 2], min_clients=3) is True

    def test_duplicates_not_double_counted(self):
        ag = _make_aggregator(5)
        # [0, 0, 1] → set = {0, 1} → 2 unique
        assert ag.verify_client_participation([0, 0, 1], min_clients=3) is False


# ---------------------------------------------------------------------------
# SecureAggregator — dropout_resilient_aggregation
# ---------------------------------------------------------------------------

class TestDropoutResilience:

    def test_no_dropout_same_as_regular(self):
        ag = _make_aggregator(3, encrypt=False)
        updates = [_make_update(i) for i in range(3)]
        result_normal = ag.aggregate_masked_updates(updates)
        result_dropout = ag.dropout_resilient_aggregation(updates, expected_clients=3)
        assert torch.allclose(result_normal["w"], result_dropout["w"], atol=1e-4)

    def test_with_dropout_returns_dict(self):
        ag = _make_aggregator(3, encrypt=False)
        updates = [_make_update(i) for i in range(2)]  # 1 dropout
        result = ag.dropout_resilient_aggregation(
            updates, expected_clients=3, participating_client_ids=[0, 1]
        )
        assert "w" in result and "b" in result

    def test_to_dict_keys(self):
        ag = _make_aggregator(3)
        d = ag.to_dict()
        assert set(d.keys()) == {"num_clients", "encryption_enabled", "mask_scale", "num_pairwise_masks"}


# ---------------------------------------------------------------------------
# SecureAggregator — encrypt/decrypt gradients
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="pycryptodome required")
class TestEncryptDecryptGradients:

    def test_encrypt_then_decrypt_small_tensor(self, enc_key):
        ag = _make_aggregator(2)
        grads = {"w": torch.tensor([0.1, -0.2, 0.3])}
        shapes = {"w": grads["w"].shape}
        encrypted = ag.encrypt_gradients(grads, enc_key)
        assert "w" in encrypted
        decrypted = ag.decrypt_gradients(encrypted, enc_key, shapes)
        assert torch.allclose(decrypted["w"], grads["w"], atol=0.01)


# ---------------------------------------------------------------------------
# SecureChannel (stub)
# ---------------------------------------------------------------------------

class TestSecureChannel:

    def test_init_tls_on(self):
        ch = SecureChannel(use_tls=True)
        assert ch.use_tls is True

    def test_init_tls_off(self):
        ch = SecureChannel(use_tls=False)
        assert ch.use_tls is False

    def test_send_returns_bytes(self, enc_key):
        ch = SecureChannel()
        result = ch.send_encrypted({"w": torch.zeros(2)}, enc_key)
        assert isinstance(result, bytes)

    def test_receive_returns_dict(self, enc_key):
        ch = SecureChannel()
        result = ch.receive_encrypted(b"", enc_key)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# recommend_production_libraries
# ---------------------------------------------------------------------------

class TestRecommendProductionLibraries:

    def test_returns_dict(self):
        result = recommend_production_libraries()
        assert isinstance(result, dict)

    def test_contains_pysyft(self):
        result = recommend_production_libraries()
        assert "PySyft" in result

    def test_contains_several_libs(self):
        result = recommend_production_libraries()
        assert len(result) >= 4

    def test_all_values_are_strings(self):
        result = recommend_production_libraries()
        assert all(isinstance(v, str) for v in result.values())
