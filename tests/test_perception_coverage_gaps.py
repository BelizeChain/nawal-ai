"""
Coverage gap tests for perception modules.

Covers:
- perception/visual_cortex.py (19 miss lines)
- perception/text_cortex.py (15 miss lines)
- perception/auditory_cortex.py (59 miss lines)
"""

import hashlib
import math
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List

import pytest
import numpy as np
import torch

from perception.text_cortex import TextCortex, _project, _l2_normalize
from perception.visual_cortex import VisualCortex, _check_stub_hash


# ===========================================================================
# text_cortex.py
# ===========================================================================


class TestTextCortexBertEmbed:
    """Cover _bert_embed path (lines 196-230)."""

    def test_bert_embed_success(self):
        """Successful BERT embedding path."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        # Tokenizer returns dict with attention_mask
        enc = {
            "input_ids": torch.zeros(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5),
        }
        # Make it dict-like but also have .to()
        mock_enc = MagicMock()
        mock_enc.__getitem__ = lambda self, key: enc[key]
        mock_enc.to.return_value = mock_enc
        mock_tokenizer.return_value = mock_enc

        # Model output
        hidden = torch.randn(1, 5, 768)
        mock_output = MagicMock()
        mock_output.last_hidden_state = hidden
        mock_model.return_value = mock_output
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model

        with patch("transformers.AutoTokenizer") as mock_at, \
             patch("transformers.AutoModel") as mock_am:
            mock_at.from_pretrained.return_value = mock_tokenizer
            mock_am.from_pretrained.return_value = mock_model

            tc = TextCortex(embed_dim=256, model_name="bert-base-uncased", device="cpu")
            result = tc.encode("hello world")
            assert len(result) == 256

    def test_bert_embed_load_failure_fallback(self):
        """BERT load fails → falls back to hash-ngram."""
        from transformers import AutoTokenizer as _AT
        with patch.object(_AT, "from_pretrained", side_effect=Exception("no model")):
            tc = TextCortex(embed_dim=256, model_name="bert-base-uncased", device="cpu")
            tc._load_model()
            # model_name should be set to None after failure
            assert tc.model_name is None
            # Should still produce embeddings via hash
            result = tc.encode("hello world")
            assert len(result) == 256

    def test_bert_embed_projection(self):
        """BERT output dim != embed_dim → projection applied."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        enc = {
            "input_ids": torch.zeros(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5),
        }
        mock_enc = MagicMock()
        mock_enc.__getitem__ = lambda self, key: enc[key]
        mock_enc.to.return_value = mock_enc
        mock_tokenizer.return_value = mock_enc

        # Model output with different size than embed_dim
        hidden = torch.randn(1, 5, 128)  # 128 != 256
        mock_output = MagicMock()
        mock_output.last_hidden_state = hidden
        mock_model.return_value = mock_output
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model

        with patch("transformers.AutoTokenizer") as mock_at, \
             patch("transformers.AutoModel") as mock_am:
            mock_at.from_pretrained.return_value = mock_tokenizer
            mock_am.from_pretrained.return_value = mock_model

            tc = TextCortex(embed_dim=256, model_name="bert-base-uncased", device="cpu")
            result = tc.encode("hello world")
            assert len(result) == 256


class TestTextCortexProject:
    """Cover _project utility."""

    def test_truncation(self):
        vec = list(range(10))
        result = _project(vec, 5)
        assert len(result) == 5
        assert result == [0, 1, 2, 3, 4]

    def test_expansion(self):
        vec = [1.0, 2.0, 3.0]
        result = _project(vec, 6)
        assert len(result) == 6


class TestTextCortexMisc:
    def test_preprocess_non_string(self):
        tc = TextCortex()
        result = tc.preprocess(42)
        assert result == "42"

    def test_encode_empty_string(self):
        tc = TextCortex()
        result = tc.encode("")
        assert result == [0.0] * tc.embed_dim

    def test_to_world_state(self):
        tc = TextCortex()
        embedding = [0.5] * 256
        ws = tc._to_world_state(embedding, "hello")
        assert ws.text_embedding == embedding


# ===========================================================================
# visual_cortex.py
# ===========================================================================


class TestVisualCortexPreprocess:
    """Cover preprocess branches."""

    def test_preprocess_string_path(self):
        """String input → open as PIL image."""
        vc = VisualCortex(model_name=None, stub_mode=True)
        # Mock PILImage.open
        with patch("perception.visual_cortex.PILImage") as mock_pil:
            mock_img = MagicMock()
            mock_img.convert.return_value = mock_img
            mock_pil.open.return_value = mock_img
            result = vc.preprocess("/fake/path.png")
            mock_pil.open.assert_called_once_with("/fake/path.png")

    def test_preprocess_numpy_array(self):
        """numpy array → PIL.fromarray."""
        vc = VisualCortex(model_name=None, stub_mode=True)
        arr = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        result = vc.preprocess(arr)
        # Should return PIL Image (converted via fromarray)

    def test_preprocess_bytes(self):
        """bytes input → PIL.open(BytesIO)."""
        vc = VisualCortex(model_name=None, stub_mode=True)
        # Create a minimal valid PNG bytes
        from PIL import Image as PILImage
        import io
        img = PILImage.new("RGB", (8, 8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        result = vc.preprocess(raw_bytes)
        # Should return PIL Image

    def test_preprocess_pil_image(self):
        """PIL Image input → pass through."""
        from PIL import Image as PILImage
        vc = VisualCortex(model_name=None, stub_mode=True)
        img = PILImage.new("RGB", (8, 8))
        result = vc.preprocess(img)
        assert result is img


class TestVisualCortexEncode:
    def test_encode_stub_pil(self):
        """Stub mode with PIL image."""
        from PIL import Image as PILImage
        vc = VisualCortex(model_name=None, stub_mode=True, embed_dim=64)
        img = PILImage.new("RGB", (8, 8), color=(255, 0, 0))
        result = vc.encode(img)
        assert len(result) == 64
        # Verify it's L2-normalized
        mag = math.sqrt(sum(x * x for x in result))
        assert abs(mag - 1.0) < 0.01

    def test_encode_stub_numpy(self):
        """Stub mode with numpy array."""
        vc = VisualCortex(model_name=None, stub_mode=True, embed_dim=64)
        arr = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        result = vc.encode(arr)
        assert len(result) == 64

    def test_encode_stub_bytes(self):
        """Stub mode with raw bytes."""
        vc = VisualCortex(model_name=None, stub_mode=True, embed_dim=64)
        # _stub_embed handles bytes directly via hashing, bypass preprocess
        result = vc._stub_embed(b"fakeimagebytes" * 100)
        assert len(result) == 64

    def test_encode_stub_unknown_type(self):
        """Stub mode with unknown type → uses id hash."""
        vc = VisualCortex(model_name=None, stub_mode=True, embed_dim=64)
        result = vc.encode(12345)
        assert len(result) == 64

    def test_encode_clip_load_failure(self):
        """CLIP load fails → activates stub mode."""
        from transformers import CLIPProcessor as _CP
        vc = VisualCortex(model_name="openai/clip-vit-base-patch32", stub_mode=False, embed_dim=64)
        # Force load failure by patching from_pretrained
        with patch.object(_CP, "from_pretrained", side_effect=Exception("no CLIP")):
            vc._load_model()  # Should catch exception and set stub_mode
        assert vc.stub_mode is True

    def test_clip_embed_success(self):
        """Successful CLIP encoding."""
        from PIL import Image as PILImage

        mock_processor = MagicMock()
        mock_model = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        feats = torch.randn(1, 512)
        mock_model.get_image_features.return_value = feats

        vc = VisualCortex(model_name="clip", embed_dim=512, stub_mode=False, device="cpu")
        vc._processor = mock_processor
        vc._model = mock_model
        vc._loaded = True
        # Don't set stub_mode here

        img = PILImage.new("RGB", (8, 8))
        result = vc._clip_embed(img)
        assert len(result) == 512


class TestVisualCortexToWorldState:
    def test_to_world_state(self):
        vc = VisualCortex(model_name=None, stub_mode=True)
        ws = vc._to_world_state([0.1] * 512, "img_input")
        assert ws.image_embedding == [0.1] * 512
        assert ws.metadata["cortex"] == "VisualCortex"


# ===========================================================================
# auditory_cortex.py
# ===========================================================================


class TestAuditoryCortexPreprocess:
    def test_preprocess_string_path(self):
        """String path → load audio file."""
        from perception.auditory_cortex import AuditoryCortex
        ac = AuditoryCortex(model_name=None, stub_mode=True)

        with patch.object(ac, "_load_audio_file", return_value=(np.zeros(1000), 16000)):
            with patch.object(ac, "_resample", return_value=np.zeros(1000)):
                result = ac.preprocess("/fake/audio.wav")
                assert isinstance(result, np.ndarray)

    def test_preprocess_bytes_soundfile(self):
        """Bytes input with soundfile available."""
        from perception.auditory_cortex import AuditoryCortex
        import perception.auditory_cortex as ac_mod

        ac = AuditoryCortex(model_name=None, stub_mode=True)

        mock_sf = MagicMock()
        mock_sf.read.return_value = (np.zeros(1000, dtype=np.float32), 16000)
        with patch.object(ac_mod, "SOUNDFILE_AVAILABLE", True), \
             patch.object(ac_mod, "sf", mock_sf, create=True):
            with patch.object(ac, "_resample", return_value=np.zeros(1000)):
                result = ac.preprocess(b"\x00" * 100)

    def test_preprocess_numpy_stereo(self):
        """Numpy 2D (stereo) → mean to mono."""
        from perception.auditory_cortex import AuditoryCortex
        ac = AuditoryCortex(model_name=None, stub_mode=True)
        arr = np.random.randn(2, 1000).astype(np.float32)  # (channels, samples)
        result = ac.preprocess(arr)
        assert result.ndim == 1

    def test_preprocess_numpy_mono(self):
        """Numpy 1D (mono) → pass through."""
        from perception.auditory_cortex import AuditoryCortex
        ac = AuditoryCortex(model_name=None, stub_mode=True)
        arr = np.random.randn(1000).astype(np.float32)
        result = ac.preprocess(arr)
        assert result.dtype == np.float32


class TestAuditoryCortexEncode:
    def test_encode_stub_numpy(self):
        from perception.auditory_cortex import AuditoryCortex
        ac = AuditoryCortex(model_name=None, stub_mode=True, embed_dim=64)
        arr = np.random.randn(1000).astype(np.float32)
        result = ac.encode(arr)
        assert len(result) == 64

    def test_encode_stub_bytes(self):
        from perception.auditory_cortex import AuditoryCortex
        ac = AuditoryCortex(model_name=None, stub_mode=True, embed_dim=64)
        result = ac.encode(b"\x00" * 100)
        assert len(result) == 64

    def test_encode_stub_list(self):
        from perception.auditory_cortex import AuditoryCortex
        ac = AuditoryCortex(model_name=None, stub_mode=True, embed_dim=64)
        result = ac.encode([0.1, 0.2, 0.3])
        assert len(result) == 64


class TestAuditoryCortexTranscribe:
    def test_transcribe_stub(self):
        from perception.auditory_cortex import AuditoryCortex
        ac = AuditoryCortex(model_name=None, stub_mode=True)
        result = ac.transcribe(np.zeros(1000))
        assert "audio input" in result

    def test_transcribe_model_load_failure(self):
        from perception.auditory_cortex import AuditoryCortex
        from transformers import WhisperProcessor as _WP
        ac = AuditoryCortex(model_name="openai/whisper-base", stub_mode=False)
        # Force load failure by patching from_pretrained
        with patch.object(_WP, "from_pretrained", side_effect=Exception("no whisper")):
            ac._load_model()
        # Should fallback to stub
        assert ac.stub_mode is True
        result = ac.transcribe(np.zeros(1000))
        assert "audio input" in result


class TestAuditoryCortexLoadModel:
    def test_load_model_success(self):
        from perception.auditory_cortex import AuditoryCortex

        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with patch("transformers.WhisperProcessor") as mock_wp, \
             patch("transformers.WhisperModel") as mock_wm:
            mock_wp.from_pretrained.return_value = mock_processor
            mock_wm.from_pretrained.return_value = mock_model

            ac = AuditoryCortex(model_name="whisper", stub_mode=False, device="cpu")
            ac._load_model()
            assert ac._loaded is True
            assert ac.stub_mode is False

    def test_load_model_already_loaded(self):
        from perception.auditory_cortex import AuditoryCortex
        ac = AuditoryCortex(model_name=None, stub_mode=True)
        ac._loaded = True
        ac._load_model()  # Should return early


class TestAuditoryCortexWhisperEmbed:
    def test_whisper_embed_success(self):
        from perception.auditory_cortex import AuditoryCortex

        mock_processor = MagicMock()
        mock_model = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        # Encoder output
        hidden = torch.randn(1, 100, 512)
        mock_enc_out = MagicMock()
        mock_enc_out.last_hidden_state = hidden
        mock_model.encoder.return_value = mock_enc_out

        ac = AuditoryCortex(model_name="whisper", embed_dim=512, stub_mode=False, device="cpu")
        ac._processor = mock_processor
        ac._model = mock_model
        ac._loaded = True

        audio = np.random.randn(16000).astype(np.float32)
        result = ac._whisper_embed(audio)
        assert len(result) == 512


class TestAuditoryCortexWhisperTranscribe:
    def test_whisper_transcribe_success(self):
        from perception.auditory_cortex import AuditoryCortex

        mock_processor = MagicMock()
        mock_model = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = lambda self, key: torch.zeros(1, 80, 3000)
        mock_processor.return_value = mock_inputs
        mock_processor.batch_decode.return_value = ["Hello world"]

        mock_model.generate.return_value = torch.zeros(1, 10, dtype=torch.long)

        ac = AuditoryCortex(model_name="whisper", stub_mode=False, device="cpu", language="en")
        ac._processor = mock_processor
        ac._model = mock_model
        ac._loaded = True

        audio = np.random.randn(16000).astype(np.float32)
        result = ac._whisper_transcribe(audio)
        assert result == "Hello world"


class TestAuditoryCortexAudioIO:
    def test_load_audio_file_soundfile(self):
        from perception.auditory_cortex import AuditoryCortex
        import perception.auditory_cortex as ac_mod
        ac = AuditoryCortex(model_name=None, stub_mode=True)

        mock_sf = MagicMock()
        mock_sf.read.return_value = (np.zeros(1000), 16000)
        with patch.object(ac_mod, "SOUNDFILE_AVAILABLE", True), \
             patch.object(ac_mod, "sf", mock_sf, create=True):
            audio, sr = ac._load_audio_file("/fake/audio.wav")
            assert sr == 16000

    def test_load_audio_file_torchaudio(self):
        from perception.auditory_cortex import AuditoryCortex
        import perception.auditory_cortex as ac_mod
        ac = AuditoryCortex(model_name=None, stub_mode=True)

        mock_ta = MagicMock()
        mock_ta.load.return_value = (torch.zeros(1, 1000), 16000)
        with patch.object(ac_mod, "SOUNDFILE_AVAILABLE", False), \
             patch.object(ac_mod, "TORCHAUDIO_AVAILABLE", True), \
             patch.object(ac_mod, "torchaudio", mock_ta, create=True):
            audio, sr = ac._load_audio_file("/fake/audio.wav")
            assert sr == 16000

    def test_load_audio_file_nothing_available(self):
        from perception.auditory_cortex import AuditoryCortex
        ac = AuditoryCortex(model_name=None, stub_mode=True)

        with patch("perception.auditory_cortex.SOUNDFILE_AVAILABLE", False), \
             patch("perception.auditory_cortex.TORCHAUDIO_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="Cannot load audio"):
                ac._load_audio_file("/fake/audio.wav")


class TestAuditoryCortexResample:
    def test_resample_same_rate(self):
        from perception.auditory_cortex import AuditoryCortex
        audio = np.random.randn(1000).astype(np.float32)
        result = AuditoryCortex._resample(audio, 16000, 16000)
        assert np.array_equal(result, audio)

    def test_resample_numpy_fallback(self):
        from perception.auditory_cortex import AuditoryCortex
        audio = np.random.randn(1000).astype(np.float32)
        with patch("perception.auditory_cortex.TORCHAUDIO_AVAILABLE", False):
            result = AuditoryCortex._resample(audio, 16000, 8000)
            assert len(result) > 0

    def test_resample_torchaudio(self):
        from perception.auditory_cortex import AuditoryCortex
        import perception.auditory_cortex as ac_mod
        audio = np.random.randn(1000).astype(np.float32)
        mock_ta = MagicMock()
        resampled = torch.randn(1, 500)
        mock_ta.functional.resample.return_value = resampled
        with patch.object(ac_mod, "TORCHAUDIO_AVAILABLE", True), \
             patch.object(ac_mod, "torchaudio", mock_ta, create=True):
            result = AuditoryCortex._resample(audio, 16000, 8000)
            assert result is not None


class TestAuditoryCortexToWorldState:
    def test_to_world_state(self):
        from perception.auditory_cortex import AuditoryCortex
        ac = AuditoryCortex(model_name=None, stub_mode=True)
        ws = ac._to_world_state([0.1] * 512, "audio_input")
        assert ws.audio_embedding == [0.1] * 512
        assert ws.metadata["cortex"] == "AuditoryCortex"
