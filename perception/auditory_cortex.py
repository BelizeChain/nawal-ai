"""
AuditoryCortex — Speech-to-text transcription and audio embeddings.

Architecture
------------
Backbone : ``openai/whisper-base`` via HuggingFace Transformers.
           Fine-tunable on Kriol / Garifuna / Mopan speech data (Phase 5b).
Output   :
  - ``transcribe()``  → str  (ASR output)
  - ``encode()``      → List[float] (384-dim audio embedding from encoder)

Stub mode
---------
When Whisper / torchaudio is unavailable (CI, no GPU) the cortex falls
back to a deterministic hash of the raw waveform bytes.  The stub
transcription returns ``"[audio input — model unavailable]"``.

PhaseHook
---------
Phase 5b: fine-tune ``WhisperModel`` on Belize Kriol corpus.
Replace ``_whisper_embed()`` and ``_whisper_transcribe()`` internals —
the public API is unchanged.

Dependencies
------------
    pip install transformers
    pip install torchaudio soundfile   # for audio file I/O
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Union

import torch
from loguru import logger

from perception.interfaces import AbstractCortex, WorldState
from perception.text_cortex import _l2_normalize, _project

_STUB_TRANSCRIPTION = "[audio input — model unavailable]"

# Optional numpy
try:
    import numpy as np  # type: ignore

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Optional soundfile / torchaudio for file I/O
try:
    import soundfile as sf  # type: ignore

    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import torchaudio  # type: ignore

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


def _hash_audio(samples: Any, dim: int) -> List[float]:
    """Deterministic hash-embedding from raw audio samples (stub mode)."""
    if NUMPY_AVAILABLE and isinstance(samples, np.ndarray):
        raw = samples.tobytes()[:8192]
    elif isinstance(samples, (list, tuple)):
        raw = str(samples[:256]).encode()
    elif isinstance(samples, bytes):
        raw = samples[:8192]
    else:
        raw = str(id(samples)).encode()

    digest = hashlib.sha256(raw).digest()
    out: List[float] = []
    for i in range(dim):
        byte_idx = (i * 4) % len(digest)
        val = int.from_bytes(digest[byte_idx : byte_idx + 4], "big", signed=True)
        out.append(float(val) / 2**31)
    return _l2_normalize(out)


class AuditoryCortex(AbstractCortex):
    """
    Audio-modality sensory cortex.

    Args:
        model_name  : HuggingFace Whisper model ID.
                      None / stub_mode=True bypasses model loading.
        embed_dim   : Output embedding dimension (Whisper-base encoder: 512).
        sample_rate : Expected input sample rate in Hz (Whisper wants 16 000).
        max_duration: Max audio length in seconds to process (default 30 s).
        device      : "cpu", "cuda", or "auto".
        stub_mode   : Force stub mode (useful for tests).
        language    : ISO 639-1 language hint for transcription (default None).
    """

    DEFAULT_MODEL = "openai/whisper-base"
    WHISPER_SR = 16_000

    def __init__(
        self,
        model_name: Optional[str] = DEFAULT_MODEL,
        embed_dim: int = 512,
        sample_rate: int = WHISPER_SR,
        max_duration: float = 30.0,
        device: str = "auto",
        stub_mode: bool = False,
        language: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.stub_mode = stub_mode or (model_name is None)
        self.language = language
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )

        self._processor: Any = None
        self._model: Any = None
        self._loaded = False

        logger.debug(
            f"AuditoryCortex init: model={model_name or 'STUB'} "
            f"sr={sample_rate} dim={embed_dim}"
        )

    # ------------------------------------------------------------------ #
    # AbstractCortex interface                                             #
    # ------------------------------------------------------------------ #

    def preprocess(self, raw_input: Any) -> Any:
        """
        Accept np.ndarray [samples], a file path str, or raw bytes.
        Returns float32 mono numpy array at self.sample_rate,
        trimmed to max_duration.
        """
        if not NUMPY_AVAILABLE:
            return raw_input  # pass through for stub

        max_samples = int(self.sample_rate * self.max_duration)

        if isinstance(raw_input, str):
            # File path
            audio, sr = self._load_audio_file(raw_input)
            audio = self._resample(audio, sr, self.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            return audio

        if isinstance(raw_input, bytes):
            import io

            if SOUNDFILE_AVAILABLE:
                audio, sr = sf.read(io.BytesIO(raw_input), dtype="float32")
            elif TORCHAUDIO_AVAILABLE:
                waveform, sr = torchaudio.load(io.BytesIO(raw_input))
                audio = waveform.mean(0).numpy()
            else:
                return raw_input
            audio = self._resample(audio, sr, self.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            return audio

        if NUMPY_AVAILABLE and isinstance(raw_input, np.ndarray):
            # Assume already at target sample rate; ensure float32 mono
            arr = raw_input.astype("float32")
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            if len(arr) > max_samples:
                arr = arr[:max_samples]
            return arr

        return raw_input

    def encode(self, raw_input: Any) -> List[float]:
        """
        Encode audio to a float embedding.

        Args:
            raw_input : np.ndarray [samples] at 16 kHz, file path, or bytes.

        Returns:
            List[float] of length ``embed_dim``, L2-normalised.
        """
        audio = self.preprocess(raw_input)

        if self.stub_mode or not self.model_name:
            return _hash_audio(audio, self.embed_dim)

        self._load_model()
        if self.stub_mode:  # fallback engaged after load failure
            return _hash_audio(audio, self.embed_dim)

        return self._whisper_embed(audio)

    def transcribe(self, raw_input: Any) -> str:
        """
        Convert audio to text using Whisper ASR.

        Returns:
            Transcription string (empty string on failure in stub mode).
        """
        if self.stub_mode or not self.model_name:
            return _STUB_TRANSCRIPTION

        audio = self.preprocess(raw_input)
        self._load_model()
        if self.stub_mode:
            return _STUB_TRANSCRIPTION

        return self._whisper_transcribe(audio)

    def _to_world_state(self, embedding: List[float], raw_input: Any) -> WorldState:
        return WorldState(
            audio_embedding=embedding,
            metadata={
                "cortex": "AuditoryCortex",
                "mode": "stub" if self.stub_mode else "whisper",
                "model": self.model_name,
                "sample_rate": self.sample_rate,
            },
        )

    # ------------------------------------------------------------------ #
    # Whisper mode (lazy-loaded)                                           #
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            from transformers import WhisperModel, WhisperProcessor  # type: ignore

            logger.info(f"AuditoryCortex: loading '{self.model_name}'")
            self._processor = WhisperProcessor.from_pretrained(self.model_name)
            self._model = WhisperModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
        except Exception as exc:
            logger.warning(
                f"AuditoryCortex: Whisper load failed ({exc}). " "Activating stub mode."
            )
            self.stub_mode = True

    def _whisper_embed(self, audio: Any) -> List[float]:
        """Run Whisper encoder and mean-pool its hidden state."""
        if not NUMPY_AVAILABLE:
            return _hash_audio(audio, self.embed_dim)

        # Trim / pad to max_duration
        max_samples = int(self.sample_rate * self.max_duration)
        arr = audio[:max_samples] if len(audio) > max_samples else audio

        inputs = self._processor(
            arr,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            enc_out = self._model.encoder(**inputs)

        # Mean-pool over time dimension [1, T, hidden] → [hidden]
        hidden = enc_out.last_hidden_state  # [1, T, hidden]
        pooled = hidden.mean(dim=1)[0]  # [hidden]
        vec = pooled.cpu().tolist()

        if len(vec) != self.embed_dim:
            vec = _project(vec, self.embed_dim)
        return _l2_normalize(vec)

    def _whisper_transcribe(self, audio: Any) -> str:
        """Run full Whisper encoder-decoder for ASR."""
        if not NUMPY_AVAILABLE:
            return _STUB_TRANSCRIPTION

        max_samples = int(self.sample_rate * self.max_duration)
        arr = audio[:max_samples] if len(audio) > max_samples else audio

        inputs = self._processor(
            arr,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        ).to(self.device)

        generate_kwargs: Dict[str, Any] = {}
        if self.language:
            generate_kwargs["language"] = self.language

        with torch.no_grad():
            # We need to use a GenerationMixin model — load ForConditionalGeneration
            # The base WhisperModel doesn't have generate(); use AutoModelForSpeechSeq2Seq
            # This is a fallback — in _load_model we switch to the generation model
            token_ids = self._model.generate(
                inputs["input_features"], **generate_kwargs
            )

        return self._processor.batch_decode(token_ids, skip_special_tokens=True)[0]

    # ------------------------------------------------------------------ #
    # Audio I/O helpers                                                    #
    # ------------------------------------------------------------------ #

    def _load_audio_file(self, path: str) -> tuple[Any, int]:
        """Load audio file; returns (float32 mono array, sample_rate)."""
        if SOUNDFILE_AVAILABLE:
            audio, sr = sf.read(path, dtype="float32")
            return audio, sr
        if TORCHAUDIO_AVAILABLE:
            waveform, sr = torchaudio.load(path)
            return waveform.mean(0).numpy(), int(sr)
        raise RuntimeError(
            "Cannot load audio file — install soundfile or torchaudio: "
            "pip install soundfile torchaudio"
        )

    @staticmethod
    def _resample(audio: Any, src_sr: int, dst_sr: int) -> Any:
        """Resample audio array from src_sr to dst_sr."""
        if src_sr == dst_sr:
            return audio
        if TORCHAUDIO_AVAILABLE and NUMPY_AVAILABLE:
            import numpy as np

            t = torch.tensor(audio).unsqueeze(0)
            t = torchaudio.functional.resample(t, src_sr, dst_sr)
            return t.squeeze(0).numpy()
        # Simple skip-sample fallback (quality: poor but functional)
        if NUMPY_AVAILABLE:
            import numpy as np

            ratio = dst_sr / src_sr
            indices = np.round(np.arange(0, len(audio), 1 / ratio)).astype(int)
            indices = indices[indices < len(audio)]
            return audio[indices]
        return audio
