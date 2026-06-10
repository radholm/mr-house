"""Wake-word detection via openWakeWord (fully local).

We feed 16 kHz int16 frames in and ask for the highest score across the
configured models. When it crosses the threshold we report a trigger and apply
a short refractory period so a single utterance doesn't fire repeatedly.

To make a *real* "Mr. House" wake word, train/convert a model and drop the
``.onnx``/``.tflite`` file into ``assets/`` then list it under
``wake_word.models`` in ``config.yaml``. Until then we fall back to a bundled
model so the pipeline still runs.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    from openwakeword.model import Model as _OWWModel
    import openwakeword
except Exception as exc:  # pragma: no cover
    _OWWModel = None
    openwakeword = None
    log.warning("openwakeword unavailable (%s); wake word disabled.", exc)


class WakeWordDetector:
    def __init__(
        self,
        models: list[str],
        threshold: float = 0.5,
        refractory_s: float = 2.0,
        repo_root: Optional[Path] = None,
    ) -> None:
        self.threshold = threshold
        self.refractory_s = refractory_s
        self._last_fire = 0.0
        self._model = None
        self._scores: dict[str, float] = {}

        if _OWWModel is None:
            return

        # Resolve any model paths relative to the repo root.
        resolved: list[str] = []
        for m in models:
            p = Path(m)
            if repo_root and not p.is_absolute() and p.suffix in {".onnx", ".tflite"}:
                p = repo_root / p
            resolved.append(str(p) if p.suffix in {".onnx", ".tflite"} else m)

        try:
            # Ensure base melspectrogram/embedding models are present.
            try:
                openwakeword.utils.download_models()
            except Exception:
                pass
            self._model = _OWWModel(
                wakeword_models=resolved,
                inference_framework="onnx",
            )
            log.info("Wake-word models loaded: %s", list(self._model.models.keys()))
        except Exception as exc:
            log.error("Failed to load wake-word models %s: %s", resolved, exc)
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset()

    def process(self, frame_int16: np.ndarray) -> Optional[str]:
        """Feed a frame; return the model name that triggered, else None."""
        if self._model is None:
            return None
        self._scores = self._model.predict(frame_int16)
        if not self._scores:
            return None
        name, score = max(self._scores.items(), key=lambda kv: kv[1])
        if score >= self.threshold and (time.time() - self._last_fire) > self.refractory_s:
            self._last_fire = time.time()
            self.reset()
            log.info("Wake word '%s' detected (score=%.2f).", name, score)
            return name
        return None

    @property
    def last_scores(self) -> dict[str, float]:
        return dict(self._scores)

