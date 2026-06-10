"""Prepare a Piper training dataset from a single long recording.

Piper's trainer wants a folder of short clips (~3-15s) plus a ``metadata.csv``
of ``filename|text``. You have one long file, so this script:

  1. decodes your audio (any format ffmpeg/PyAV can read),
  2. transcribes it with faster-whisper to get sentence-level segments + text,
  3. cuts a clip per segment, resamples to mono 22050 Hz 16-bit WAV,
  4. filters clips that are too short/long or have empty text,
  5. writes ``<out>/wav/utt_XXXX.wav`` and ``<out>/metadata.csv``.

Run on any machine with the project deps installed (e.g. your Mac), then copy
the output folder to your training box.

Example:
    python scripts/prepare_voice_dataset.py \
        --input recordings/mr_house_1h.wav \
        --out datasets/mr_house \
        --whisper-model medium.en

IMPORTANT: open ``metadata.csv`` afterwards and fix obvious transcription
mistakes — ASR errors directly hurt the trained voice's pronunciation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
except Exception as exc:  # pragma: no cover
    print(f"Missing deps ({exc}). Install with: pip install -r requirements.txt")
    raise SystemExit(1)


def _clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return audio
    n = int(round(len(audio) * dst / src))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a Piper dataset from one long audio file.")
    ap.add_argument("--input", required=True, help="Path to the long recording (wav/mp3/m4a/...).")
    ap.add_argument("--out", required=True, help="Output dataset directory.")
    ap.add_argument("--sample-rate", type=int, default=22050, help="Output WAV sample rate (Piper: 22050).")
    ap.add_argument("--whisper-model", default="medium.en",
                    help="faster-whisper model for transcription (medium.en/large-v3 = better text).")
    ap.add_argument("--language", default="en")
    ap.add_argument("--min-sec", type=float, default=2.0, help="Drop clips shorter than this.")
    ap.add_argument("--max-sec", type=float, default=16.0, help="Drop clips longer than this.")
    ap.add_argument("--pad-ms", type=int, default=120, help="Padding added to each side of a clip.")
    ap.add_argument("--gain", type=float, default=1.0, help="Optional amplitude multiplier.")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input not found: {in_path}")
        return 1

    out_dir = Path(args.out)
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    sr = args.sample_rate

    print(f"==> Decoding {in_path.name} at {sr} Hz...")
    audio = decode_audio(str(in_path), sampling_rate=sr)  # float32 mono
    total_sec = len(audio) / sr
    print(f"    {total_sec/60:.1f} min of audio.")

    print(f"==> Transcribing with '{args.whisper_model}' (this can take a while)...")
    model = WhisperModel(args.whisper_model, device="auto", compute_type="int8")
    audio16 = _resample(audio, sr, 16000)
    segments, _info = model.transcribe(
        audio16,
        language=args.language,
        beam_size=5,
        vad_filter=True,                       # split on natural pauses
        vad_parameters={"min_silence_duration_ms": 400},
        condition_on_previous_text=False,
    )

    pad = args.pad_ms / 1000.0
    rows: list[tuple[str, str]] = []
    kept_sec = 0.0
    idx = 0
    for seg in segments:
        text = _clean_text(seg.text)
        dur = seg.end - seg.start
        if not text or dur < args.min_sec or dur > args.max_sec:
            continue
        start = max(0.0, seg.start - pad)
        end = min(total_sec, seg.end + pad)
        clip = audio[int(start * sr): int(end * sr)]
        if len(clip) < int(args.min_sec * sr):
            continue
        # Peak-normalize lightly to keep levels consistent, then apply gain.
        peak = float(np.max(np.abs(clip))) or 1.0
        clip = (clip / peak * 0.95 * args.gain).astype(np.float32)
        clip = np.clip(clip, -1.0, 1.0)

        name = f"utt_{idx:04d}.wav"
        sf.write(str(wav_dir / name), clip, sr, subtype="PCM_16")
        rows.append((name, text))
        kept_sec += (end - start)
        idx += 1

    csv_path = out_dir / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        for name, text in rows:
            fh.write(f"{name}|{text}\n")

    print("\n==> Done.")
    print(f"    clips written : {len(rows)}")
    print(f"    kept audio    : {kept_sec/60:.1f} min")
    print(f"    wav dir       : {wav_dir}")
    print(f"    metadata.csv  : {csv_path}")
    if not rows:
        print("    !! No clips produced — check the input audio / try a smaller --min-sec.")
        return 1
    print("\nNext: review metadata.csv for transcription errors, then train (see docs/TRAINING.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


