"""Transcribe a folder of separate voice clips into a Piper dataset.

Use this when you already have many short, single-speaker clips (e.g. extracted
game voice lines in .ogg). For each audio file it:

  1. decodes the clip (ogg/wav/mp3/m4a/flac/opus — anything PyAV/ffmpeg reads),
  2. transcribes it with faster-whisper,
  3. writes a mono 22050 Hz 16-bit WAV into ``<out>/wav/``,
  4. appends a ``filename|text`` row to ``<out>/metadata.csv``.

Example:
    python scripts/transcribe_clips.py \
        --input-dir voices/mr_house_ogg \
        --out datasets/mr_house \
        --whisper-model medium.en

Then review metadata.csv (fix ASR errors) and train — see docs/TRAINING.md.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
except Exception as exc:  # pragma: no cover
    print(f"Missing deps ({exc}). Install with: pip install -r requirements.txt")
    raise SystemExit(1)

AUDIO_EXTS = {".ogg", ".wav", ".mp3", ".m4a", ".flac", ".opus", ".aac", ".wma"}


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _safe_name(stem: str) -> str:
    """Make a filesystem/CSV-safe wav name from an original stem."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return (name or "clip") + ".wav"


def main() -> int:
    ap = argparse.ArgumentParser(description="Transcribe a folder of clips into a Piper dataset.")
    ap.add_argument("--input-dir", required=True, help="Folder containing the clips.")
    ap.add_argument("--out", required=True, help="Output dataset directory.")
    ap.add_argument("--whisper-model", default="medium.en",
                    help="faster-whisper model (medium.en / large-v3 = better text).")
    ap.add_argument("--language", default="en")
    ap.add_argument("--sample-rate", type=int, default=22050, help="Output WAV rate (Piper: 22050).")
    ap.add_argument("--pattern", default="*", help="Glob within input-dir (e.g. '*.ogg').")
    ap.add_argument("--recursive", action="store_true", help="Search sub-folders too.")
    ap.add_argument("--min-sec", type=float, default=0.4, help="Skip clips shorter than this.")
    ap.add_argument("--max-sec", type=float, default=30.0, help="Skip clips longer than this.")
    ap.add_argument("--rename", action="store_true",
                    help="Rename clips to utt_0000.wav, ... (default keeps original names).")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        print(f"Input dir not found: {in_dir}")
        return 1

    globber = in_dir.rglob if args.recursive else in_dir.glob
    files = sorted(
        p for p in globber(args.pattern)
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    if not files:
        print(f"No audio files matching '{args.pattern}' in {in_dir}")
        return 1
    print(f"==> Found {len(files)} clips. Loading Whisper '{args.whisper_model}'...")

    model = WhisperModel(args.whisper_model, device="auto", compute_type="int8")

    out_dir = Path(args.out)
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    sr = args.sample_rate

    rows: list[tuple[str, str]] = []
    skipped = 0
    seen_names: set[str] = set()

    for i, path in enumerate(files, 1):
        try:
            audio = decode_audio(str(path), sampling_rate=sr)  # float32 mono
        except Exception as exc:
            print(f"  [{i}/{len(files)}] SKIP (decode failed): {path.name} ({exc})")
            skipped += 1
            continue

        dur = len(audio) / sr
        if dur < args.min_sec or dur > args.max_sec:
            print(f"  [{i}/{len(files)}] skip (len {dur:.1f}s): {path.name}")
            skipped += 1
            continue

        # Whisper wants 16 kHz.
        audio16 = audio if sr == 16000 else _resample(audio, sr, 16000)
        segments, _ = model.transcribe(
            audio16, language=args.language, beam_size=5,
            without_timestamps=True, condition_on_previous_text=False,
        )
        text = _clean_text(" ".join(s.text for s in segments))
        if not text:
            print(f"  [{i}/{len(files)}] skip (no speech): {path.name}")
            skipped += 1
            continue

        if args.rename:
            name = f"utt_{len(rows):04d}.wav"
        else:
            name = _safe_name(path.stem)
            # Avoid collisions if two originals sanitise to the same name.
            base = name[:-4]
            n = 1
            while name in seen_names:
                name = f"{base}_{n}.wav"
                n += 1
        seen_names.add(name)

        # Light peak-normalise for consistent levels.
        peak = float(np.max(np.abs(audio))) or 1.0
        clip = np.clip(audio / peak * 0.95, -1.0, 1.0).astype(np.float32)
        sf.write(str(wav_dir / name), clip, sr, subtype="PCM_16")
        rows.append((name, text))
        print(f"  [{i}/{len(files)}] {name}: {text[:70]}")

    csv_path = out_dir / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        for name, text in rows:
            fh.write(f"{name}|{text}\n")

    print("\n==> Done.")
    print(f"    transcribed : {len(rows)}")
    print(f"    skipped     : {skipped}")
    print(f"    wav dir     : {wav_dir}")
    print(f"    metadata.csv: {csv_path}")
    if not rows:
        return 1
    print("\nNext: review metadata.csv for transcription errors, then train (docs/TRAINING.md).")
    return 0


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return audio
    n = int(round(len(audio) * dst / src))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())


