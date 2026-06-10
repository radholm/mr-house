# Training a custom Mr. House voice (Piper)

You have a set of separate voice clips of the target speaker. This guide takes
you from those clips to a working `en_US-house-medium.onnx` voice that drops
straight into Mr. House.

There are two phases:

1. **Prepare the dataset** (easy, do it on any machine — your Mac is fine).
2. **Fine-tune & export** the voice (needs a GPU; on Windows use **WSL2**).

> TL;DR: a few hundred clean clips are enough for a good **fine-tune** of an
> existing medium voice — *not* training from scratch. Always pass `--ckpt_path`
> with a medium checkpoint.

---

## Phase 1 — Build the dataset

Piper needs a folder of short clips plus a `metadata.csv` (`filename|text`). If
you already have **separate clips** (e.g. extracted game voice lines in `.ogg`),
the included script transcribes each file, resamples to mono 22050 Hz, and writes
the CSV.

```bash
# from the repo root, in the project venv
python scripts/transcribe_clips.py \
  --input-dir /path/to/mr_house_ogg \
  --out datasets/mr_house \
  --pattern "*.ogg" \
  --whisper-model medium.en        # large-v3 = even better transcripts, slower
```

Output:

```
datasets/mr_house/
  wav/<clip>.wav, ...                   # mono 22050 Hz, 16-bit (one per input clip)
  metadata.csv                          # <clip>.wav|Text...
```

Useful flags: `--recursive` (search sub-folders), `--rename` (use `utt_0000.wav`
names instead of the originals), `--min-sec` / `--max-sec` (skip clips that are
too short/long). Any format ffmpeg can read works (`.ogg`, `.wav`, `.mp3`,
`.m4a`, `.flac`, `.opus`).

**Then do the most important step:** open `metadata.csv` and **fix transcription
mistakes**. ASR errors teach the model wrong pronunciations. Budget 30–60 min of
proof-reading — it's the single biggest quality lever. Also delete rows for any
clip with music, noise, laughter, or overlapping speakers.

Tips:
- The clips are already separated, so each becomes one training utterance.
  Ideal lengths are ~3–15s; very short single-word clips still help.
- More clean, single-speaker data = better. A few hundred lines is plenty for a
  solid fine-tune.

---

## Phase 2 — Train on Windows (via WSL2 + NVIDIA GPU)

Piper's trainer is Linux-oriented (`apt-get`, a Cython build step). On Windows the
clean path is **WSL2 Ubuntu** with your NVIDIA GPU.

> **Do you actually need WSL?** No — it's just the smoothest *local* option
> because it matches Piper's documented Linux environment. Alternatives:
>
> - **Native Windows (no WSL).** Works, but fiddlier:
>   1. Install **Visual Studio Build Tools (MSVC)** with the "Desktop
>      development with C++" workload (the C/C++ compiler).
>   2. Install **espeak-ng for Windows** and add it to `PATH`.
>   3. Install the Cython build deps in the Piper venv:
>      `pip install scikit-build cython cmake ninja` — without `scikit-build`
>      the build fails with `ModuleNotFoundError: No module named 'skbuild'`.
>   4. Replace the bash `build_monotonic_align.sh` with the equivalent
>      `python setup.py build_ext --inplace`.
>
>   PyTorch + CUDA run natively on Windows.
> - **Cloud GPU (no WSL, no local GPU needed)** — often the *easiest overall*,
>   especially without a strong local NVIDIA card. Rent a Linux GPU box (RunPod,
>   vast.ai, Lambda) or use Google Colab/Kaggle, upload your prepared
>   `datasets/mr_house` folder, and run the same commands below. **Phase 1 (the
>   dataset prep) still runs locally** — only training moves to the cloud.
>
> Pick WSL2 or native Windows if you have a good NVIDIA GPU; pick a cloud GPU if
> you don't (or just want the least setup). The training commands are identical.

> **GPU note:** Fine-tuning realistically needs an **NVIDIA GPU** (8 GB VRAM is
> reported to work; 12 GB+ is comfortable). Pure-CPU training is technically
> possible but painfully slow. Make sure you have recent NVIDIA drivers — CUDA in
> WSL works through the Windows driver, you do **not** install a separate CUDA
> driver inside WSL.

### 2.1 Install WSL2 (PowerShell, admin)

```powershell
wsl --install -d Ubuntu
# reboot if prompted, then open "Ubuntu" from the Start menu
```

### 2.2 System packages (inside Ubuntu/WSL)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build python3-venv python3-pip espeak-ng git
```

### 2.3 Clone Piper and install the trainer

```bash
git clone https://github.com/OHF-voice/piper1-gpl.git
cd piper1-gpl
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[train]'
./build_monotonic_align.sh
python3 setup.py build_ext --inplace      # dev build, since we run from the repo
```

Verify the GPU is visible:
```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### 2.4 Get a base checkpoint to fine-tune from

Fine-tuning from a **medium** checkpoint is dramatically faster and better than
from scratch. Grab one (English, medium) from the Piper checkpoints repo, e.g.
`en_US-lessac-medium` or `en_US-ryan-medium`:

- <https://huggingface.co/datasets/rhasspy/piper-checkpoints> → `en/en_US/<voice>/medium/*.ckpt`

Download the `.ckpt` into WSL, e.g. `~/checkpoints/en_US-ryan-medium.ckpt`.

### 2.5 Copy your dataset into WSL

From your prepared folder (Phase 1). Your Windows drives are mounted under
`/mnt/c`, so for example:

```bash
cp -r /mnt/c/Users/<you>/dev/mr-house/datasets/mr_house ~/mr_house_data
```

### 2.6 Train

```bash
python3 -m piper.train fit \
  --data.voice_name "house" \
  --data.csv_path ~/mr_house_data/metadata.csv \
  --data.audio_dir ~/mr_house_data/wav/ \
  --model.sample_rate 22050 \
  --data.espeak_voice "en-us" \
  --data.cache_dir ~/mr_house_cache/ \
  --data.config_path ~/mr_house_out/config.json \
  --data.batch_size 16 \
  --ckpt_path ~/checkpoints/en_US-ryan-medium.ckpt
```

- **`--data.batch_size`**: lower it if you hit out-of-memory (try 8 for 8 GB VRAM,
  16–32 for more).
- **`--ckpt_path`**: the medium checkpoint from 2.4 (fine-tune warm start).
- Checkpoints are written under `lightning_logs/`. Training runs until you stop
  it (Ctrl-C). Listen to samples and stop when it sounds good — often a few
  thousand steps for a fine-tune.
- See all options: `python3 -m piper.train fit --help`.

### 2.7 Export to ONNX

```bash
python3 -m piper.train.export_onnx \
  --checkpoint lightning_logs/version_0/checkpoints/<last>.ckpt \
  --output-file ~/mr_house_out/en_US-house-medium.onnx
```

Piper voices need the `.onnx` **and** a matching `.onnx.json`. Use the config
written during training:

```bash
cp ~/mr_house_out/config.json ~/mr_house_out/en_US-house-medium.onnx.json
```

You now have two files:
```
en_US-house-medium.onnx
en_US-house-medium.onnx.json
```

---

## Phase 3 — Use it in Mr. House

Copy both files into the project's voices folder (from WSL back to Windows):

```bash
cp ~/mr_house_out/en_US-house-medium.onnx*      /mnt/c/Users/<you>/dev/mr-house/src/mr_house/assets/voices/
```

Then point `config.yaml` at the new voice:

```yaml
tts:
  voice: "src/mr_house/assets/voices/en_US-house-medium.onnx"
```

Run it:
```bash
python run.py --check        # tts should report 'ok' with the new voice
python run.py
```

That's it — Mr. House now speaks in the trained voice. The `voice_fx` chain
(reverb/EQ/filters) in `config.yaml` still applies on top, so tweak it to taste.

---

## Troubleshooting

- **`torch.cuda.is_available()` is False** — update your Windows NVIDIA driver,
  restart WSL (`wsl --shutdown` in PowerShell), and reinstall the trainer venv.
  Don't install a CUDA toolkit *driver* inside WSL.
- **Out of memory** — lower `--data.batch_size`; close other GPU apps.
- **Robotic / wrong pronunciation** — usually dirty transcripts. Re-check
  `metadata.csv`, remove noisy clips, and train a bit longer.
- **No NVIDIA GPU** — consider a cloud GPU (e.g. a rented 3090/A6000) for a few
  hours, or a Colab/Kaggle notebook; CPU training is not practical.

