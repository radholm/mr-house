#!/usr/bin/env bash
# One-shot setup for Mr. House on macOS / Linux.
#   ./scripts/setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Mr. House setup"

# --- 0. pick a Python 3.10+ interpreter ---
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    if "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      PY="$c"; break
    fi
  fi
done
if [[ -z "$PY" ]]; then
  echo "!! Need Python >= 3.10. Install it (e.g. 'brew install python@3.13') and retry."
  exit 1
fi
echo "==> Using interpreter: $PY ($($PY --version 2>&1))"

# --- 1. system deps (macOS via Homebrew) ---
if [[ "$(uname)" == "Darwin" ]]; then
  if command -v brew >/dev/null 2>&1; then
    echo "==> Installing system deps via Homebrew (portaudio, uv, ollama)"
    brew list portaudio >/dev/null 2>&1 || brew install portaudio
    brew list uv >/dev/null 2>&1 || brew install uv            # provides uvx for MCP servers
    # Use the official Ollama app (cask). The plain 'ollama' formula has
    # shipped builds missing the llama-server runner binary.
    if ! brew list --cask ollama-app >/dev/null 2>&1; then
      brew install --cask ollama-app || echo "!! install Ollama from https://ollama.com manually"
    fi
  else
    echo "!! Homebrew not found. Install portaudio, uv and Ollama manually."
  fi
fi

# --- 2. python venv + deps ---
echo "==> Creating virtualenv (.venv)"
rm -rf .venv
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
echo "==> Installing Python requirements (this can take a few minutes)"
pip install -r requirements.txt

# --- 3. openWakeWord base models ---
echo "==> Downloading openWakeWord base models"
python - <<'PY'
try:
    import openwakeword
    openwakeword.utils.download_models()
    print("   openWakeWord models ready.")
except Exception as e:
    print(f"   skipped: {e}")
PY

# --- 4. Piper voice ---
VOICE_DIR="src/mr_house/assets/voices"
VOICE="en_US-ryan-high"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high"
mkdir -p "$VOICE_DIR"
if [[ ! -f "$VOICE_DIR/$VOICE.onnx" ]]; then
  echo "==> Downloading Piper voice: $VOICE"
  curl -L --fail -o "$VOICE_DIR/$VOICE.onnx"      "$BASE/$VOICE.onnx" || echo "   voice download failed; grab one manually."
  curl -L --fail -o "$VOICE_DIR/$VOICE.onnx.json" "$BASE/$VOICE.onnx.json" || true
else
  echo "==> Piper voice already present."
fi

# --- 5. Ollama model ---
if command -v ollama >/dev/null 2>&1; then
  if ! curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
    echo "==> Starting Ollama server in the background"
    (ollama serve >/tmp/ollama.log 2>&1 &)
    for _ in 1 2 3 4 5 6 7 8; do
      curl -s http://localhost:11434/api/version >/dev/null 2>&1 && break
      sleep 2
    done
  fi
  echo "==> Pulling Ollama model (llama3.2:3b)"
  ollama pull llama3.2:3b || echo "   pull failed; run 'ollama serve' then 'ollama pull llama3.2:3b'."
fi

echo ""
echo "==> Done. Next:"
echo "    1. Drop a portrait at src/mr_house/assets/house.png (optional)."
echo "    2. source .venv/bin/activate"
echo "    3. python run.py --check     # verify subsystems"
echo "    4. python run.py             # launch Mr. House"

