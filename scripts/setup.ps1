# One-shot setup for Mr. House on Windows (PowerShell).
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#
# Installs Python deps into a venv, downloads a Piper voice + openWakeWord
# models, and (via winget) installs uv and Ollama, then pulls the LLM model.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "==> Mr. House setup (Windows)"

# --- 0. find a Python 3.10+ interpreter -----------------------------------
function Find-Python {
    foreach ($cand in @("py -3.13", "py -3.12", "py -3.11", "py -3.10", "python", "py -3")) {
        $parts = $cand.Split(" ")
        $exe = $parts[0]
        $args = $parts[1..($parts.Length - 1)]
        try {
            $ver = & $exe @args -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and [version]$ver -ge [version]"3.10") {
                return ,($exe, $args)
            }
        } catch { }
    }
    return $null
}

$py = Find-Python
if ($null -eq $py) {
    Write-Host "!! Need Python >= 3.10. Install it (winget install Python.Python.3.13) and retry." -ForegroundColor Red
    exit 1
}
$pyExe = $py[0]; $pyArgs = $py[1]
Write-Host "==> Using interpreter: $pyExe $pyArgs"

# --- 1. system deps via winget --------------------------------------------
if (Get-Command winget -ErrorAction SilentlyContinue) {
    Write-Host "==> Installing uv and Ollama via winget (skip if already present)"
    winget install --id astral-sh.uv         --accept-source-agreements --accept-package-agreements -e 2>$null | Out-Null
    winget install --id Ollama.Ollama         --accept-source-agreements --accept-package-agreements -e 2>$null | Out-Null
} else {
    Write-Host "!! winget not found. Install uv (https://astral.sh/uv) and Ollama (https://ollama.com) manually."
}

# --- 2. python venv + deps -------------------------------------------------
Write-Host "==> Creating virtualenv (.venv)"
if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" }
& $pyExe @pyArgs -m venv .venv
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip
Write-Host "==> Installing Python requirements (this can take a few minutes)"
& $venvPy -m pip install -r requirements.txt

# --- 3. openWakeWord base models ------------------------------------------
Write-Host "==> Downloading openWakeWord base models"
& $venvPy -c "import openwakeword; openwakeword.utils.download_models(); print('   ok')" 2>$null

# --- 4. Piper voice --------------------------------------------------------
$voiceDir = Join-Path $root "src\mr_house\assets\voices"
$voice = "en_US-ryan-high"
$base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high"
New-Item -ItemType Directory -Force -Path $voiceDir | Out-Null
$onnx = Join-Path $voiceDir "$voice.onnx"
if (-not (Test-Path $onnx)) {
    Write-Host "==> Downloading Piper voice: $voice"
    Invoke-WebRequest -Uri "$base/$voice.onnx"      -OutFile $onnx
    Invoke-WebRequest -Uri "$base/$voice.onnx.json" -OutFile (Join-Path $voiceDir "$voice.onnx.json")
} else {
    Write-Host "==> Piper voice already present."
}

# --- 5. Ollama model -------------------------------------------------------
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "==> Pulling Ollama model (llama3.2:3b). Ollama runs in the background on Windows."
    try { ollama pull llama3.2:3b } catch { Write-Host "   pull failed; start Ollama, then 'ollama pull llama3.2:3b'." }
} else {
    Write-Host "   Ollama not on PATH yet. Open a new terminal (or sign out/in), then 'ollama pull llama3.2:3b'."
}

Write-Host ""
Write-Host "==> Done. Next:"
Write-Host "    1. (optional) drop a portrait at src\mr_house\assets\house.png"
Write-Host "    2. .\.venv\Scripts\Activate.ps1"
Write-Host "    3. python run.py --check     # verify subsystems"
Write-Host "    4. python run.py             # launch Mr. House"

