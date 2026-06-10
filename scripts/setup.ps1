# One-shot setup for Mr. House on Windows (PowerShell).
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#
# Installs Python deps into a venv, downloads a Piper voice + openWakeWord
# models, and (via winget) installs uv and Ollama, then pulls the LLM model.

$ErrorActionPreference = "Stop"

# By default, under ErrorActionPreference=Stop, PowerShell turns *anything a
# native program writes to stderr* (even harmless warnings/progress) into a
# terminating "NativeCommandError". Disable that here:
#  - PS 7.3+: opt out via the preference variable.
#  - All versions: run native commands through Invoke-Native (below), which
#    temporarily relaxes ErrorActionPreference so stderr can't kill the script.
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Invoke-Native {
    # Run a native command without letting its stderr/exit code throw.
    # Check $LASTEXITCODE afterwards for real failures.
    param([Parameter(Mandatory)][scriptblock]$Cmd)
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Cmd } finally { $ErrorActionPreference = $old }
}

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "==> Mr. House setup (Windows)"

# --- 0. find a Python 3.10+ interpreter -----------------------------------
function Find-Python {
    foreach ($cand in @("py -3.13", "py -3.12", "py -3.11", "py -3.10", "python", "py -3")) {
        $parts = $cand.Split(" ")
        $exe = $parts[0]
        # Args after the exe (empty for a bare "python"; avoids the 1..0 range bug).
        $rest = if ($parts.Length -gt 1) { $parts[1..($parts.Length - 1)] } else { @() }
        try {
            $ver = & $exe @rest -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and [version]$ver -ge [version]"3.10") {
                return ,($exe, $rest)
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
function Install-IfMissing {
    # Install a winget package only when its command isn't already on PATH.
    param([string]$Command, [string]$WingetId, [string]$Label)
    if (Get-Command $Command -ErrorAction SilentlyContinue) {
        Write-Host "==> $Label already installed; skipping."
        return
    }
    Write-Host "==> Installing $Label via winget..."
    Invoke-Native { winget install --id $WingetId --accept-source-agreements --accept-package-agreements -e | Out-Null }
}

if (Get-Command winget -ErrorAction SilentlyContinue) {
    Install-IfMissing -Command "uv"     -WingetId "astral-sh.uv"  -Label "uv"
    Install-IfMissing -Command "ollama" -WingetId "Ollama.Ollama" -Label "Ollama"
} else {
    Write-Host "!! winget not found. Install uv (https://astral.sh/uv) and Ollama (https://ollama.com) manually."
}

# --- 2. python venv + deps -------------------------------------------------
Write-Host "==> Creating virtualenv (.venv)"
if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" }
Invoke-Native { & $pyExe @pyArgs -m venv .venv }
if ($LASTEXITCODE -ne 0) { Write-Host "!! Failed to create venv." -ForegroundColor Red; exit 1 }

$venvPy = Join-Path $root ".venv\Scripts\python.exe"
Invoke-Native { & $venvPy -m pip install --upgrade pip }
Write-Host "==> Installing Python requirements (this can take a few minutes)"
Invoke-Native { & $venvPy -m pip install -r requirements.txt }
if ($LASTEXITCODE -ne 0) { Write-Host "!! pip install failed. See output above." -ForegroundColor Red; exit 1 }

# --- 3. openWakeWord base models ------------------------------------------
Write-Host "==> Downloading openWakeWord base models"
Invoke-Native { & $venvPy -c "import openwakeword; openwakeword.utils.download_models()" }
if ($LASTEXITCODE -eq 0) { Write-Host "   ok" }
else { Write-Host "   (skipped; models will download on first run)" }

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
    Invoke-Native { ollama pull llama3.2:3b }
    if ($LASTEXITCODE -ne 0) { Write-Host "   pull failed; start Ollama, then 'ollama pull llama3.2:3b'." }
} else {
    Write-Host "   Ollama not on PATH yet. Open a new terminal (or sign out/in), then 'ollama pull llama3.2:3b'."
}

Write-Host ""
Write-Host "==> Done. Next:"
Write-Host "    1. (optional) drop a portrait at src\mr_house\assets\house.png"
Write-Host "    2. .\.venv\Scripts\Activate.ps1"
Write-Host "    3. python run.py --check     # verify subsystems"
Write-Host "    4. python run.py             # launch Mr. House"

