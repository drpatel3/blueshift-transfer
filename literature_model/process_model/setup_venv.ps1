# process_model — one-shot venv bootstrap.
#
# Usage (PowerShell, from the `dev/` directory OR `dev/process_model/`):
#   ./process_model/setup_venv.ps1
#
# Creates .venv at the repo root (dev/.venv) and installs process_model/requirements.txt.
# Idempotent: re-running upgrades deps in place.

$ErrorActionPreference = "Stop"

# Anchor to the repo root (parent of process_model/)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Split-Path -Parent $scriptDir
Set-Location $repoRoot

$venvPath = Join-Path $repoRoot ".venv"
$reqFile  = Join-Path $scriptDir "requirements.txt"

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating venv at $venvPath"
    python -m venv $venvPath
} else {
    Write-Host "Reusing existing venv at $venvPath"
}

$activate = Join-Path $venvPath "Scripts\Activate.ps1"
. $activate

python -m pip install --upgrade pip
python -m pip install -r $reqFile

Write-Host ""
Write-Host "Done. To activate in a new shell:"
Write-Host "  . $activate"
Write-Host ""
Write-Host "Smoke tests:"
Write-Host "  python -m pytest process_model/tests/ -v"
Write-Host "  python -m process_model.throughput --tea"
