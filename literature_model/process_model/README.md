# process_model — VS Code quickstart

Self-contained environment for the Cu-sulfide digital twin and BlueShift TEA.

## First-time setup

From a PowerShell terminal at the repo root (`dev/`):

```powershell
./process_model/setup_venv.ps1
```

This creates `dev/.venv` and installs everything in
`process_model/requirements.txt` (numpy, scipy, openpyxl, pytest).

## Point VS Code at the venv

1. Open the `dev/` folder in VS Code.
2. `Ctrl+Shift+P` → **Python: Select Interpreter**.
3. Pick `./.venv/Scripts/python.exe`.

New terminals inside VS Code will activate the venv automatically (because
VS Code's Python extension runs `Activate.ps1` on shell init once the
interpreter is selected).

## Smoke test

```powershell
python -m pytest process_model/tests/ -v
python -m process_model.throughput --tea
```

Expect 15 passing TEA tests and a stream/power/TEA report for the
reference 2000 tph @ 0.8% Cu feed.

## Re-running setup

`setup_venv.ps1` is idempotent — re-run it to pull in new dependencies after
editing `requirements.txt`.
