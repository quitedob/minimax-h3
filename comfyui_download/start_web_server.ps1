$ErrorActionPreference = 'Stop'
$env:COMFY_SAGE3 = '0'
$env:PYTHONUTF8 = '1'
$root = 'F:\python\h3\ComfyUI_sage3_py312'

if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) {
    Write-Output 'STILL-LISTENING'
    exit 0
}

Start-Process -FilePath "$root\venv\Scripts\python.exe" -ArgumentList @(
    '-s', 'main.py',
    '--extra-model-paths-config', 'extra_model_paths.yaml',
    '--disable-auto-launch',
    '--listen', '127.0.0.1',
    '--port', '8188'
) -WorkingDirectory $root -WindowStyle Hidden `
    -RedirectStandardOutput "$root\cloud_server.log" `
    -RedirectStandardError  "$root\cloud_server_error.log"

Write-Output 'STARTED'
