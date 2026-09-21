$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$root = 'F:\python\h3\ComfyUI_windows_portable'
$port = 8188

# Production stack: Turbo v4 + NVFP4 TE + SolAttn + EasyCache.
# Do NOT pass --use-sage-attention. SageAttention2 has no native sm_120 kernel
# and can abort this GPU after several H3 sampling steps; see docs/devlog.md §27.

if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
    Write-Output 'STILL-LISTENING'
    exit 0
}

Start-Process -FilePath "$root\python_embeded\python.exe" -ArgumentList @(
    '-s', 'main.py',
    '--extra-model-paths-config', 'extra_model_paths.yaml',
    '--disable-auto-launch',
    '--listen', '127.0.0.1',
    '--port', "$port"
) -WorkingDirectory "$root\ComfyUI" -WindowStyle Hidden `
    -RedirectStandardOutput "$root\cloud_server.log" `
    -RedirectStandardError  "$root\cloud_server_error.log"

Write-Output 'STARTED'
