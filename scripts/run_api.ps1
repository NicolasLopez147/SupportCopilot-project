[CmdletBinding()]
param(
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Reload,
    [switch]$ServiceMode
)

$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Environnement virtuel introuvable. Crée d'abord .venv et installe les dépendances."
}

$args = @("-m", "uvicorn", "services.gateway_service.app.main:app", "--host", $ListenHost, "--port", $Port.ToString())
if ($Reload) {
    $args += "--reload"
}

if ($ServiceMode) {
    $env:GATEWAY_USE_EMBEDDED_MODE = "false"
}

Write-Host "[api] Lancement du gateway-service FastAPI sur http://$ListenHost`:$Port"
& $python @args
