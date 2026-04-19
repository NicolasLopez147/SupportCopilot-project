[CmdletBinding()]
param(
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 8002,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Environnement virtuel introuvable. Crée d'abord .venv et installe les dépendances."
}

$args = @("-m", "uvicorn", "services.summary_service.app.main:app", "--host", $ListenHost, "--port", $Port.ToString())
if ($Reload) {
    $args += "--reload"
}

Write-Host "[summary-service] Lancement sur http://$ListenHost`:$Port"
& $python @args
