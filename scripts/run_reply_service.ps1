[CmdletBinding()]
param(
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 8003,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Environnement virtuel introuvable. Crée d'abord .venv et installe les dépendances."
}

$args = @("-m", "uvicorn", "services.reply_service.app.main:app", "--host", $ListenHost, "--port", $Port.ToString())
if ($Reload) {
    $args += "--reload"
}

Write-Host "[reply-service] Lancement sur http://$ListenHost`:$Port"
& $python @args
