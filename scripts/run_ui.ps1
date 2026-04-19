[CmdletBinding()]
param(
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Environnement virtuel introuvable. Cree d'abord .venv et installe les dependances."
}

$args = @(
    "-m",
    "streamlit",
    "run",
    "ui/app.py",
    "--server.address",
    $ListenHost,
    "--server.port",
    $Port.ToString()
)

Write-Host "[ui] Lancement de Streamlit sur http://$ListenHost`:$Port"
& $python @args
