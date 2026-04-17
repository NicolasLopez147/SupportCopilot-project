param(
    [string]$OutputPath = "outputs\experiments\comparison_overview.json"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Interpréteur introuvable dans $Python. Activez ou créez d'abord l'environnement virtuel."
}

Write-Host "==> Exécution de l'évaluation globale : intention + résumé + réponse" -ForegroundColor Cyan

& $Python -m src.experiments.eval.eval_copilot_components --output-path $OutputPath
if ($LASTEXITCODE -ne 0) {
    throw "L'évaluation globale a échoué."
}

Write-Host ""
Write-Host "[done] Évaluation globale terminée." -ForegroundColor Green
Write-Host "[info] Output: $OutputPath"
