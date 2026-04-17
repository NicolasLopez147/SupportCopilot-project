param(
    [string]$InputPath = "data\synthetic\synthetic_reply_eval.jsonl",
    [string]$OutputPath = "outputs\copilot\runtime\pipeline_full_run.json",
    [switch]$ResetFeedback,
    [int]$Limit = 0,
    [string]$ConversationId
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Interpréteur introuvable dans $Python. Activez ou créez d'abord l'environnement virtuel."
}

$Arguments = @(
    "-m", "src.copilot.pipeline.run_support_copilot",
    "--input-path", $InputPath,
    "--output-path", $OutputPath
)

if ($ConversationId) {
    $Arguments += @("--conversation-id", $ConversationId)
} else {
    $Arguments += "--run-all"
}

if ($ResetFeedback) {
    $Arguments += "--reset-feedback"
}

if ($Limit -gt 0) {
    $Arguments += @("--limit", "$Limit")
}

Write-Host "==> Exécution du pipeline SupportCopilot" -ForegroundColor Cyan
& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "L'exécution du pipeline a échoué."
}

Write-Host ""
Write-Host "[done] Pipeline exécuté." -ForegroundColor Green
Write-Host "[info] Output: $OutputPath"
Write-Host "[info] Mémoire des échecs :"
Write-Host "  - data\feedback\memory\intent_failures.jsonl"
Write-Host "  - data\feedback\memory\summary_failures.jsonl"
Write-Host "  - data\feedback\memory\reply_failures.jsonl"
