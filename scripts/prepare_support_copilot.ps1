param(
    [switch]$Force,
    [switch]$SkipIntent,
    [switch]$SkipSummary,
    [switch]$SkipReply
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Interpréteur introuvable dans $Python. Activez ou créez d'abord l'environnement virtuel."
}

function Invoke-Step {
    param(
        [string]$Label,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "==> $Label" -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Échec de l'étape : $Label"
    }
}

function Ensure-Artifact {
    param(
        [string]$Label,
        [string]$ArtifactPath,
        [string[]]$Command,
        [switch]$Skip
    )

    if ($Skip) {
        Write-Host "[skip] $Label" -ForegroundColor Yellow
        return
    }

    if ((-not $Force) -and (Test-Path $ArtifactPath)) {
        Write-Host "[ok] $Label existe déjà -> $ArtifactPath" -ForegroundColor Green
        return
    }

    Invoke-Step -Label $Label -Arguments $Command
}

$IntentArtifact = Join-Path $ProjectRoot "outputs\experiments\intent\synthetic_embedding\intent_synthetic_model.joblib"
$SummaryArtifact = Join-Path $ProjectRoot "outputs\experiments\summary\lora_base\final_model"
$ReplyArtifact = Join-Path $ProjectRoot "outputs\experiments\reply\lora_base\final_model"

Ensure-Artifact `
    -Label "Entraîner le classifieur d'intention synthétique" `
    -ArtifactPath $IntentArtifact `
    -Command @("-m", "src.experiments.baselines.train_synthetic_intent") `
    -Skip:$SkipIntent

Ensure-Artifact `
    -Label "Entraîner le modèle LoRA de base pour le résumé" `
    -ArtifactPath $SummaryArtifact `
    -Command @("-m", "src.experiments.llm.train_lora_summary") `
    -Skip:$SkipSummary

Ensure-Artifact `
    -Label "Entraîner le modèle LoRA de base pour la réponse" `
    -ArtifactPath $ReplyArtifact `
    -Command @("-m", "src.experiments.llm.train_lora_reply") `
    -Skip:$SkipReply

Write-Host ""
Write-Host "[done] Les artefacts principaux du pipeline sont prêts." -ForegroundColor Green
