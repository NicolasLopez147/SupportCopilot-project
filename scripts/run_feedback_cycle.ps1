param(
    [ValidateSet("reply", "summary", "all")]
    [string]$Target = "all",
    [string]$InputPath = "data\synthetic\synthetic_reply_eval.jsonl",
    [string]$ReplyTestPath = "data\synthetic\reply_test.jsonl",
    [switch]$ResetFeedback
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

$PipelineArgs = @(
    "-m", "src.copilot.pipeline.run_support_copilot",
    "--input-path", $InputPath,
    "--run-all",
    "--output-path", "outputs\copilot\runtime\pipeline_full_run.json"
)
if ($ResetFeedback) {
    $PipelineArgs += "--reset-feedback"
}

Invoke-Step -Label "Exécuter le pipeline avec les critiques" -Arguments $PipelineArgs
Invoke-Step -Label "Construire les candidats de réentraînement" -Arguments @("-m", "src.copilot.feedback.build_retraining_sets")
Invoke-Step -Label "Construire les jeux augmentés d'entraînement" -Arguments @("-m", "src.copilot.feedback.build_augmented_training_sets")

if (($Target -eq "reply") -or ($Target -eq "all")) {
    Invoke-Step -Label "Générer les réponses baseline" -Arguments @(
        "-m", "src.copilot.retrieval.rag.generate_reply_baseline",
        "--input-path", $ReplyTestPath
    )
    Invoke-Step -Label "Générer les réponses avec retrieval" -Arguments @(
        "-m", "src.copilot.retrieval.rag.generate_reply_with_retrieval",
        "--input-path", $ReplyTestPath
    )
    Invoke-Step -Label "Générer les réponses LoRA de base" -Arguments @(
        "-m", "src.experiments.llm.generate_lora_reply_predictions",
        "--input-path", $ReplyTestPath,
        "--output-path", "outputs\experiments\reply\lora_base\test_replies.json"
    )
    Invoke-Step -Label "Réentraîner le modèle Feedback-LoRA pour la réponse" -Arguments @(
        "-m", "src.experiments.llm.train_lora_reply_feedback"
    )
    Invoke-Step -Label "Générer les réponses Feedback-LoRA" -Arguments @(
        "-m", "src.experiments.llm.generate_lora_reply_predictions",
        "--input-path", $ReplyTestPath,
        "--adapter-dir", "outputs\experiments\reply\lora_feedback\final_model",
        "--output-path", "outputs\experiments\reply\lora_feedback\test_replies.json"
    )
    Invoke-Step -Label "Évaluer les méthodes de réponse" -Arguments @(
        "-m", "src.experiments.eval.eval_reply_methods",
        "--lora-feedback-path", "outputs\experiments\reply\lora_feedback\test_replies.json"
    )
}

if (($Target -eq "summary") -or ($Target -eq "all")) {
    Invoke-Step -Label "Générer les résumés LoRA de base" -Arguments @(
        "-m", "src.experiments.llm.generate_lora_summary_predictions",
        "--input-path", $ReplyTestPath,
        "--output-path", "outputs\experiments\summary\lora_base\test_predictions.json",
        "--limit", "0"
    )
    Invoke-Step -Label "Réentraîner le modèle Feedback-LoRA pour le résumé" -Arguments @(
        "-m", "src.experiments.llm.train_lora_summary_feedback"
    )
    Invoke-Step -Label "Générer les résumés Feedback-LoRA" -Arguments @(
        "-m", "src.experiments.llm.generate_lora_summary_predictions",
        "--input-path", $ReplyTestPath,
        "--adapter-dir", "outputs\experiments\summary\lora_feedback\final_model",
        "--output-path", "outputs\experiments\summary\lora_feedback\test_predictions.json",
        "--limit", "0"
    )
    Invoke-Step -Label "Évaluer les méthodes de résumé" -Arguments @(
        "-m", "src.experiments.eval.eval_summary_methods"
    )
}

Invoke-Step -Label "Vue globale : intention + résumé + réponse" -Arguments @(
    "-m", "src.experiments.eval.eval_copilot_components"
)

Write-Host ""
Write-Host "[done] Boucle de feedback terminée pour : $Target" -ForegroundColor Green
