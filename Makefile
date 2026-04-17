PYTHON := .venv/Scripts/python.exe
POWERSHELL := powershell -NoProfile -ExecutionPolicy Bypass

INPUT ?= data/synthetic/synthetic_reply_eval.jsonl
REPLY_TEST ?= data/synthetic/reply_test.jsonl
OUTPUT ?= outputs/experiments/comparison_overview.json
CONVERSATION_ID ?=
LIMIT ?= 0

.PHONY: help prepare prepare-force prepare-skip-summary pipeline pipeline-reset pipeline-one feedback-reply feedback-summary feedback-all eval-all

help:
	@echo Cibles disponibles :
	@echo   make prepare            - Préparer les artefacts principaux de SupportCopilot
	@echo   make prepare-force      - Réentraîner / reconstruire tous les artefacts principaux
	@echo   make prepare-skip-summary - Préparer sans le modèle de résumé de base
	@echo   make pipeline           - Exécuter le pipeline batch sans réinitialiser la mémoire de feedback
	@echo   make pipeline-reset     - Exécuter le pipeline batch et réinitialiser la mémoire de feedback
	@echo   make pipeline-one CONVERSATION_ID=... - Exécuter le pipeline pour une seule conversation
	@echo   make feedback-reply     - Lancer la boucle complète de feedback pour la réponse
	@echo   make feedback-summary   - Lancer la boucle complète de feedback pour le résumé
	@echo   make feedback-all       - Lancer la boucle complète de feedback pour réponse et résumé
	@echo   make eval-all           - Afficher la vue globale d'évaluation

prepare:
	$(POWERSHELL) -File scripts/prepare_support_copilot.ps1

prepare-force:
	$(POWERSHELL) -File scripts/prepare_support_copilot.ps1 -Force

prepare-skip-summary:
	$(POWERSHELL) -File scripts/prepare_support_copilot.ps1 -SkipSummary

pipeline:
	$(POWERSHELL) -File scripts/run_copilot_batch.ps1 -InputPath "$(INPUT)" $(if $(filter-out 0,$(LIMIT)),-Limit $(LIMIT),)

pipeline-reset:
	$(POWERSHELL) -File scripts/run_copilot_batch.ps1 -InputPath "$(INPUT)" -ResetFeedback $(if $(filter-out 0,$(LIMIT)),-Limit $(LIMIT),)

pipeline-one:
	$(POWERSHELL) -File scripts/run_copilot_batch.ps1 -InputPath "$(REPLY_TEST)" -ConversationId "$(CONVERSATION_ID)"

feedback-reply:
	$(POWERSHELL) -File scripts/run_feedback_cycle.ps1 -Target reply -InputPath "$(INPUT)" -ReplyTestPath "$(REPLY_TEST)" -ResetFeedback

feedback-summary:
	$(POWERSHELL) -File scripts/run_feedback_cycle.ps1 -Target summary -InputPath "$(INPUT)" -ReplyTestPath "$(REPLY_TEST)" -ResetFeedback

feedback-all:
	$(POWERSHELL) -File scripts/run_feedback_cycle.ps1 -Target all -InputPath "$(INPUT)" -ReplyTestPath "$(REPLY_TEST)" -ResetFeedback

eval-all:
	$(POWERSHELL) -File scripts/eval_all.ps1 -OutputPath "$(OUTPUT)"
