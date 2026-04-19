PYTHON := .venv/Scripts/python.exe
POWERSHELL := powershell -NoProfile -ExecutionPolicy Bypass

INPUT ?= data/synthetic/synthetic_reply_eval.jsonl
REPLY_TEST ?= data/synthetic/reply_test.jsonl
OUTPUT ?= outputs/experiments/comparison_overview.json
CONVERSATION_ID ?=
LIMIT ?= 0

.PHONY: help prepare prepare-force prepare-skip-summary api api-reload api-service api-service-reload ui intent-service intent-service-reload summary-service summary-service-reload reply-service reply-service-reload docker-build docker-up docker-down docker-logs smoke pipeline pipeline-reset pipeline-one feedback-reply feedback-summary feedback-all eval-all

help:
	@echo Cibles disponibles :
	@echo   make prepare              - Préparer les artefacts principaux de SupportCopilot
	@echo   make prepare-force        - Réentraîner / reconstruire tous les artefacts principaux
	@echo   make prepare-skip-summary - Préparer sans le modèle de résumé de base
	@echo   make api                  - Lancer le gateway-service en mode embedded
	@echo   make api-reload           - Lancer le gateway-service en mode embedded avec rechargement
	@echo   make api-service          - Lancer le gateway-service en mode orchestration HTTP
	@echo   make api-service-reload   - Lancer le gateway-service en mode orchestration HTTP avec rechargement
	@echo   make ui                   - Lancer l'interface Streamlit
	@echo   make intent-service       - Lancer le microservice d'intention
	@echo   make intent-service-reload - Lancer le microservice d'intention avec rechargement
	@echo   make summary-service      - Lancer le microservice de résumé
	@echo   make summary-service-reload - Lancer le microservice de résumé avec rechargement
	@echo   make reply-service        - Lancer le microservice de réponse
	@echo   make reply-service-reload - Lancer le microservice de réponse avec rechargement
	@echo   make docker-build         - Construire les images Docker
	@echo   make docker-up            - Lancer toute l'architecture avec Docker Compose
	@echo   make docker-down          - Arrêter l'architecture Docker Compose
	@echo   make docker-logs          - Voir les logs Docker Compose
	@echo   make smoke                - Executer les smoke tests locaux
	@echo   make pipeline             - Exécuter le pipeline batch sans réinitialiser la mémoire de feedback
	@echo   make pipeline-reset       - Exécuter le pipeline batch et réinitialiser la mémoire de feedback
	@echo   make pipeline-one CONVERSATION_ID=... - Exécuter le pipeline pour une seule conversation
	@echo   make feedback-reply       - Lancer la boucle complète de feedback pour la réponse
	@echo   make feedback-summary     - Lancer la boucle complète de feedback pour le résumé
	@echo   make feedback-all         - Lancer la boucle complète de feedback pour réponse et résumé
	@echo   make eval-all             - Afficher la vue globale d'évaluation

prepare:
	$(POWERSHELL) -File scripts/prepare_support_copilot.ps1

prepare-force:
	$(POWERSHELL) -File scripts/prepare_support_copilot.ps1 -Force

prepare-skip-summary:
	$(POWERSHELL) -File scripts/prepare_support_copilot.ps1 -SkipSummary

api:
	$(POWERSHELL) -File scripts/run_api.ps1

api-reload:
	$(POWERSHELL) -File scripts/run_api.ps1 -Reload

api-service:
	$(POWERSHELL) -File scripts/run_api.ps1 -ServiceMode

api-service-reload:
	$(POWERSHELL) -File scripts/run_api.ps1 -ServiceMode -Reload

ui:
	$(POWERSHELL) -File scripts/run_ui.ps1

intent-service:
	$(POWERSHELL) -File scripts/run_intent_service.ps1

intent-service-reload:
	$(POWERSHELL) -File scripts/run_intent_service.ps1 -Reload

summary-service:
	$(POWERSHELL) -File scripts/run_summary_service.ps1

summary-service-reload:
	$(POWERSHELL) -File scripts/run_summary_service.ps1 -Reload

reply-service:
	$(POWERSHELL) -File scripts/run_reply_service.ps1

reply-service-reload:
	$(POWERSHELL) -File scripts/run_reply_service.ps1 -Reload

docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

smoke:
	$(PYTHON) scripts/smoke_test_services.py

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
