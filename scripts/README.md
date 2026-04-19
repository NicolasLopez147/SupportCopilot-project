# Scripts

Ces scripts couvrent les flux operationnels les plus frequents du projet.

Un `Makefile` existe aussi a la racine pour lancer les memes flux avec des commandes courtes comme :

```powershell
make prepare
make api
make api-service
make ui
make intent-service
make summary-service
make reply-service
make docker-up
make smoke
make pipeline-reset
make feedback-all
make eval-all
```

Sous Windows, cela fonctionne si `make` est disponible via Git Bash, MSYS2, Chocolatey, Scoop ou WSL.

## 1. Preparer les artefacts de base du pipeline

Entraine ou reutilise :

- le classifieur synthetique d'intention
- le modele LoRA de base pour le resume
- le modele LoRA de base pour la reponse

```powershell
.\scripts\prepare_support_copilot.ps1
```

Options utiles :

```powershell
.\scripts\prepare_support_copilot.ps1 -Force
.\scripts\prepare_support_copilot.ps1 -SkipSummary
```

## 2. Lancer l'API FastAPI

Expose le pipeline du copilote avec une API REST et la documentation Swagger integree.

```powershell
.\scripts\run_api.ps1
```

Avec rechargement automatique :

```powershell
.\scripts\run_api.ps1 -Reload
```

En mode orchestration HTTP vers les trois microservices :

```powershell
.\scripts\run_api.ps1 -ServiceMode
```

Points d'acces utiles :

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/api/v1/health`
- `http://127.0.0.1:8000/api/v1/copilot/run`

## 3. Lancer le microservice d'intention

Expose le classifieur d'intention comme service independant.

```powershell
.\scripts\run_intent_service.ps1
```

Avec rechargement automatique :

```powershell
.\scripts\run_intent_service.ps1 -Reload
```

Points d'acces utiles :

- `http://127.0.0.1:8001/docs`
- `http://127.0.0.1:8001/api/v1/health`
- `http://127.0.0.1:8001/api/v1/intent`

## 4. Lancer l'interface Streamlit

L'interface utilisateur parle uniquement au `gateway-service`.

```powershell
.\scripts\run_ui.ps1
```

Point d'acces utile :

- `http://127.0.0.1:8501`

## 5. Lancer le microservice de resume

Expose le generateur de resume comme service independant.

```powershell
.\scripts\run_summary_service.ps1
```

Avec rechargement automatique :

```powershell
.\scripts\run_summary_service.ps1 -Reload
```

Points d'acces utiles :

- `http://127.0.0.1:8002/docs`
- `http://127.0.0.1:8002/api/v1/health`
- `http://127.0.0.1:8002/api/v1/summary`

## 6. Lancer le microservice de reponse

Expose le generateur de reponse comme service independant.

```powershell
.\scripts\run_reply_service.ps1
```

Avec rechargement automatique :

```powershell
.\scripts\run_reply_service.ps1 -Reload
```

Points d'acces utiles :

- `http://127.0.0.1:8003/docs`
- `http://127.0.0.1:8003/api/v1/health`
- `http://127.0.0.1:8003/api/v1/reply`

## 7. Executer le pipeline complet en lot

Lance intention + resume + reponse + critiques et enregistre la memoire des echecs.

```powershell
.\scripts\run_copilot_batch.ps1 -ResetFeedback
```

Avec une limite :

```powershell
.\scripts\run_copilot_batch.ps1 -Limit 20 -ResetFeedback
```

Avec une conversation precise :

```powershell
.\scripts\run_copilot_batch.ps1 -ConversationId "billing_vs_technical_routing_014"
```

## 8. Lancer l'architecture complete avec Docker

Construire et lancer les quatre services avec Docker Compose :

```powershell
make docker-up
```

Construire uniquement :

```powershell
make docker-build
```

Arreter les conteneurs :

```powershell
make docker-down
```

Voir les logs :

```powershell
make docker-logs
```

Important :

- les artefacts de modeles doivent deja exister localement dans `outputs/experiments`
- les documents KB doivent exister dans `data/kb`
- ces dossiers sont montes dans les conteneurs via `docker-compose.yml`

## 9. Executer les smoke tests locaux

Ces checks valident :

- les imports principaux
- les endpoints `/` et `/api/v1/health`
- la propagation du header `X-Request-ID`
- l'import de l'interface Streamlit

```powershell
make smoke
```

## 10. Executer la boucle complete de feedback

Cette commande :

1. execute le pipeline avec les critiques
2. reconstruit les candidats et les jeux augmentes
3. reentraine le modele de feedback
4. regenere les predictions
5. reevalue

Reponse uniquement :

```powershell
.\scripts\run_feedback_cycle.ps1 -Target reply -ResetFeedback
```

Resume uniquement :

```powershell
.\scripts\run_feedback_cycle.ps1 -Target summary -ResetFeedback
```

Reponse + resume :

```powershell
.\scripts\run_feedback_cycle.ps1 -Target all -ResetFeedback
```

## 11. Voir une evaluation globale du systeme

Ce script resume en une seule sortie :

- les metriques du classifieur d'intention
- la comparaison des resumes
- la comparaison des reponses
- avec ou sans feedback, selon les artefacts disponibles

```powershell
.\scripts\eval_all.ps1
```

Si vous voulez enregistrer l'overview dans un autre chemin :

```powershell
.\scripts\eval_all.ps1 -OutputPath "outputs\experiments\my_overview.json"
```
