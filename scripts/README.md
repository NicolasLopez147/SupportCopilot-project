# Scripts

Ces scripts couvrent les flux opérationnels les plus fréquents du projet.

Un `Makefile` existe aussi à la racine pour lancer les mêmes flux avec des commandes courtes comme :

```powershell
make prepare
make pipeline-reset
make feedback-all
make eval-all
```

Sous Windows, cela fonctionne si `make` est disponible via Git Bash, MSYS2, Chocolatey, Scoop ou WSL.

## 1. Préparer les artefacts de base du pipeline

Entraîne ou réutilise :

- le classifieur synthétique d'intention
- le modèle LoRA de base pour le résumé
- le modèle LoRA de base pour la réponse

```powershell
.\scripts\prepare_support_copilot.ps1
```

Options utiles :

```powershell
.\scripts\prepare_support_copilot.ps1 -Force
.\scripts\prepare_support_copilot.ps1 -SkipSummary
```

## 2. Exécuter le pipeline complet en lot

Lance intention + résumé + réponse + critiques et enregistre la mémoire des échecs.

```powershell
.\scripts\run_copilot_batch.ps1 -ResetFeedback
```

Avec une limite :

```powershell
.\scripts\run_copilot_batch.ps1 -Limit 20 -ResetFeedback
```

Avec une conversation précise :

```powershell
.\scripts\run_copilot_batch.ps1 -ConversationId "billing_vs_technical_routing_014"
```

## 3. Exécuter la boucle complète de feedback

Cette commande :

1. exécute le pipeline avec les critiques
2. reconstruit les candidats et les jeux augmentés
3. réentraîne le modèle de feedback
4. régénère les prédictions
5. réévalue

Réponse uniquement :

```powershell
.\scripts\run_feedback_cycle.ps1 -Target reply -ResetFeedback
```

Résumé uniquement :

```powershell
.\scripts\run_feedback_cycle.ps1 -Target summary -ResetFeedback
```

Réponse + résumé :

```powershell
.\scripts\run_feedback_cycle.ps1 -Target all -ResetFeedback
```

## 4. Voir une évaluation globale du système

Ce script résume en une seule sortie :

- les métriques du classifieur d'intention
- la comparaison des résumés
- la comparaison des réponses
- avec ou sans feedback, selon les artefacts disponibles

```powershell
.\scripts\eval_all.ps1
```

Si vous voulez enregistrer l'overview dans un autre chemin :

```powershell
.\scripts\eval_all.ps1 -OutputPath "outputs\experiments\my_overview.json"
```
