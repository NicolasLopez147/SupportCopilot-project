# SupportCopilot-LLM

Sprint 1 sets up the data foundation for a technical support copilot project focused on three future tasks:

- intent classification
- structured ticket summarization
- agent reply suggestion

The current repository already covers dataset download checks, raw dataset inspection, conversion to a unified schema, and interim-data validation.

## Current Scope

The repository now has two main layers:

- `src/experiments`: baselines, fine-tuning, evaluation, and synthetic-data experiments
- `src/copilot`: the integrated SupportCopilot pipeline, critics, retrieval, and feedback loop

The active copilot flow is centered on the synthetic support domain:

- synthetic support conversations in `data/synthetic`
- support KB documents in `data/kb`
- critic memory and retraining assets in `data/feedback`

Older bootstrap datasets such as `banking77`, `tweetsum`, and `customer_support_tweets` remain in the repository as historical experiment assets, but they are no longer the active reference for the support-domain pipeline.

All converted samples follow the shared `SupportSample` schema defined in [sample_schema.py](/C:/Users/nicol/Desktop/Projet%20support-copilot-llm/src/schemas/sample_schema.py).

## Project Structure

```text
archive/
  notebooks/
configs/
data/
  raw/
  interim/
  processed/
  synthetic/
  feedback/
    memory/
    candidates/
    augmented/
  kb/
outputs/
  experiments/
    intent/
    summary/
    reply/
    data_generation/
  copilot/
    runtime/
reports/
  tables/
src/
  copilot/
    critics/
    feedback/
    pipeline/
    retrieval/
      rag/
  data/
  experiments/
    baselines/
    eval/
    llm/
  schemas/
  utils/
services/
  gateway_service/
  intent_service/
  summary_service/
  reply_service/
shared/
  config/
  schemas/
  logging/
  utils/
ui/
```

## Environment Setup

Create and activate the virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Workflow Scripts

For the most common project workflows, use the PowerShell wrappers under `scripts/`.

Prepare the main pipeline artifacts:

```powershell
.\scripts\prepare_support_copilot.ps1
```

Launch the FastAPI service:

```powershell
.\scripts\run_api.ps1
```

Launch the Streamlit UI:

```powershell
.\scripts\run_ui.ps1
```

Run the integrated copilot pipeline with critics:

```powershell
.\scripts\run_copilot_batch.ps1 -ResetFeedback
```

Run the offline feedback loop and reevaluate:

```powershell
.\scripts\run_feedback_cycle.ps1 -Target all -ResetFeedback
```

Show a global evaluation overview for intent, summary, and reply:

```powershell
.\scripts\eval_all.ps1
```

A short usage guide is available in [scripts/README.md](/C:/Users/nicol/Desktop/Projet%20support-copilot-llm/scripts/README.md).

## Makefile Shortcuts

If you have `make` available, the repository root also provides a small `Makefile` that wraps the same operational workflows:

```powershell
make prepare
make api
make pipeline-reset
make feedback-all
make eval-all
```

Useful variants:

```powershell
make api-reload
make ui
make smoke
make pipeline-one CONVERSATION_ID=billing_vs_technical_routing_014
make pipeline-reset LIMIT=20
make eval-all OUTPUT=outputs/experiments/my_overview.json
```

## FastAPI Service

Sprint 7 introduces the first API layer of the microservices transition through the `gateway-service`.

Run it locally with:

```powershell
.\scripts\run_api.ps1
```

or:

```powershell
make api
```

Available endpoints:

- `GET /`
- `GET /api/v1/health`
- `POST /api/v1/copilot/run`
- `POST /api/v1/copilot/run/batch`

Interactive documentation is available at:

- `http://127.0.0.1:8000/docs`

Example request body for `POST /api/v1/copilot/run`:

```json
{
  "conversation_id": "demo_001",
  "scenario": "billing_vs_technical_routing",
  "persist_feedback": false,
  "messages": [
    {
      "speaker": "customer",
      "text": "My internet keeps freezing and I am not sure if this should go to billing or technical support."
    },
    {
      "speaker": "agent",
      "text": "I can help you figure that out. Can you tell me more about the service issue?"
    }
  ]
}
```

## Microservices Transition

The repository now also contains the first step of the microservices transition:

- `services/gateway_service`: current public entrypoint
- `services/intent_service`: implemented intent microservice
- `services/summary_service`: implemented summary microservice
- `services/reply_service`: implemented reply microservice
- `shared/`: shared config, schemas, logging, and utility helpers

At this stage, `gateway-service` runs in an embedded compatibility mode while the three downstream services are still being separated.

You can run the intent microservice independently with:

```powershell
.\scripts\run_intent_service.ps1
```

You can run the summary microservice independently with:

```powershell
.\scripts\run_summary_service.ps1
```

You can run the reply microservice independently with:

```powershell
.\scripts\run_reply_service.ps1
```

## Docker Deployment

The repository now includes a Docker-based local deployment for the full microservices stack.

Build and run all services with:

```powershell
make docker-up
```

This starts:

- `gateway-service`
- `intent-service`
- `summary-service`
- `reply-service`

The compose setup mounts local artifacts from:

- `data/`
- `outputs/`

This means:

- model artifacts must already exist locally in `outputs/experiments`
- KB files must already exist in `data/kb`

Useful commands:

```powershell
make docker-build
make docker-down
make docker-logs
```

## Streamlit UI

Sprint 8 adds a first user interface layer with Streamlit.

The UI talks only to `gateway-service`, which means it does not call downstream microservices directly.

Run it with:

```powershell
.\scripts\run_ui.ps1
```

or:

```powershell
make ui
```

Default access point:

- `http://127.0.0.1:8501`

The UI provides:

- production-style conversation input
- hidden auto-generated conversation identifier
- optional advanced scenario override for debugging
- gateway health check
- final intent, summary, and reply display
- critic and fallback visibility for technical demos

## CI/CD

Sprint 9 adds a minimal CI/CD layer with GitHub Actions.

The workflow currently performs:

- dependency installation
- Python source compilation
- import and API smoke tests
- Docker Compose configuration validation

The workflow file is located in:

- `.github/workflows/ci.yml`

You can also run the local smoke checks with:

```powershell
make smoke
```

## Dataset Setup

### 1. Hugging Face datasets

These are handled by the project script:

- `PolyAI/banking77`
- `MohammadOthman/mo-customer-support-tweets-945k`

Run:

```powershell
.\.venv\Scripts\python.exe -m src.data.download_datasets
```

### 2. External `tweetsum` dataset

Expected files in [data/raw/tweetsum](/C:/Users/nicol/Desktop/Projet%20support-copilot-llm/data/raw/tweetsum):

- `train.jsonl`
- `valid.jsonl`
- `test.jsonl`
- `tweet_sum_processor.py`
- `twcs.csv`

`tweetsum` itself comes from the official repository:

- [TweetSumm repository](https://github.com/guyfe/Tweetsumm)

`twcs.csv` must come from the original Kaggle dataset used by TweetSumm:

- [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)

After placing those files in the expected folder, verify everything with:

```powershell
.\.venv\Scripts\python.exe -m src.data.download_datasets
```

## Data Pipeline

Inspect raw datasets:

```powershell
.\.venv\Scripts\python.exe -m src.data.inspect_datasets
```

Build the unified interim datasets:

```powershell
.\.venv\Scripts\python.exe -m src.data.build_unified_dataset
```

Validate generated `jsonl` files:

```powershell
.\.venv\Scripts\python.exe -m src.data.validate_interim_datasets
```

## Sprint 1 Output

At the end of Sprint 1, the project should produce:

- interim intent data from `banking77`
- interim summarization data from `tweetsum`
- interim noisy conversation data from `customer_support_tweets`
- schema validation reports for all generated `jsonl` files

## Sprint 2 Results

### Intent Classification Baselines

Results on `banking77`:

| Model | Representation | Classifier | Accuracy | Macro F1 |
|---|---|---|---:|---:|
| TF-IDF baseline | TF-IDF (`ngram_range=(1,2)`) | Logistic Regression | 0.8578 | 0.8568 |
| Embedding baseline | `all-MiniLM-L6-v2` sentence embeddings | Logistic Regression | 0.9081 | 0.9081 |

Initial conclusion:

> Replacing TF-IDF with pretrained sentence embeddings improved both accuracy and macro F1 by about 5 points, suggesting that semantic text representations capture intent distinctions more effectively than purely lexical features on Banking77.

### Dialogue Summarization Baseline

For the first summarization baseline, the project uses:

- dataset: `tweetsum`
- model: `philschmid/bart-large-cnn-samsum`
- task setup: conversation-to-summary generation

Preprocessing includes:

- HTML decoding
- URL removal
- handle removal
- placeholder normalization
- emoji removal

Automatic evaluation currently combines:

- `BERTScore` as the main semantic metric
- `source-grounded entailment` as a complementary factual-support signal
- a small manual review template for qualitative inspection

Current summary evaluation results on a small test subset:

| Model | Dataset | Samples | BERTScore F1 | Source Support | Notes |
|---|---|---:|---:|---:|---|
| `philschmid/bart-large-cnn-samsum` | `tweetsum` | 10 | 0.8761 | 0.6056 | Reasonable baseline, tends to be useful but sometimes verbose |

Summary interpretation:

> The dialogue summarization baseline produces semantically close summaries according to BERTScore and generally reasonable outputs in manual inspection. The source-grounded entailment score is kept as a complementary experimental metric rather than the primary success criterion.

### Sprint 2 Reproduction

Run the intent baselines:

```powershell
python -m src.experiments.baselines.train_tfidf_intent
python -m src.experiments.baselines.train_embedding_intent
```

Run the summarization baseline:

```powershell
python -m src.experiments.baselines.api_summary
```

Run summary evaluation:

```powershell
python -m src.experiments.eval.eval_summary
```

### Sprint 2 Status

Sprint 2 is considered complete with:

- two reproducible intent classification baselines
- one local dialogue summarization baseline
- automatic evaluation outputs for intent and summary
- a manual review template for summary quality checks

## Sprint 3 Results

### Summarization Fine-Tuning

Sprint 3 focuses on improving dialogue summarization over the Sprint 2 baseline using:

- full fine-tuning of `google/flan-t5-small`
- parameter-efficient fine-tuning with `LoRA` on the same base model

Current comparison on the same `tweetsum` test subset:

| Model | Setup | Samples | BERTScore F1 | Source Support | Notes |
|---|---|---:|---:|---:|---|
| `philschmid/bart-large-cnn-samsum` | Baseline | 10 | 0.8761 | 0.6056 | Strong zero-shot/local baseline |
| `google/flan-t5-small` | Full fine-tuning | 10 | 0.8870 | 0.9827 | Best source-grounded support on the evaluation subset |
| `google/flan-t5-small` | LoRA fine-tuning | 10 | 0.8904 | 0.9023 | Best BERTScore, competitive with full fine-tuning |

Sprint 3 interpretation:

> Both fine-tuning strategies improved over the baseline summarization model. Full fine-tuning produced the strongest source-grounded support score, while LoRA achieved the best semantic similarity score with a lighter adaptation strategy.

### Sprint 3 Reproduction

Train the full fine-tuning model:

```powershell
python -m src.experiments.llm.train_full_summary
python -m src.experiments.llm.generate_full_summary_predictions
python -m src.experiments.eval.eval_summary "outputs\experiments\summary\full_finetune\test_predictions.json"
```

Train the LoRA model:

```powershell
python -m src.experiments.llm.train_lora_summary
python -m src.experiments.llm.generate_lora_summary_predictions
python -m src.experiments.eval.eval_summary "outputs\experiments\summary\lora_base\test_predictions.json"
```

### Sprint 3 Status

Sprint 3 is considered complete with:

- one full fine-tuning pipeline for summarization
- one LoRA fine-tuning pipeline for summarization
- reproducible prediction scripts for both
- comparable evaluation outputs against the Sprint 2 baseline

## Sprint 4 Results

### Suggested Reply Generation

Sprint 4 focuses on the agent reply suggestion component. Because the project KB is technical-support oriented and not well aligned with the original `tweetsum` conversations, this sprint uses a synthetic benchmark built from the support KB scenarios.

The synthetic reply dataset was:

- generated locally with `ollama` using `llama3.2`
- grounded in the KB scenarios under [data/kb](/C:/Users/nicol/Desktop/Projet%20support-copilot-llm/data/kb)
- later split into train / valid / test for reply-generation experiments

The current synthetic dataset contains:

- 115 total conversations
- 85 train examples
- 15 validation examples
- 15 test examples

Three reply-generation strategies were compared:

1. `google/flan-t5-small` baseline
2. `google/flan-t5-small` with semantic retrieval from the KB
3. `google/flan-t5-small` with LoRA fine-tuning on the synthetic reply dataset

Current comparison on the synthetic reply test set:

| Method | Model | Samples | BERTScore F1 | Notes |
|---|---|---:|---:|---|
| Baseline reply generation | `google/flan-t5-small` | 15 | 0.8615 | Reasonable baseline without external context |
| Retrieval-grounded reply generation | `google/flan-t5-small` + KB retrieval | 15 | 0.8580 | Retrieval did not improve results consistently in this setup |
| LoRA fine-tuned reply generation | `google/flan-t5-small` + LoRA | 15 | 0.8626 | Best overall result on the synthetic benchmark |

Sample-level comparison:

- baseline wins: `4`
- retrieval wins: `4`
- LoRA wins: `6`
- ties: `1`

Sprint 4 interpretation:

> On the synthetic support benchmark, LoRA fine-tuning produced the best reply-generation performance, though only by a small margin over the plain T5 baseline. Retrieval-based generation remained competitive but did not show a clear advantage in the current prompt-and-retrieval configuration.

### Sprint 4 Reproduction

Generate or expand the synthetic benchmark:

```powershell
python -m src.data.generate_synthetic_reply_dataset --model llama3.2 --samples-per-scenario 15
python -m src.data.rewrite_synthetic_references --input-path "data\synthetic\synthetic_reply_eval.jsonl" --output-path "data\synthetic\synthetic_reply_eval_rewritten.jsonl" --model llama3.2
```

Train LoRA for reply generation:

```powershell
python -m src.experiments.llm.train_lora_reply --input-path "data\synthetic\synthetic_reply_eval_rewritten.jsonl"
```

Generate reply predictions:

```powershell
python -m src.copilot.retrieval.rag.generate_reply_baseline --input-path "data\synthetic\reply_test.jsonl"
python -m src.copilot.retrieval.rag.generate_reply_with_retrieval --input-path "data\synthetic\reply_test.jsonl"
python -m src.experiments.llm.generate_lora_reply_predictions --input-path "data\synthetic\reply_test.jsonl"
```

Evaluate the three methods:

```powershell
python -m src.experiments.eval.eval_reply_methods
```

### Sprint 4 Status

Sprint 4 is considered complete with:

- a compact technical-support KB for retrieval grounding
- semantic indexing and local retrieval over KB chunks
- a synthetic reply-generation benchmark aligned with the KB scenarios
- three comparable reply-generation methods
- automatic comparison outputs and a manual review template for reply quality

## Sprint 5 Results

### End-to-End SupportCopilot Pipeline

Sprint 5 focused on integrating the strongest available components into a single technical-support copilot pipeline that can process a conversation and return:

- predicted intent
- conversation summary
- suggested next agent reply

The integrated pipeline uses:

- synthetic-support intent classification with `all-MiniLM-L6-v2` embeddings + Logistic Regression
- LoRA summarization with `google/flan-t5-small`
- LoRA reply generation with `google/flan-t5-small`

The main outcome of Sprint 5 was architectural rather than purely metric-based:

- the pipeline ran end to end on synthetic support conversations
- intent prediction was coherent on the synthetic-support domain
- summary and reply generation exposed important quality limitations
- these weaknesses motivated the addition of critic and fallback layers in the next sprint

Sprint 5 interpretation:

> The end-to-end integration confirmed that the project components could be chained into a single SupportCopilot workflow. At the same time, it showed that raw generative quality was still not strong enough for reliable final outputs, especially for reply generation, which created the need for a critic-driven improvement loop.

### Sprint 5 Reproduction

Train the synthetic-support intent classifier:

```powershell
python -m src.experiments.baselines.train_synthetic_intent
```

Run the integrated pipeline:

```powershell
python -m src.copilot.pipeline.run_support_copilot --input-path "data\synthetic\reply_test.jsonl"
```

### Sprint 5 Status

Sprint 5 is considered complete with:

- a first end-to-end SupportCopilot pipeline
- synthetic-support intent classification integrated into the pipeline
- summarization and reply generation integrated into the same workflow
- evidence that the raw pipeline required quality-control layers before iterative improvement

## Sprint 6 Results

### Agentic Critics and Offline Feedback Loop

Sprint 6 introduced an agentic quality-control layer on top of the SupportCopilot pipeline. Instead of trusting the raw outputs directly, the system now evaluates them with critics, applies safer fallbacks when necessary, stores failures in memory, and uses those failures to build retraining batches for later offline improvement.

The Sprint 6 architecture now includes:

- `intent_critic`
- `summary_critic`
- `reply_critic`
- fallback handling for low-quality outputs
- failure memory for intent, summary, and reply
- retraining-set construction from critic failures
- augmented retraining batches
- incremental LoRA retraining for reply generation

This created a full improvement loop:

1. run the pipeline over many conversations
2. detect low-quality outputs with critics
3. store failures in memory
4. convert failures into retraining candidates
5. merge candidates into augmented training batches
6. continue LoRA reply training from the existing adapter
7. compare the new adapter against the previous methods

Final reply-generation comparison after the feedback loop:

| Method | Model | Samples | BERTScore F1 | Notes |
|---|---|---:|---:|---|
| Baseline reply generation | `google/flan-t5-small` | 15 | 0.8615 | Plain generation baseline |
| Retrieval-grounded reply generation | `google/flan-t5-small` + KB retrieval | 15 | 0.8580 | Competitive, but still below the strongest models |
| LoRA fine-tuned reply generation | `google/flan-t5-small` + LoRA | 15 | 0.8626 | Strongest model before feedback retraining |
| Feedback-guided LoRA reply generation | `google/flan-t5-small` + LoRA + critic-memory retraining | 15 | 0.8712 | Best final result after the offline feedback loop |

Sample-level comparison after feedback retraining:

- baseline wins: `1`
- retrieval wins: `2`
- LoRA wins: `4`
- feedback-LoRA wins: `7`
- ties: `1`

Sprint 6 interpretation:

> The critic-driven offline retraining loop improved the reply-generation model beyond the original LoRA baseline and all earlier methods. This validates the agentic architecture: critics and failure memory did not just make the pipeline safer at inference time, they also produced useful supervision signals for iterative improvement.

### Sprint 6 Reproduction

Run the pipeline over many conversations and populate failure memory:

```powershell
python -m src.copilot.pipeline.run_support_copilot --input-path "data\synthetic\synthetic_reply_eval.jsonl" --run-all --reset-feedback --output-path "outputs\copilot\runtime\pipeline_full_run.json"
```

Build retraining candidates and augmented training batches:

```powershell
python -m src.copilot.feedback.build_retraining_sets
python -m src.copilot.feedback.build_augmented_training_sets
```

Continue LoRA reply training from the previous adapter:

```powershell
python -m src.experiments.llm.train_lora_reply_feedback
python -m src.experiments.llm.generate_lora_reply_predictions --adapter-dir "outputs\experiments\reply\lora_feedback\final_model" --output-path "outputs\experiments\reply\lora_feedback\test_replies.json"
```

Evaluate all methods, including the feedback-retrained LoRA:

```powershell
python -m src.experiments.eval.eval_reply_methods --lora-feedback-path "outputs\experiments\reply\lora_feedback\test_replies.json"
```

### Sprint 6 Status

Sprint 6 is considered complete with:

- a critic layer for intent, summary, and reply generation
- fallback handling integrated into the pipeline
- persistent failure memory for all three tasks
- scripts to build retraining candidates and augmented training sets
- incremental feedback-guided LoRA retraining for reply generation
- a measurable improvement over the previous best reply-generation model

## Notes

- `tweetsum` reconstruction depends on the correct `twcs.csv` version from the original Thought Vector Kaggle dataset.
- `customer_support_tweets` currently provides a `train` split only, which is expected for this stage.
- The next logical step is to decide whether to extend the same feedback loop to summarization and intent retraining, or to move toward a more production-oriented evaluation on less synthetic support conversations.
