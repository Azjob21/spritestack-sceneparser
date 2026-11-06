# Model Checkpoints

This directory houses the weights and configuration files for both stages of the NLP Scene Parser pipeline:

## Pipeline Models

| Component | Model Name | Architecture | Size | Included in Git? | Purpose |
|---|---|---|---|---|---|
| **Model A** | `SpriteStack_Model_Slim_v2` | Fine-tuned DistilBERT Token Classification | ~265 MB | Tokenizer + config included (`model.safetensors` excluded via `.gitignore`) | Extracts game objects, counts, positions, themes from free-form natural language prompts |
| **Model B** | `final_conditional_architect.pth` | 4-layer Autoregressive Transformer | ~3.2 MB | **Yes** (committed directly) | Generates 16×16 tile grids conditioned on Model A's 27-dim spatial semantic blueprint vector |

## Checksums

Verify weight integrity using SHA-256:
```bash
sha256sum -c checksums.txt
```

## Adding Model A Safetensors

If `model.safetensors` (~265MB) is not present under `models/model_a/`:
1. The pipeline automatically falls back to an intelligent heuristic regex/rule-based NER parser.
2. To run the full fine-tuned DistilBERT model, download `model.safetensors` from GitHub Release assets or copy it from your training checkpoint:
   ```bash
   cp path/to/model.safetensors models/model_a/model.safetensors
   ```
