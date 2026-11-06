# Training & Dataset Generation

This directory contains standalone scripts used to generate the synthetic scene description datasets and train the models.

## Scripts

1. `model_a_generate_dataset.py`:
   Generates synthetic natural language scene descriptions with token-level BIO tags (`B-OBJECT`, `B-POSITION`, `B-SCENE_TYPE`, `B-COUNT`) covering combinations of game objects, counts, positions, and themes.

2. `model_b_train_transformer.py`:
   Kaggle / local training script for the `ConditionalArchitectTransformer` on the enriched Video Game Level Corpus (VGLC). Trains the 4-layer transformer on 256-token level grids conditioned on 27-dim spatial vectors.
