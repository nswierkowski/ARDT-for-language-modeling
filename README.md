# ARDT‑Language

This repository implements the ideas from *On the Power of Decision Trees in Auto‑Regressive Language Modeling* and extends them with a number of additional utilities and modalities (music, audio, tabular features, wavelets, …).  The original paper shows that a simple autoregressive decision‑tree model (ARDT) trained on context vectors can compete surprisingly well with standard neural sequence models.  Here you will find a **Python** reproduction of that work together with scripts, datasets and helper functions to train, evaluate and play with ARDT models.

---

## Features

- **Core ARDT model** (`src/models/ardt.py`)
  - classification or regression mode
  - context vector computed by exponentially decayed average of embeddings
  - support for `decision_tree`, `random_forest`, `xgboost`, `lightgbm`
  - interactive generation CLI (see `main.py`)
  - utilities for perplexity, BLEU, path‑embedding extraction
- **Kernel estimator** for sklearn hyperparameter tuning (`ARDTEstimator`) used by `nested_train_model`
- **Simple dataset loader** (`ARDTSimpleDataset`) that converts texts to `(context, target)` pairs
- **Vocabulary & embedding helpers** (`utils/handler_functions.py`)
  - build vocab from raw texts
  - load GloVe, GPT‑2 or MiniLM embeddings (with caching)
  - compute target vectors
- **Audio / music support**
  - `AudioARDTDataset` builds MFCC‑based contexts for waveform prediction
  - `train_music_model_with_ardt` & `compute_mcd` in `main.py`
- **Wave‑based model** using wavelet features (`wave_config.yaml` and related code)
- **Tabular extension** (`src/models/TabularARDT.py` + `src/scripts/run_tabular_ardt.py`)
  - combine one‑hot/tabular features with autoregressive context
  - example on `MidiCaps` dataset with genre conditioning
- **Tree‑embedding utilities** (`src/tree_embedding/embedding.py`)
- Convenience scripts: `run_ardt_classifier.py`, `run_tabular_ardt.py` for training/generation


## Installation

```bash
# clone the repository
git clone <this-repo-url>
cd ARDT-language

# create a Python environment (suggested: venv or conda)
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

> You will need `ffmpeg`/`librosa` installed for audio experiments and a GPU for large datasets.


## Directory Structure

```
ARDT-language/
├── config.yaml           # example config for text
├── music_config.yaml     # example config for music
├── wave_config.yaml      # example config for wavelet data
├── main.py               # high‑level training/interactive script
├── requirements.txt
├── src/
│   ├── datasets/         # dataset helpers
│   ├── models/           # ARDT and variants
│   ├── scripts/          # CLI entrypoints (tabular, classifier, etc.)
│   ├── tree_embedding/   # tree code used for special embeddings
│   └── wav_facade/       # wave utilities
├── utils/                # general helper functions
└── ...
```


## Usage Examples

### Training a language model

```bash
python main.py train_model config.yaml
```

Edit `config.yaml` (or create your own) to point at a dataset, specify embedding type, model type, context size, etc.  The script saves:

- model parameters (`ardt_model.pkl`)
- vocabulary (`word2idx.yaml`)
- embeddings (`embeddings.npy`)
- training statistics (`results.json`)

You can also run `nested_train_model` to perform hyperparameter search with `HalvingRandomSearchCV`.

### Interactive generation

```bash
python main.py run_model config.yaml
# then type prompts at the >>> prompt
```

Control special commands:
- `~max_len=N` to change generation length
- `~exit` to quit

### Music / Audio

```bash
python main.py train_music_model_with_ardt music_config.yaml
```

The audio pipeline uses MFCC features and computes Mel‑Cepstral Distortion (MCD) for evaluation.

### Wavelet / ECG dataset

Analogous interface via `wave_config.yaml`.  See `main.py:train_music_model_with_ardt` for details.

### Tabular Experiments

```bash
python src/scripts/run_tabular_ardt.py train --dataset benjamintli/MidiCaps-preprocessed \
    --context_size 16 --alpha 0.5 --embed_dim 128 --model_type lightgbm

# generate from saved model:
python src/scripts/run_tabular_ardt.py generate --model_path tabular_ardt.pkl \
    --genre Jazz --length 50
```

This script demonstrates conditioning generation on tabular labels (e.g. music genre).

---

## Configuration

All high‑level scripts expect a YAML file with two top‑level sections (`train_model` and `run_model`).  See the example configs in the repo for possible keys.

Key parameters include:
- `dataset_name` / `texts` – HuggingFace dataset or plain text file
- `embedding` – path to GloVe file (or use `embedding_type=gpt2|minilm|tree`)
- `model_type` – one of `decision_tree`, `random_forest`, `xgboost`, `lightgbm`
- `model_params` – passed to the underlying sklearn/XGBoost/LGBM estimator
- `context_size`, `alpha` – autoregressive context settings


## Extending the Code

- To add a new dataset: implement a `torch.utils.data.Dataset` that yields `(context, target)` pairs.
- To plug in a new embedding source, modify `utils/handler_functions.py`.
- The ARDT class is framework‑agnostic; you can call `fit(..)` on any iterable of context vectors and next‑token labels.


## References

- **Paper**: On the Power of Decision Trees in Auto‑Regressive Language Modeling.  [Link]([https://arxiv.org/abs/XXXX](https://arxiv.org/abs/2409.19150))  (replace with the actual link).

---

**Have fun experimenting!** 🎯
