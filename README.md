# VAE-Lipreading

Lip-reading project built for CS 179. The repository contains a unified notebook pipeline for training a VAE on video segments, exporting latent features, and evaluating several downstream classifiers.

## Project Summary

The final project uses a pretrained 3D ResNet encoder inside a Variational Autoencoder to learn lip-reading features from GRID word-level video clips. The VAE produces latent means and variances, and those features are then used by three classifier families:

- an MLP on latent features
- a DNN on sampled latent points
- XGBoost on the same sampled latent points

We also include a separate path for precomputed vector features, which lets you skip the VAE when features already exist.

For the full project description, see [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

## Recommended Notebook

Use [merged_lipreading_pipeline.ipynb](merged_lipreading_pipeline.ipynb) as the main notebook. It combines the workflows that were previously split across multiple notebooks and lets you switch behavior with `RUN_MODE`.

Legacy notebooks are still present under `Supplemental Materials/`, `data_for_179/`, and `new_data/` for reference, but the merged notebook is the canonical entry point.

## Repository Layout

- [merged_lipreading_pipeline.ipynb](merged_lipreading_pipeline.ipynb) - unified notebook for VAE training, latent export, and classifier experiments
- [Supplemental Materials/](Supplemental%20Materials) - earlier notebook versions and supporting materials
- [data_for_179/](data_for_179) - notebook variants tied to the CS 179 submission workflow
- [new_data/](new_data) - notebook variants for later latent/vector experiments
- [requirements.txt](requirements.txt) - Python dependencies

## Dataset

The project uses the GRID corpus from Sheffield: https://spandh.dcs.shef.ac.uk/gridcorpus/

Each sample consists of a short MPG video and an `.align` file that marks which word is spoken at which timestamps. The subset used in the paper is based on 10 speakers covering all 53 words, with additional out-of-distribution testing on other speakers.

## Results From the Paper

- VAE training time: about 93 minutes
- Best MLP test accuracy on latent features: 77.3%
- Out-of-distribution MLP accuracy: 52.4%
- DNN test accuracy on sampled latent points: 24.87%
- XGBoost test accuracy on sampled latent points: 24.6%

The key takeaway from the write-up is that the latent mean and log-variance were more useful than stochastic sampling for classification.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## How To Run

1. Open [merged_lipreading_pipeline.ipynb](merged_lipreading_pipeline.ipynb).
2. Set `RUN_MODE` in the config cell:
	- `vae_train_export`
	- `mlp_from_latent`
	- `dnn_xgb_from_latent`
	- `mlp_from_vector`
3. Run the notebook top to bottom.

If latent files do not exist yet, run `vae_train_export` first so the downstream classifiers have features to load.


