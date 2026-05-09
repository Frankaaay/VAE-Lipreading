# VAE-Lipreading

Lip-reading project built for CS 179. This repository contains an end-to-end pipeline for training a Variational Autoencoder (VAE) on video segments, extracting latent features, and evaluating downstream classifiers (MLP, DNN, XGBoost) to perform word-level lipreading.

## Project Summary

We use a pretrained 3D ResNet encoder inside a Variational Autoencoder to learn spatiotemporal lip-reading features from GRID word-level video clips. The VAE produces latent means and variances. These structured, deterministic features are then used to train an **MLP Classifier** which achieved **77.3% test accuracy** and **52.4% out-of-distribution accuracy** across 53 vocabulary words.

## Repository Structure

- `src/config.py`: Centralized hyperparameters and configurations.
- `src/dataset.py`: OpenCV video processing, PyTorch datasets, and dataloaders.
- `src/models.py`: Architectures for the VAE (ResNet3D-18 backbone), MLP, and DNN.
- `src/train.py`: CLI script to run training for either the VAE or the classifiers.
- `src/utils.py`: Helper functions for metrics and evaluating dataloaders.
- `app.py`: Streamlit frontend for deployment/inference showcase.
- `merged_lipreading_pipeline.ipynb`: (Optional) Jupyter notebook version of the entire pipeline.
- `Project Thesis.md`: Detailed project report and findings.

## How to Train (Locally or Google Colab)

You can easily train this model without a local GPU by running it on Google Colab directly from your GitHub repository.

1. Open a new Google Colab notebook and set the runtime to a GPU (e.g., T4).
2. Clone your repository and navigate into it:
   ```bash
   !git clone https://github.com/YOUR_USERNAME/VAE-Lipreading.git
   %cd VAE-Lipreading
   !pip install -r requirements.txt
   ```
3. Run the VAE training to learn features and export latent representations:
   ```bash
   !python -m src.train --mode vae
   ```
4. Run the classifier training on the extracted latent features:
   ```bash
   !python -m src.train --mode mlp
   ```
5. Once trained, download the `vae_trained.pt` and `MLPClassifier_full.pt` models to use in your web app.

## Web App Showcase

We provide a Streamlit frontend (`app.py`) to easily demonstrate the model. 

1. Install Streamlit:
   ```bash
   pip install streamlit
   ```
2. Place your trained `.pt` weight files in the repository root.
3. Run the app:
   ```bash
   streamlit run app.py
   ```

*Note: This app is optimized for deployment on Hugging Face Spaces for portfolio showcases.*
