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
   !pip install -r requirements.txt kagglehub
   ```
3. Download the dataset and run the VAE training to learn features and export latent representations:
   ```python
   import kagglehub
   import os

   # This downloads the 15GB lipreading dataset and returns the path
   dataset_path = kagglehub.dataset_download("mohamedbentalb/lipreading-dataset")
   print("Dataset downloaded to:", dataset_path)

   # Pass the dataset_path directly via a simple environment variable
   os.environ["DATA_ROOT"] = dataset_path
   ```
   ```bash
   !sed -i 's/DATA_ROOT: str = "data"/DATA_ROOT: str = os.getenv("DATA_ROOT", "data")/g' src/config.py
   !python -m src.train --mode vae
   ```
4. Run the classifier training on the extracted latent features:
   ```bash
   !python -m src.train --mode mlp
   ```
5. Once trained, download the `vae_trained.pt` and `MLPClassifier_full.pt` models to use in your web app.

### Downloading Data Locally

If you wish to download the full 15GB dataset to your local machine for testing or local training, you can use the provided download script:

```bash
python download_data.py
```
This script uses `kagglehub` to fetch the dataset and automatically moves it into a local `data/` folder inside the project. Note that this folder is ignored by git to prevent accidental uploads.

## Evaluation

To verify the test accuracy reproduced in the thesis, we provide an evaluation script. This script loads the saved test latent features and tests them against the trained tabular classifier:

```bash
python evaluate.py
```
This should yield an accuracy of around **80.3%**, matching the results stated in our findings.

## Inference and Web App Prototype

We provide a Streamlit frontend (`app.py`) as a basic inference UI prototype. 

**Important Data Limitation:** This model is designed strictly for **word-level lipreading**. During training, we used the dataset's `.align` files to crop out the exact frames where a single word was spoken (usually ~0.5 seconds). 
If you upload a raw, uncropped 3-second `.mpg` video from the GRID dataset into the web app, it will process the entire full sentence and the silence padding at once. Due to the high class imbalance (silence taking up >25% of the frames), the model will overwhelmingly predict `sil` (silence) for uncropped videos.

To test `app.py` correctly, you must upload a pre-cropped video that isolates *only* the specific word being spoken, mirroring the training data alignment.

1. Install Streamlit:
   ```bash
   pip install streamlit
   ```
2. Place your trained `.pt` weight files in the `data/` folder.
3. Run the app:
   ```bash
   streamlit run app.py
   ```

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
