import os
import torch
import random
import numpy as np
from dataclasses import dataclass

@dataclass
class Config:
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset properties
    DATA_ROOT: str = "data"
    NUM_SUBSETS: int = 10

    # VAE params
    LATENT_DIM: int = 128
    NUM_CLASSES: int = 53
    VAE_EPOCHS: int = 10
    VAE_LR: float = 1e-4
    KL_WEIGHT: float = 0.1

    # Latent MLP params
    USE_MU_ONLY: bool = False
    MLP_EPOCHS: int = 250
    MLP_LR: float = 1e-3
    MLP_BATCH_SIZE: int = 64

    # DNN/XGB params
    LATENT_AUG_SAMPLES: int = 5

    # Vector feature mode files
    TRAIN_FEATURES_NPY: str = "train_feats_full.npy"
    TEST_FEATURES_NPY: str = "test_feats_full.npy"

    # Label files
    TR_LABELS: str = "tr_labels.pt"
    VAL_LABELS: str = "val_labels.pt"
    TEST_LABELS: str = "test_labels.pt"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

cfg = Config()
