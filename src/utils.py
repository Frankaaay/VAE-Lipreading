import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

def build_label_index(labels: List[str]) -> Tuple[List[str], Dict[str, int], np.ndarray]:
    unique_labels = sorted(set(labels))
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    labels_idx = np.array([label_to_idx[l] for l in labels], dtype=np.int64)
    return unique_labels, label_to_idx, labels_idx

def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = False) -> DataLoader:
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def evaluate_loader(model: nn.Module, dl: DataLoader, device: str = "cpu"):
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(xb.to(device)) for xb, _ in dl], dim=0)
        targets = torch.cat([yb for _, yb in dl], dim=0).to(device)
        preds = logits.argmax(dim=1)
    acc = accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())
    return acc, targets.cpu().numpy(), preds.cpu().numpy()
