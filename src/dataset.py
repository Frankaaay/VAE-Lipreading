import os
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((112, 112)),
    T.ToTensor(),
    T.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]),
])

def collect_video_align_pairs(data_root: str = "data", num_subsets: int = 10):
    pairs = []
    for subset in range(1, num_subsets + 1):
        subset_path = f"subset{subset}"
        video_dir = os.path.join(data_root, subset_path, f"s{subset}")
        align_dir = os.path.join(data_root, subset_path, "align")

        if not os.path.isdir(video_dir) or not os.path.isdir(align_dir):
            continue

        video_files = [f for f in os.listdir(video_dir) if f.endswith(".mpg")]
        for vf in video_files:
            stem = os.path.splitext(vf)[0]
            align_path = os.path.join(align_dir, stem + ".align")
            video_path = os.path.join(video_dir, vf)
            if os.path.exists(align_path):
                pairs.append((video_path, align_path))
    print(f"Found {len(pairs)} video-align pairs")
    return pairs

def split_pairs(pairs):
    train_pairs, valtest_pairs = train_test_split(pairs, test_size=0.4, random_state=42)
    val_pairs, test_pairs = train_test_split(valtest_pairs, test_size=0.5, random_state=42)
    return train_pairs, val_pairs, test_pairs

def parse_align_file(align_path: str):
    alignments = []
    with open(align_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, word = parts
                alignments.append((int(start), int(end), word))
    return alignments

def extract_word_frames(video_path: str, alignments):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    word_frames = []
    for start, end, word in alignments:
        start_idx = int(start / 1000)
        end_idx = int(end / 1000)
        seq = frames[start_idx:end_idx]
        if len(seq) > 0:
            word_frames.append((word, seq))
    return word_frames

def preprocess_frames(frames):
    processed = [_transform(frame) for frame in frames]
    video_tensor = torch.stack(processed)
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    return video_tensor.unsqueeze(0)

def load_latent_triplet(prefix: str = "tr", use_mu_only: bool = False):
    logvar = torch.load(f"{prefix}_logvar.pt")
    mu = torch.load(f"{prefix}_mu.pt")
    labels = torch.load(f"{prefix}_labels.pt")

    features = []
    for i in range(len(mu)):
        m = mu[i][0].detach().numpy()
        if use_mu_only:
            features.append(m)
        else:
            l = logvar[i][0].detach().numpy()
            features.append(np.concatenate((l, m), axis=0))
    X = np.array(features, dtype=np.float32)
    return X, labels

from .utils import build_label_index
def load_train_val_test_from_latents(use_mu_only: bool = False):
    X_train, y_train_text = load_latent_triplet("tr", use_mu_only=use_mu_only)
    X_val, y_val_text = load_latent_triplet("val", use_mu_only=use_mu_only)
    X_test, y_test_text = load_latent_triplet("test", use_mu_only=use_mu_only)

    all_labels_text = list(y_train_text) + list(y_val_text) + list(y_test_text)
    unique_labels, label_to_idx, _ = build_label_index(all_labels_text)

    y_train = np.array([label_to_idx[t] for t in y_train_text], dtype=np.int64)
    y_val = np.array([label_to_idx[t] for t in y_val_text], dtype=np.int64)
    y_test = np.array([label_to_idx[t] for t in y_test_text], dtype=np.int64)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), unique_labels, label_to_idx
