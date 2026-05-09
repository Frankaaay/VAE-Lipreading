import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import argparse

from .config import cfg, set_seed
from .utils import build_label_index, make_loader, evaluate_loader
from .dataset import (collect_video_align_pairs, split_pairs, parse_align_file,
                      extract_word_frames, preprocess_frames, load_train_val_test_from_latents)
from .models import VideoVAE, MLPClassifier, SimpleDNN

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def train_vae_and_export():
    print("Running VAE training and latency export...")
    pairs = collect_video_align_pairs(cfg.DATA_ROOT, cfg.NUM_SUBSETS)
    if len(pairs) == 0:
        raise ValueError("No video-align pairs found. Check DATA_ROOT.")

    train_pairs, val_pairs, test_pairs = split_pairs(pairs)

    all_words = []
    for _, align_path in train_pairs:
        for _, _, word in parse_align_file(align_path):
            all_words.append(word)
    label_encoder = {w: i for i, w in enumerate(sorted(set(all_words)))}

    vae = VideoVAE(latent_dim=cfg.LATENT_DIM, num_classes=cfg.NUM_CLASSES).to(cfg.DEVICE)
    optimizer = optim.Adam(vae.parameters(), lr=cfg.VAE_LR)
    recon_loss_fn = nn.MSELoss()
    cls_loss_fn = nn.CrossEntropyLoss()

    vae.train()
    for epoch in range(cfg.VAE_EPOCHS):
        running_loss = 0.0
        count = 0
        for video_path, align_path in tqdm(train_pairs, desc=f"VAE epoch {epoch+1}/{cfg.VAE_EPOCHS}"):
            alignments = parse_align_file(align_path)
            word_frames = extract_word_frames(video_path, alignments)

            for word, frames in word_frames:
                if word not in label_encoder:
                    continue

                x = preprocess_frames(frames).to(cfg.DEVICE)
                y = torch.tensor([label_encoder[word]], device=cfg.DEVICE)

                optimizer.zero_grad()
                recon, mu, logvar, logits = vae(x)
                target = vae.encoder(x).flatten(1).detach()
                
                recon_loss = recon_loss_fn(recon, target)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
                cls_loss = cls_loss_fn(logits, y)

                loss = recon_loss + cfg.KL_WEIGHT * kl_loss + cls_loss
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
                count += 1

        avg_loss = running_loss / max(count, 1)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

    torch.save(vae.state_dict(), "vae_trained.pt")
    print("Saved vae_trained.pt")

    def export_latents(split_pairs, prefix):
        vae.eval()
        all_mu, all_logvar, all_labels = [], [], []
        with torch.no_grad():
            for video_path, align_path in tqdm(split_pairs, desc=f"Export {prefix}"):
                alignments = parse_align_file(align_path)
                word_frames = extract_word_frames(video_path, alignments)
                for word, frames in word_frames:
                    x = preprocess_frames(frames).to(cfg.DEVICE)
                    mu, logvar = vae.encode(x)
                    all_mu.append(mu.cpu())
                    all_logvar.append(logvar.cpu())
                    all_labels.append(word)

        torch.save(all_mu, f"{prefix}_mu.pt")
        torch.save(all_logvar, f"{prefix}_logvar.pt")
        torch.save(all_labels, f"{prefix}_labels.pt")
        print(f"Saved {prefix}_mu.pt, {prefix}_logvar.pt, {prefix}_labels.pt")

    export_latents(train_pairs, "tr")
    export_latents(val_pairs, "val")
    export_latents(test_pairs, "test")

def train_mlp_core(train_dl, valid_dl, d_in, d_out):
    model = MLPClassifier(d_in=d_in, d_out=d_out).to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.MLP_LR, weight_decay=1e-4)

    for epoch in range(1, cfg.MLP_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(cfg.DEVICE), yb.to(cfg.DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * xb.size(0)

        if epoch % 10 == 0:
            val_acc, _, _ = evaluate_loader(model, valid_dl, device=cfg.DEVICE)
            print(f"Epoch {epoch:03d} | train loss={epoch_loss/len(train_dl.dataset):.4f} | val acc={val_acc*100:.2f}%")

    return model

def run_mlp_from_latent():
    (X_train, y_train), (X_val, y_val), (X_test, y_test), unique_labels, label_to_idx = load_train_val_test_from_latents(
        use_mu_only=cfg.USE_MU_ONLY
    )

    train_dl = make_loader(X_train, y_train, batch_size=cfg.MLP_BATCH_SIZE, shuffle=True)
    val_dl = make_loader(X_val, y_val, batch_size=cfg.MLP_BATCH_SIZE, shuffle=False)
    test_dl = make_loader(X_test, y_test, batch_size=cfg.MLP_BATCH_SIZE, shuffle=False)

    model = train_mlp_core(train_dl, val_dl, d_in=X_train.shape[1], d_out=len(unique_labels))

    test_acc, y_true, y_pred = evaluate_loader(model, test_dl, device=cfg.DEVICE)
    print(f"TEST accuracy: {test_acc*100:.2f}%")
    print(classification_report(y_true, y_pred, digits=4))

    out_name = "MLPClassifier_mu_only.pt" if cfg.USE_MU_ONLY else "MLPClassifier_full.pt"
    torch.save(model, out_name)
    print(f"Saved {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lipreading Training Pipeline")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["vae", "mlp"],
                        help="Which stage of the pipeline to run.")
    args = parser.parse_args()
    
    set_seed(cfg.SEED)
    
    if args.mode == "vae":
        train_vae_and_export()
    elif args.mode == "mlp":
        run_mlp_from_latent()