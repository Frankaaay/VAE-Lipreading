import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from src.models import MLPClassifier

def evaluate_downloaded_model():
    print("Loading test dataset...")
    try:
        mu = torch.load('data/test_mu.pt', weights_only=True)
        logvar = torch.load('data/test_logvar.pt', weights_only=True)
        test_labels = torch.load('data/test_labels.pt', weights_only=True)
        train_labels = torch.load('data/tr_labels.pt', weights_only=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Build unique label index from train_labels
    all_labels_text = list(train_labels) + list(test_labels)
    unique_labels = sorted(set(all_labels_text))
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    
    # Prepare test features
    features = []
    for i in range(len(mu)):
        m = mu[i][0].detach().numpy()
        l = logvar[i][0].detach().numpy()
        features.append(np.concatenate((l, m), axis=0))
    X_test = np.array(features, dtype=np.float32)
    y_test = np.array([label_to_idx[t] for t in test_labels], dtype=np.int64)
    
    X_tensor = torch.from_numpy(X_test)
    
    print("Loading MLP classifier...")
    try:
        mlp = torch.load('data/MLPClassifier_full.pt', map_location='cpu', weights_only=False)
        mlp.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Evaluating...")
    with torch.no_grad():
        logits = mlp(X_tensor)
        preds = logits.argmax(dim=1).cpu().numpy()
        
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc*100:.2f}%")
    
if __name__ == "__main__":
    evaluate_downloaded_model()
