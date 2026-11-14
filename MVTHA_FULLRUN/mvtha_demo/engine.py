"""Model building, evaluation and checkpoint helpers shared by train_demo.py / demo.py / tests."""
import numpy as np
import torch

from model.mvtha import Model


def pick_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model(cfg):
    return Model(num_class=cfg["num_class"], num_point=25, num_person=cfg["num_person"],
                 in_channels=3, channels=cfg["channels"], num_blocks=cfg["num_blocks"],
                 align_views=cfg["align_views"], dropout=cfg.get("dropout", 0.1))


@torch.no_grad()
def evaluate(model, loader, device, num_views=None):
    """Returns (accuracy, predictions, labels).  num_views=v keeps only the first v views."""
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device)
        if num_views is not None:
            x = x[:, :num_views]
        preds.append(model(x).argmax(1).cpu())
        labels.append(y)
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return float((preds == labels).mean()), preds, labels


def confusion_matrix(preds, labels, n):
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (labels, preds), 1)
    return cm


@torch.no_grad()
def predict_probs(model, x, device):
    """x: numpy/tensor [K,3,T,V,M] or [B,K,3,T,V,M] -> probabilities [B, C] (numpy)."""
    model.eval()
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.dim() == 5:
        x = x.unsqueeze(0)
    return torch.softmax(model(x.to(device)), dim=1).cpu().numpy()


def save_checkpoint(path, model, cfg, extra=None):
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "extra": extra or {}}, path)


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device)
    model = build_model(ck["cfg"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["cfg"]
