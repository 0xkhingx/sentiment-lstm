import torch, os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, classification_report
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

@torch.no_grad()
def full_evaluate(model, loader, device="cpu"):
    model.eval()
    probs, preds, gold = [], [], []
    for ids, labels, _ in loader:
        ids = ids.to(device)
        p = torch.sigmoid(model(ids)[0]).cpu().tolist()
        pr = [1 if x>0.5 else 0 for x in p]
        probs.extend(p); preds.extend(pr); gold.extend(labels.tolist())
    acc = accuracy_score(gold, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(gold, preds, average="binary", zero_division=0)
    try: auc = roc_auc_score(gold, probs)
    except: auc = float("nan")
    cm = confusion_matrix(gold, preds)
    rep = classification_report(gold, preds, digits=4)
    return {"acc":acc,"prec":prec,"rec":rec,"f1":f1,"auc":auc,"cm":cm,"report":rep}

def plot_confusion(cm, path="results/confusion.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["neg","pos"], yticklabels=["neg","pos"])
    plt.xlabel("Pred"); plt.ylabel("True"); plt.title("Confusion")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_history(hist, path="results/history.png"):
    if not hist or "epoch" not in hist[0]: return
    epochs=[h["epoch"] for h in hist]
    tr=[h["train_loss"] for h in hist]; vl=[h["val_loss"] for h in hist]; va=[h["val_acc"] for h in hist]
    fig, ax1 = plt.subplots(figsize=(7,4))
    ax1.plot(epochs, tr, label="train", marker="o"); ax1.plot(epochs, vl, label="val", marker="o")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(loc="upper left")
    ax2=ax1.twinx(); ax2.plot(epochs, va, color="green", marker="s", linestyle="--", label="acc")
    ax2.set_ylabel("Acc"); ax2.legend(loc="upper right")
    plt.title("History"); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
