import torch, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from .preprocess import tokenize, encode

def get_attention_for_text(model, text, vocab, max_len=250, device="cpu"):
    model.eval()
    ids = encode(text, vocab, max_len)
    toks = tokenize(text)[:max_len]
    x = torch.tensor([ids]).to(device)
    with torch.no_grad():
        logit, alpha = model(x)
        prob = torch.sigmoid(logit).item()
        alpha = alpha.squeeze(0).cpu().numpy()[:len(toks)]
        if alpha.sum()>0: alpha = alpha/alpha.sum()
    return toks, alpha, prob, 1 if prob>0.5 else 0

def plot_attention(toks, alpha, prob, pred, path="results/attention.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if len(toks)>30: toks=toks[:30]; alpha=alpha[:30]
    plt.figure(figsize=(max(8,len(toks)*0.5),3))
    colors = plt.cm.Reds(alpha/(alpha.max() or 1))
    y=np.arange(len(toks))
    plt.barh(y, alpha, color=colors, edgecolor="black", linewidth=0.5)
    plt.yticks(y, toks); plt.gca().invert_yaxis()
    plt.xlabel("Attention"); plt.title(f"{'POS' if pred else 'NEG'} {prob:.3f}")
    for i,a in enumerate(alpha): plt.text(a+0.002,i,f"{a:.3f}",va="center",fontsize=7)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()

def plot_heatmap(toks, alpha, prob, pred, path="results/heatmap.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if len(toks)>40: toks=toks[:40]; alpha=alpha[:40]
    plt.figure(figsize=(max(10,len(toks)*0.4),2.5))
    sns.heatmap(alpha.reshape(1,-1), cmap="Reds", xticklabels=toks, yticklabels=[f"{'POS' if pred else 'NEG'} {prob:.2f}"])
    plt.xticks(rotation=45,ha="right",fontsize=8); plt.tight_layout()
    plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()

def html_highlight(toks, alpha):
    m = alpha.max() or 1
    out=[]
    for t,a in zip(toks, alpha):
        bg = f"rgba(255,80,80,{0.15+0.85*a/m:.2f})"
        out.append(f'<span style="background:{bg};padding:2px 4px;margin:1px;border-radius:3px">{t}<sub style="font-size:7px"> {a:.2f}</sub></span>')
    return " ".join(out)
