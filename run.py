import os, json, argparse, torch
from datasets import load_dataset
from src.config import config
from src.train import train
from src.evaluate import full_evaluate, plot_confusion, plot_history
from src.visualize import get_attention_for_text, plot_attention, plot_heatmap
from src.dataset import get_loaders
from src.preprocess import load_vocab
from src.model import BiLSTMAttention

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glove", action="store_true")
    p.add_argument("--glove_dim", type=int, default=100, choices=[50,100,200,300])
    p.add_argument("--freeze_glove", action="store_true")
    args = p.parse_args()
    if args.glove:
        config.use_glove = True
        config.glove_dim = args.glove_dim
        config.freeze_glove = args.freeze_glove
        config.embed_dim = args.glove_dim
        print(f"glove {args.glove_dim}d freeze={args.freeze_glove}")
    else:
        print("light mode")

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)

    ds = load_dataset("stanfordnlp/imdb")
    texts_train, labels_train = ds["train"]["text"], ds["train"]["label"]
    texts_test, labels_test = ds["test"]["text"], ds["test"]["label"]
    print(f"train {len(texts_train)} test {len(texts_test)}")

    model, vocab, acc = train(texts_train, labels_train, texts_test, labels_test, config)
    print(f"test acc {acc:.4f}")

    _, _, test_loader = get_loaders(texts_train, labels_train, texts_test, labels_test, vocab, config)
    ckpt = torch.load(config.model_path, map_location=config.device)
    model.load_state_dict(ckpt["model_state"])
    metrics = full_evaluate(model, test_loader, config.device)
    print(metrics["report"])
    print(f"acc {metrics['acc']:.4f} f1 {metrics['f1']:.4f}")

    plot_confusion(metrics["cm"])
    with open("results/history.json") as f: hist=json.load(f)
    plot_history(hist)

    with open("results/metrics_final.json","w") as f:
        json.dump({"accuracy":metrics["acc"],"f1":metrics["f1"],"auc":metrics["auc"]}, f, indent=2)

    for i, ex in enumerate([
        "This movie was absolutely wonderful, brilliant acting and great story!",
        "I hated this film, it was terrible boring and a waste of time.",
        "An absolute masterpiece, touching and inspiring, I loved every minute!",
        "Worst movie ever, awful script and horrible directing.",
        "The movie was not bad, but not great either, confusing plot but decent acting."
    ]):
        toks, alpha, prob, pred = get_attention_for_text(model, ex, vocab, config.max_len, config.device)
        print(f"{i+1} {'POS' if pred else 'NEG'} {prob:.3f} {toks[:4]}")
        plot_attention(toks, alpha, prob, pred, f"results/attention_{i+1}.png")
        plot_heatmap(toks, alpha, prob, pred, f"results/heatmap_{i+1}.png")

    from src.visualize import html_highlight
    html=[]
    for ex in ["This movie was wonderful!", "I hated this boring waste of time."]:
        toks, alpha, prob, pred = get_attention_for_text(model, ex, vocab, config.max_len, config.device)
        html.append(f"<p><b>{'POS' if pred else 'NEG'} {prob:.2f}</b> {ex}<br>{html_highlight(toks, alpha)}</p>")
    with open("results/attention_demo.html","w",encoding="utf-8") as f:
        f.write("<html><body style='font-family:sans-serif'>"+"<hr>".join(html)+"</body></html>")
    print("done — streamlit run app.py")

if __name__ == "__main__": main()
