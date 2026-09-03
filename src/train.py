import os, json, random, torch, torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from .config import config
from .preprocess import build_vocab, save_vocab
from .dataset import get_loaders
from .model import BiLSTMAttention
from .embeddings import load_glove, build_embedding_matrix, download_and_extract_glove

def train_one_epoch(model, loader, opt, crit, device):
    model.train()
    tot = 0
    for ids, labels, _ in tqdm(loader, desc="train"):
        ids, labels = ids.to(device), labels.to(device)
        opt.zero_grad()
        logits, _ = model(ids)
        loss = crit(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        tot += loss.item() * ids.size(0)
    return tot / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval()
    tot = 0
    preds, gold = [], []
    for ids, labels, _ in loader:
        ids, labels = ids.to(device), labels.to(device)
        logits, _ = model(ids)
        tot += crit(logits, labels).item() * ids.size(0)
        p = (torch.sigmoid(logits) > 0.5).float().cpu().tolist()
        preds.extend(p); gold.extend(labels.cpu().tolist())
    acc = accuracy_score(gold, preds)
    return tot / len(loader.dataset), acc

def train(texts_train, labels_train, texts_test, labels_test, cfg=config):
    random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    print(f"building vocab from {len(texts_train)}...")
    vocab = build_vocab(texts_train, max_vocab=cfg.max_vocab, min_freq=cfg.min_freq)
    print(f"vocab {len(vocab)}")
    os.makedirs(cfg.model_dir, exist_ok=True)
    save_vocab(vocab, cfg.vocab_path)

    train_loader, val_loader, test_loader = get_loaders(texts_train, labels_train, texts_test, labels_test, vocab, cfg)
    print(f"batches train {len(train_loader)} val {len(val_loader)} test {len(test_loader)}")

    mat = None; freeze = False
    if cfg.use_glove:
        gpath = cfg.glove_path
        if not os.path.exists(gpath):
            try:
                gpath = download_and_extract_glove(cfg.glove_url, os.path.dirname(gpath) or "embeddings", cfg.glove_dim)
            except Exception as e:
                print(f"glove download failed {e}, using learned")
                gpath = None
        if gpath and os.path.exists(gpath):
            if cfg.embed_dim != cfg.glove_dim:
                cfg.embed_dim = cfg.glove_dim
            glove = load_glove(gpath, cfg.glove_dim)
            mat = build_embedding_matrix(vocab, glove, cfg.glove_dim, cfg.seed)
            # freeze first 2 epochs then finetune
            freeze = True if (not cfg.freeze_glove and cfg.finetune_after>0) else cfg.freeze_glove
            print(f"using glove {cfg.glove_dim}d freeze={freeze}")

    model = BiLSTMAttention(len(vocab), cfg.embed_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, cfg.attention_hidden, mat, freeze).to(device)
    print(model)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = 0; no_imp = 0; hist = []
    for epoch in range(1, cfg.epochs+1):
        if cfg.use_glove and not cfg.freeze_glove and cfg.finetune_after>0 and epoch == cfg.finetune_after+1:
            if not model.embedding.weight.requires_grad:
                print(f"unfreezing glove at epoch {epoch}")
                model.embedding.weight.requires_grad = True
                # need to add back to optimizer
                opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        print(f"\nEpoch {epoch}/{cfg.epochs}")
        tr = train_one_epoch(model, train_loader, opt, crit, device)
        vl, va = evaluate(model, val_loader, crit, device)
        print(f"loss train {tr:.4f} val {vl:.4f} acc {va:.4f}")
        hist.append({"epoch":epoch,"train_loss":tr,"val_loss":vl,"val_acc":va})
        if va > best:
            best = va; no_imp = 0
            torch.save({"model_state":model.state_dict(),"vocab":vocab,"config":cfg.__dict__}, cfg.model_path)
            print(f"saved best {va:.4f}")
        else:
            no_imp += 1
            if no_imp >= cfg.patience:
                print("early stop"); break

    ckpt = torch.load(cfg.model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    tl, ta = evaluate(model, test_loader, crit, device)
    print(f"TEST {ta:.4f}")
    with open(os.path.join("results","history.json"),"w") as f:
        json.dump(hist+[{"test_loss":tl,"test_acc":ta}], f, indent=2)
    return model, vocab, ta
