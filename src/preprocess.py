import re, json, collections

PAD = "<PAD>"; UNK = "<UNK>"
PAD_ID = 0; UNK_ID = 1

def clean_text(t):
    t = re.sub(r"<br\s*/?>", " ", t)
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = t.lower()
    t = re.sub(r"[^a-z0-9!?']+", " ", t)  # keep ! ? for sentiment
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize(t): return clean_text(t).split()

def build_vocab(texts, max_vocab=20000, min_freq=5):
    counter = collections.Counter()
    for t in texts: counter.update(tokenize(t))
    filtered = [(w,c) for w,c in counter.items() if c >= min_freq]
    filtered.sort(key=lambda x: x[1], reverse=True)
    vocab = {PAD: PAD_ID, UNK: UNK_ID}
    for w,_ in filtered[:max_vocab-2]:
        vocab[w] = len(vocab)
    return vocab

def encode(text, vocab, max_len=250):
    toks = tokenize(text)
    ids = [vocab.get(t, UNK_ID) for t in toks[:max_len]]
    if len(ids) < max_len: ids += [PAD_ID]*(max_len-len(ids))
    return ids

def save_vocab(vocab, path):
    with open(path, "w", encoding="utf-8") as f: json.dump(vocab, f)

def load_vocab(path):
    with open(path, encoding="utf-8") as f: return json.load(f)

def decode(ids, vocab):
    inv = {v:k for k,v in vocab.items()}
    return [inv.get(i, UNK) for i in ids if i != PAD_ID]
