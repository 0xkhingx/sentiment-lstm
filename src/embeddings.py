import os, numpy as np, torch

def load_glove(path, dim=100):
    vecs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip().split(" ")
            if len(p) != dim+1: 
                continue
            vecs[p[0]] = np.array(p[1:], dtype=np.float32)
    print(f"loaded {len(vecs)} glove vectors from {path}")
    return vecs

def build_embedding_matrix(vocab, glove, dim=100, seed=42):
    np.random.seed(seed)
    size = len(vocab)
    mat = np.zeros((size, dim), dtype=np.float32)
    mat[1] = np.random.uniform(-0.25, 0.25, dim)  # UNK
    inv = {i:w for w,i in vocab.items()}
    found = 0
    for idx in range(2, size):
        w = inv[idx]
        if w in glove:
            mat[idx] = glove[w]
            found += 1
        else:
            mat[idx] = np.random.uniform(-0.25, 0.25, dim)
    print(f"glove coverage {found}/{size-2} ({found/max(1,size-2)*100:.1f}%)")
    return torch.from_numpy(mat)

# compat aliases
load_glove_vectors = load_glove
# keep old name

def download_and_extract_glove(url, dest="embeddings", dim=100):
    import zipfile, urllib.request
    os.makedirs(dest, exist_ok=True)
    txt = f"glove.6B.{dim}d.txt"
    out = os.path.join(dest, txt)
    if os.path.exists(out):
        return out
    zip_path = os.path.join(dest, "glove.6B.zip")
    if not os.path.exists(zip_path):
        print(f"downloading glove 6B (~862MB)...")
        print(f"if this fails, manually download {url} -> {zip_path}")
        urllib.request.urlretrieve(url, zip_path)
    print(f"extracting {txt}...")
    with zipfile.ZipFile(zip_path) as z:
        z.extract(txt, dest)
    return out
