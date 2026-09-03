import torch
from torch.utils.data import Dataset, DataLoader, random_split
from .preprocess import encode

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=250):
        self.texts = texts; self.labels = labels; self.vocab = vocab; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        ids = encode(self.texts[idx], self.vocab, self.max_len)
        return torch.tensor(ids), torch.tensor(float(self.labels[idx])), self.texts[idx]

def collate_fn(batch):
    ids, labels, texts = zip(*batch)
    return torch.stack(ids), torch.stack(labels), list(texts)

def get_loaders(texts_train, labels_train, texts_test, labels_test, vocab, cfg):
    train_ds = IMDBDataset(texts_train, labels_train, vocab, cfg.max_len)
    test_ds = IMDBDataset(texts_test, labels_test, vocab, cfg.max_len)
    val_size = int(len(train_ds) * cfg.val_split)
    train_size = len(train_ds) - val_size
    gen = torch.Generator().manual_seed(cfg.seed)
    a,b = random_split(train_ds, [train_size, val_size], generator=gen)
    return DataLoader(a, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn), \
           DataLoader(b, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn), \
           DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
