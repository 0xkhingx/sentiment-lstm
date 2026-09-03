# cfg - keep it simple, human dict
CFG = {
    "max_vocab": 20000,
    "max_len": 250,
    "min_freq": 5,
    "val_split": 0.1,
    # model
    "embed_dim": 128,  # 100 if glove 100d - adjusted in train.py
    "hidden": 128,
    "layers": 2,
    "dropout": 0.5,
    "attn": 128,
    # train
    "batch": 64,
    "epochs": 8,
    "lr": 1e-3,
    "wd": 1e-5,
    "patience": 2,
    # io
    "device": "cpu",
    "seed": 42,
    "model_dir": "models",
    "vocab_path": "models/vocab.json",
    "model_path": "models/best_model.pt",
    # glove - 100d best acc vs speed
    "use_glove": False,
    "glove_dim": 100,
    "glove_path": "embeddings/glove.6B.100d.txt",
    "glove_url": "https://nlp.stanford.edu/data/glove.6B.zip",
    "freeze_glove": False,
    "finetune_after": 2,
}

# compat - old code expects config.xxx
class _C:
    def __init__(self, d): self.__dict__.update(d)
    def __getitem__(self, k): return self.__dict__[k]

config = _C(CFG)
# keep attribute access working for app.py/run.py
for k,v in CFG.items():
    setattr(config, k, v)
# alias for older names
config.embed_dim = CFG["embed_dim"]
config.hidden_dim = CFG["hidden"]
config.num_layers = CFG["layers"]
config.attention_hidden = CFG["attn"]
config.batch_size = CFG["batch"]
config.weight_decay = CFG["wd"]
config.grad_clip = 5.0
