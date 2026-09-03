import argparse
from src.embeddings import download_and_extract_glove
from src.config import config

p = argparse.ArgumentParser()
p.add_argument("--dim", type=int, default=100, choices=[50,100,200,300])
args = p.parse_args()
path = download_and_extract_glove(config.glove_url, "embeddings", args.dim)
print(path)
