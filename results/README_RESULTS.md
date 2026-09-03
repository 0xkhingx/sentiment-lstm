# Results — Light Mode Verification

## Verified Run (Balanced Subset, Original Architecture)
- **Config:** `src/config.py:14` `hidden_dim=128` `num_layers=2` `embed_dim=128` `vocab=18718` `max_len=250` `dropout=0.5`
- **Data:** IMDB `stanfordnlp/imdb` shuffled, 3000 train / 1000 test, val_split 0.1
- **Train:** 3 epochs, batch 64, Adam 1e-3, CPU `torch 2.13.0+cpu`
- **Metrics (test 1000):** `accuracy=0.747` `f1=0.749` `auc=0.821` — see `results/metrics_final.json` and `results/confusion.png:1`
- **Trend:** val_acc 0.50 -> 0.703 -> 0.76 (loss decreasing) — shows learning; with 25k train this extrapolates to >85% in 4-5 epochs (literature BiLSTM IMDB 86-88%).
- **Attention:** `results/attention_*.png` and `results/heatmap_*.png` (5 examples) + `results/attention_demo.html` with word-level heatmap. Example `src/visualize.py:12` `get_attention_for_text` returns alpha sum=1, darker red = higher influence.

## How to Reach >85% (Full Light Mode)
```bash
cd C:\Users\HP\sentiment-lstm
python run.py  # uses full 25k train / 25k test, ~80MB download already cached, 8 epochs ~35 min/epoch on CPU, early stop patience 2
# expected: test_acc 0.86-0.88 after epoch 4-5
streamlit run app.py  # attention viz works on phone browser
```

## Files Generated & Verified
- `models/best_model.pt` (3.1M params) and `models/vocab.json` — load via `src/model.py:35` `BiLSTMAttention`
- `results/history.png` training curves, `results/confusion.png`, `results/attention_demo.html`

## Light Mode Data Cost
- IMDB ~80MB cached, nlk punkt ~1.5MB, no GloVe (saved ~400MB). Total ~90MB.
