# sentiment lstm

bi-lstm + attention for imdb. learned or glove, attention shows which words matter.

```
pip install -r requirements.txt
python run.py              # light, ~90MB, 25k train -> 86-88% in 4-5 epochs
python run.py --glove --glove_dim 100  # needs embeddings/glove.6B.100d.txt (~862MB zip from https://nlp.stanford.edu/data/glove.6B.zip)
streamlit run app.py
```

- vocab 20k, max_len 250, 2-layer bilstm 128, dropout 0.5
- glove 100d best tradeoff, otherwise learned 128d. coverage ~70% with glove
- attention is bahdanau, masked softmax, red = more influence

results in `results/` — confusion, history, attention pngs.

light vs glove: light is fine for 85%+, glove gives +2-3% and nicer attention.

