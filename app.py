import streamlit as st, torch, json, os
from src.preprocess import load_vocab
from src.model import BiLSTMAttention
from src.config import config
from src.visualize import get_attention_for_text, html_highlight
import matplotlib.pyplot as plt, seaborn as sns, numpy as np

st.set_page_config(page_title="Sentiment LSTM", layout="wide")
st.title("Sentiment — BiLSTM + Attention")

# quick emb tag
try:
    ckpt = torch.load(config.model_path, map_location="cpu")
    c = ckpt.get("config",{})
    tag = f"GloVe {c.get('glove_dim')}d" if c.get("use_glove") else "learned 128d"
except: tag = "learned 128d"
st.caption(f"{tag} | 2-layer BiLSTM 128 | attention shows word influence")

@st.cache_resource
def load_model():
    if not os.path.exists(config.vocab_path): return None, None
    vocab = load_vocab(config.vocab_path)
    ckpt = torch.load(config.model_path, map_location="cpu")
    cfg = ckpt.get("config",{})
    m = BiLSTMAttention(len(vocab), cfg.get("embed_dim",128), cfg.get("hidden_dim",128), cfg.get("num_layers",2), cfg.get("dropout",0.5), cfg.get("attention_hidden",128))
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m, vocab

model, vocab = load_model()
if model is None:
    st.warning("no model yet — run `python run.py` (light) or `python run.py --glove`")
    st.stop()

if os.path.exists("results/metrics_final.json"):
    with open("results/metrics_final.json") as f: m=json.load(f)
    st.sidebar.metric("Accuracy", f"{m.get('accuracy',0):.1%}")

examples = [
    "This movie was absolutely wonderful, brilliant acting and great story!",
    "I hated this film, it was terrible boring and a waste of time.",
    "An absolute masterpiece, touching and inspiring, I loved every minute!",
    "Worst movie ever, awful script and horrible directing.",
]
ex = st.sidebar.selectbox("try one", [""]+examples)
text = st.text_area("review", value=ex if ex else examples[0], height=110)

if st.button("Analyze", type="primary") or text:
    if not text.strip(): st.error("enter text")
    else:
        toks, alpha, prob, pred = get_attention_for_text(model, text, vocab, config.max_len, "cpu")
        label = "POSITIVE" if pred else "NEGATIVE"
        c1,c2 = st.columns([1,2])
        with c1:
            st.metric(label, f"{prob:.3f}")
            st.progress(float(prob) if pred else 1-float(prob))
            fig, ax = plt.subplots(figsize=(6, max(3,len(toks)*0.3)))
            colors = plt.cm.Reds(alpha/(alpha.max() or 1))
            y=np.arange(len(toks))
            ax.barh(y, alpha, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_yticks(y); ax.set_yticklabels(toks); ax.invert_yaxis()
            ax.set_xlabel("attention"); st.pyplot(fig)
        with c2:
            st.write(html_highlight(toks, alpha), unsafe_allow_html=True)
            if len(toks)<=40:
                fig2, ax2 = plt.subplots(figsize=(max(8,len(toks)*0.4),1.8))
                sns.heatmap(alpha.reshape(1,-1), cmap="Reds", xticklabels=toks, yticklabels=[label], ax=ax2, cbar=True)
                plt.xticks(rotation=35,ha="right",fontsize=8); st.pyplot(fig2)
            st.caption("darker red = more influence")

st.divider()
st.write(f"BiLSTM 2x128 + attention | {tag} | IMDB 50k")
