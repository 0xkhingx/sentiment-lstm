import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden, attn=128):
        super().__init__()
        self.W = nn.Linear(hidden, attn)
        self.v = nn.Linear(attn, 1, bias=False)

    def forward(self, h, mask=None):
        e = torch.tanh(self.W(h))
        scores = self.v(e).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        alpha = F.softmax(scores, dim=-1)
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        return context, alpha

# keep old name working
class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.5, attn_hidden=128, pad_idx=0, embedding_matrix=None, freeze_embedding=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            if freeze_embedding:
                self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.attention = Attention(hidden_dim*2, attn_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim*2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        mask = (x != 0).float()
        emb = self.dropout(self.embedding(x))
        out, _ = self.lstm(emb)
        ctx, alpha = self.attention(out, mask)
        h = self.dropout(ctx)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        logit = self.fc2(h).squeeze(-1)
        return logit, alpha
