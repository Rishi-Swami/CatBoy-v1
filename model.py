# model.py
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.embed_size = embed_size
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=False)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        hidden_dim = forward_expansion * embed_size
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, embed) -> transform to (seq_len, batch, embed)
        x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.shape

        # causal mask: prevent attending to future tokens
        # attn_mask expects shape (seq_len, seq_len)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)

        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_output)
        forward = self.feed_forward(x)
        x = self.norm2(x + forward)
        x = self.dropout(x)
        return x.transpose(0, 1)  # back to (batch, seq_len, embed)

class GPTModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, num_layers=6, heads=8, dropout=0.1, forward_expansion=4, max_position_embeddings=512):
        super(GPTModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_position_embeddings, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.embed_size = embed_size
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x):
        # x: (batch_size, seq_len)
        batch_size, seq_length = x.shape
        if seq_length > self.max_position_embeddings:
            raise ValueError(f"Sequence length {seq_length} > max_position_embeddings {self.max_position_embeddings}")

        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand(batch_size, seq_length)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.fc_out(x)  # (batch, seq_len, vocab)
        return logits
