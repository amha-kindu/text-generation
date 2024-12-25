import math
import torch
from config import *
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    # Input shape: x -> (N_BATCHES, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.embed_dim)


class PositionEncoder(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # (SEQ_LEN, 1)
        positions = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # (EMBED_DIM//2,)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * -math.log(10000.0) / embed_dim)

        # (SEQ_LEN, EMBED_DIM)
        position_encodings: torch.Tensor = torch.zeros(seq_len, embed_dim)

        # PE(positions, 2i) = sin(positions / (10000 ^ (2i/embed_dim)))
        # PE(positions, 2i) = sin(positions * exp(-2i * log(10000) / embed_dim))
        position_encodings[:, ::2] = torch.sin(positions * div_term)

        # PE(positions, 2i+1) = cos(positions / (10000 ^ (2i/embed_dim)))
        # PE(positions, 2i+1) = cos(positions * exp(-2i * log(10000) / embed_dim))
        position_encodings[:, 1::2] = torch.cos(positions * div_term)

        # (SEQ_LEN, EMBED_DIM) --> (1, SEQ_LEN, EMBED_DIM)
        position_encodings = position_encodings.unsqueeze(0)

        self.register_buffer("position_encodings", position_encodings)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.position_encodings.requires_grad_(False))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, dropout: float, heads: int):
        assert embed_dim % heads == 0, "EMBED_DIM is not divisible by heads"

        super().__init__()
        self.heads = heads
        self.d_head: int = embed_dim // heads

        self.Wq: nn.Linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk: nn.Linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv: nn.Linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wo: nn.Linear = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
    
    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM), mask -> (1, SEQ_LEN, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def forward(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        # (N_BATCHES, SEQ_LEN, EMBED_DIM) @ (EMBED_DIM, EMBED_DIM) --> (N_BATCHES, SEQ_LEN, EMBED_DIM)
        key: torch.Tensor = self.Wk(key)
        query: torch.Tensor = self.Wq(query)
        value: torch.Tensor = self.Wv(value)

        # (N_BATCHES, SEQ_LEN, EMBED_DIM) --> (N_BATCHES, SEQ_LEN, HEADS, d_head) --> (N_BATCHES, HEADS, SEQ_LEN, d_head)
        query = query.view(query.shape[0], query.shape[1], self.heads, -1).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, -1).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, -1).transpose(1, 2)

        # (N_BATCHES, HEADS, SEQ_LEN, d_head) @ (N_BATCHES, HEADS, d_head, SEQ_LEN)
        # (N_BATCHES, HEADS, SEQ_LEN, SEQ_LEN)
        attention_scores = (query @ key.transpose(2, 3)) / math.sqrt(self.d_head)
        
        if mask is not None:
            # Disable mixed-precision if enabled
            with torch.autocast(DEVICE.type, enabled=False):
                if MIXED_PRECISION_ENABLED:
                    attention_scores = attention_scores.to(torch.float32)
                attention_scores.masked_fill_(mask == False, -1e09)

                # (N_BATCHES, HEADS, SEQ_LEN, SEQ_LEN)
                attention_scores = torch.softmax(
                    attention_scores, dim=-1
                )

            if MIXED_PRECISION_ENABLED:
                attention_scores = attention_scores.to(torch.float16)

        attention_scores = self.dropout(attention_scores)

        # (N_BATCHES, head, SEQ_LEN, SEQ_LEN) @ (N_BATCHES, head, SEQ_LEN, d_head)
        # (N_BATCHES, head, SEQ_LEN, d_head)
        output: torch.Tensor = attention_scores @ value

        # (N_BATCHES, head, SEQ_LEN, d_head) -> (N_BATCHES, SEQ_LEN, head, d_head)
        output = output.transpose(1, 2)

        # (N_BATCHES, SEQ_LEN, head, d_head) -> (N_BATCHES, SEQ_LEN, EMBED_DIM)
        output = output.contiguous().view(value.shape[0], value.shape[2], -1)
        
        return self.Wo(output)
    

class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(
            self.dropout(
                torch.relu(self.linear1(x))
            )
        )


class AddAndNorm(nn.Module):
    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM), y -> (N_BATCHES, SEQ_LEN, EMBED_DIM)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x + self.dropout(self.layer_norm(y))

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float, heads: int):
        super().__init__()
        self.feed_forward = FeedForwardBlock(embed_dim, ff_dim, dropout)
        self.masked_multihead_attention = MultiHeadAttentionBlock(embed_dim, dropout, heads)
        self.add_and_norm = nn.ModuleList([AddAndNorm(embed_dim, dropout) for _ in range(2)])

    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM), mask -> (1, SEQ_LEN, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = self.add_and_norm[0](x, self.masked_multihead_attention(x, x, x, mask))
        x = self.add_and_norm[1](x, self.feed_forward(x))
        return x


class Projection(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, vocab_size)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM)
    # Output shape: (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class GPTmodel(nn.Module):
    def __init__(self, n_blocks: int, embed_dim: int, vocab_size: int, ff_dim: int, dropout: float, heads: int, seq_len: int):
        super().__init__()
        self.config = ModelConfig(
            vocab_size=vocab_size,
            n_blocks=n_blocks,
            embed_dim=embed_dim,
            dropout=dropout,
            seq_len=seq_len,
            heads=heads,
            ff_dim=ff_dim,
        )

        self.decoders = nn.ModuleList([DecoderBlock(embed_dim, ff_dim, dropout, heads) for _ in range(n_blocks)])
        self.embedding = Embedding(embed_dim, vocab_size)
        self.position_encoder = PositionEncoder(seq_len, embed_dim, dropout)
        self.projection = Projection(embed_dim, vocab_size)
        self.layer_norm = nn.LayerNorm(embed_dim)

    # Input shape: x -> (N_BATCHES, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def _embed_and_encode_position(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        return self.position_encoder(x)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM)
    # Output shape: (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
    def _project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)
    
    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM), mask -> (1, SEQ_LEN, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def _decode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for decoder in self.decoders:
            x = decoder(x, mask)
        return self.layer_norm(x)

    # Input shape: x -> (N_BATCHES, SEQ_LEN), mask -> (1, SEQ_LEN, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self._embed_and_encode_position(x)
        x = self._decode(x, mask)
        return self._project(x)
    
    @staticmethod
    def build(
        params: ModelConfig,
        state: dict = {}
    ):
        model = GPTmodel(
            n_blocks=params.n_blocks,
            embed_dim=params.embed_dim,
            vocab_size=params.vocab_size,
            ff_dim=params.ff_dim,
            dropout=params.dropout,
            heads=params.heads,
            seq_len=params.seq_len
        )

        if state:
            model.load_state_dict(state)
        else:
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        return model