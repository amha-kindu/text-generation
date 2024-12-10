import math
import torch
from config import *
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    # Input shape: x -> (N_BATCHES, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, D_MODEL)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionEncoder(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # (SEQ_LEN, 1)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # (D_MODEL//2,)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -math.log(10000.0) / d_model)

        # (SEQ_LEN, D_MODEL)
        self.pe: torch.Tensor = torch.zeros(seq_len, d_model)

        # PE(pos, 2i) = sin(pos / (10000 ^ (2i/dmodel)))
        # PE(pos, 2i) = sin(pos * exp(-2i * log(10000) / dmodel))
        self.pe[:, ::2] = torch.sin(pos * div_term)

        # PE(pos, 2i+1) = cos(pos / (10000 ^ (2i/dmodel)))
        # PE(pos, 2i+1) = cos(pos * exp(-2i * log(10000) / dmodel))
        self.pe[:, 1::2] = torch.cos(pos * div_term)

        # (SEQ_LEN, D_MODEL) --> (1, SEQ_LEN, D_MODEL)
        self.pe = self.pe.unsqueeze(0)

        self.register_buffer("pe_no_grad", self.pe)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, D_MODEL)
    # Output shape: (N_BATCHES, SEQ_LEN, D_MODEL)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe_no_grad.requires_grad_(False))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float, heads: int):
        assert d_model % heads == 0, "D_MODEL is not divisible by heads"

        super().__init__()
        self.heads = heads
        self.d_head: int = d_model // heads

        self.Wq: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.Wk: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.Wv: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.Wo: nn.Linear = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
    
    # Input shape: x -> (N_BATCHES, SEQ_LEN, D_MODEL), mask -> (1, SEQ_LEN, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, D_MODEL)
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # (N_BATCHES, SEQ_LEN, D_MODEL) @ (D_MODEL, D_MODEL) --> (N_BATCHES, SEQ_LEN, D_MODEL)
        query: torch.Tensor = self.Wq(x)
        key: torch.Tensor = self.Wk(x)
        value: torch.Tensor = self.Wv(x)

        # (N_BATCHES, SEQ_LEN, D_MODEL) --> (N_BATCHES, SEQ_LEN, HEADS, d_head) --> (N_BATCHES, HEADS, SEQ_LEN, d_head)
        query = query.view(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)
        key = key.view(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)
        value = value.view(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)

        # (N_BATCHES, HEADS, SEQ_LEN, d_head) @ (N_BATCHES, HEADS, d_head, SEQ_LEN)
        # (N_BATCHES, HEADS, SEQ_LEN, SEQ_LEN)
        attention_scores = (query @ key.transpose(2, 3)) / math.sqrt(self.d_head)

        if mask is not None:
            attention_scores.masked_fill_(mask == False, -1e09)

        # (N_BATCHES, HEADS, SEQ_LEN, SEQ_LEN)
        attention_scores = torch.softmax(
            attention_scores, dim=-1
        )

        attention_scores = self.dropout(attention_scores)

        # (N_BATCHES, head, SEQ_LEN, SEQ_LEN) @ (N_BATCHES, head, SEQ_LEN, d_head)
        # (N_BATCHES, head, SEQ_LEN, d_head)
        output: torch.Tensor = attention_scores @ value

        # (N_BATCHES, head, SEQ_LEN, d_head) -> (N_BATCHES, SEQ_LEN, head, d_head)
        output = output.transpose(1, 2)

        # (N_BATCHES, SEQ_LEN, head, d_head) -> (N_BATCHES, SEQ_LEN, D_MODEL)
        output = output.contiguous().view(x.shape[0], x.shape[1], -1)
        
        return self.Wo(output)
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, dff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)
        self.dropout = nn.Dropout(dropout)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, D_MODEL)
    # Output shape: (N_BATCHES, SEQ_LEN, D_MODEL)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(
            self.dropout(
                torch.relu(self.linear1(x))
            )
        )
        

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, D_MODEL)
    # Output shape: (N_BATCHES, SEQ_LEN, D_MODEL)
    def forward(self, x: torch.Tensor, sublayer: nn.Module):
        return x + self.dropout(sublayer(self.layer_norm(x)))
        

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, dff: int, dropout: float, heads: int):
        super().__init__()
        self.attention_head = MultiHeadAttentionBlock(d_model, dropout, heads)
        self.feed_forward = FeedForwardBlock(d_model, dff, dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    # Input shape: x -> (N_BATCHES, SEQ_LEN, D_MODEL), mask -> (1, SEQ_LEN, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, D_MODEL)
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = self.residual_connections[0](x, lambda x: self.attention_head(x, mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x


class Projection(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, D_MODEL)
    # Output shape: (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class GPTmodel(nn.Module):
    def __init__(self, n_blocks: int, d_model: int, vocab_size: int, dff: int, dropout: float, heads: int, seq_len: int):
        super().__init__()
        self.config = ModelConfig(
            vocab_size=vocab_size,
            n_blocks=n_blocks,
            d_model=d_model,
            dropout=dropout,
            seq_len=seq_len,
            heads=heads,
            dff=dff,
        )

        self.decoders = nn.ModuleList([DecoderBlock(d_model, dff, dropout, heads) for _ in range(n_blocks)])
        self.embedding = Embedding(d_model, vocab_size)
        self.position_encoder = PositionEncoder(seq_len, d_model, dropout)
        self.projection = Projection(d_model, vocab_size)
        self.layer_norm = nn.LayerNorm(d_model)

    # Input shape: x -> (N_BATCHES, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, D_MODEL)
    def _embed_and_encode_position(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        return self.position_encoder(x)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, D_MODEL)
    # Output shape: (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
    def _project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)
    
    # Input shape: x -> (N_BATCHES, SEQ_LEN, D_MODEL), mask -> (1, SEQ_LEN, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, D_MODEL)
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
            d_model=params.d_model,
            vocab_size=params.vocab_size,
            dff=params.dff,
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