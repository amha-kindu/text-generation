import torch, math
import torch.nn as nn
from config import *


class Embedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)

    # Input shape: x -> (N_BATCHES, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, D_MODEL)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embedding(x) * math.sqrt(D_MODEL))


class PositionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(DROPOUT)
        
        # (SEQ_LEN, 1)
        pos = torch.arange(0, SEQ_LEN, dtype=torch.float).unsqueeze(1)

        # (D_MODEL//2,)
        div_term = torch.exp(torch.arange(0, D_MODEL, 2, dtype=torch.float) * -math.log(10000.0) / D_MODEL)

        # (SEQ_LEN, D_MODEL)
        self.pe: torch.Tensor = torch.zeros(SEQ_LEN, D_MODEL)

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
    def __init__(self):
        assert D_MODEL % HEADS == 0, "D_MODEL is not divisible by heads"

        super().__init__()
        self.d_head: int = D_MODEL // HEADS

        self.Wq: nn.Linear = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk: nn.Linear = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv: nn.Linear = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wo: nn.Linear = nn.Linear(D_MODEL, D_MODEL, bias=False)

        self.dropout = nn.Dropout(DROPOUT)
    
    # Input shape: x -> (N_BATCHES, SEQ_LEN, D_MODEL), mask -> (1, SEQ_LEN, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, D_MODEL)
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # (N_BATCHES, SEQ_LEN, D_MODEL) @ (D_MODEL, D_MODEL) --> (N_BATCHES, SEQ_LEN, D_MODEL)
        query: torch.Tensor = self.Wq(x)
        key: torch.Tensor = self.Wk(x)
        value: torch.Tensor = self.Wv(x)

        # (N_BATCHES, SEQ_LEN, D_MODEL) --> (N_BATCHES, SEQ_LEN, HEADS, d_head) --> (N_BATCHES, HEADS, SEQ_LEN, d_head)
        query = query.view(x.shape[0], x.shape[1], HEADS, -1).transpose(1, 2)
        key = key.view(x.shape[0], x.shape[1], HEADS, -1).transpose(1, 2)
        value = value.view(x.shape[0], x.shape[1], HEADS, -1).transpose(1, 2)

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
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(D_MODEL, DFF)
        self.linear2 = nn.Linear(DFF, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, D_MODEL)
    # Output shape: (N_BATCHES, SEQ_LEN, D_MODEL)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(
            self.dropout(
                torch.relu(self.linear1(x))
            )
        )
        

class ResidualConnection(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, D_MODEL)
    # Output shape: (N_BATCHES, SEQ_LEN, D_MODEL)
    def forward(self, x: torch.Tensor, sublayer: nn.Module):
        return x + self.dropout(sublayer(self.layer_norm(x)))
        

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_head = MultiHeadAttentionBlock()
        self.feed_forward = FeedForwardBlock()
        self.residual_connections = nn.ModuleList([ResidualConnection() for _ in range(2)])

    # Input shape: x -> (N_BATCHES, SEQ_LEN, D_MODEL), mask -> (1, SEQ_LEN, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, D_MODEL)
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = self.residual_connections[0](x, lambda x: self.attention_head(x, mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x


class Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(D_MODEL, VOCAB_SIZE)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, D_MODEL)
    # Output shape: (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class GPTmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoders = nn.ModuleList([DecoderBlock() for _ in range(N_BLOCKS)])
        self.embedding = Embedding()
        self.position_encoder = PositionEncoder()
        self.projection = Projection()
        self.layer_norm = nn.LayerNorm(D_MODEL)

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
        state: dict = {}
    ):
        model = GPTmodel()

        if state:
            model.load_state_dict(state)
        else:
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        return model