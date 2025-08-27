import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAdapter(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        assert rank > 0, "rank must be > 0"
        
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.rank = rank
        self.merged = False
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / max(1, rank)

        self.down = nn.Parameter(torch.empty(base.in_features, rank, dtype=self.base.weight.dtype))
        self.up = nn.Parameter(torch.empty(rank, base.out_features, dtype=self.base.weight.dtype))

        # LoRA paper: down ~ N(0, 1/r), up = 0
        nn.init.normal_(self.down, mean=0.0, std=1 / max(1, rank))
        nn.init.zeros_(self.up)

    @torch.no_grad()
    def merge(self):
        if self.merged:
            return
        
        # (OUT_DIM, RANK) @ (RANK, IN_DIM) --> (OUT_DIM, IN_DIM)
        delta_w = F.linear(self.up.t(), self.down, bias=None)
        self.base.weight.add_(delta_w.to(self.base.weight.dtype), alpha=self.scaling)
        self.merged = True
    
    @torch.no_grad()
    def unmerge(self):
        if not self.merged:
            return
        
        # (OUT_DIM, RANK) @ (RANK, IN_DIM) --> (OUT_DIM, IN_DIM)
        delta_w = F.linear(self.up.t(), self.down, bias=None)
        self.base.weight.sub_(delta_w.to(self.base.weight.dtype), alpha=self.scaling)
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base(x)

        # (N_BATCHES, SEQ_LEN, IN_DIM) @ (IN_DIM, RANK) --> (N_BATCHES, SEQ_LEN, RANK)
        r = F.linear(self.dropout(x), self.down.t(), bias=None)
        
        # (N_BATCHES, SEQ_LEN, RANK) @ (RANK, OUT_DIM) --> (N_BATCHES, SEQ_LEN, OUT_DIM)
        r = F.linear(r, self.up.t(), bias=None)
        
        return self.base(x) + self.scaling * r
