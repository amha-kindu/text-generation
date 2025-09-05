import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAdapter(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        assert rank > 0, "rank must be > 0"
        
        # Snapshot base weights as immutable buffers (not Parameters)
        self.register_buffer("base_weight", base.weight.detach().clone())
        if base.bias is not None:
            self.register_buffer("base_bias", base.bias.detach().clone())
        else:
            self.base_bias = None

        self.rank = rank
        self.merged = False
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / max(1, rank)

        self.down = nn.Parameter(torch.empty(base.in_features, rank, dtype=base.weight.dtype))
        self.up = nn.Parameter(torch.empty(rank, base.out_features, dtype=base.weight.dtype))

        # LoRA paper: down ~ N(0, 1/r), up = 0
        nn.init.normal_(self.down, mean=0.0, std=1 / max(1, rank))
        nn.init.zeros_(self.up)
    
    def base(self, x: torch.Tensor) -> nn.Linear:
        return F.linear(x, self.base_weight, self.base_bias)

    @torch.no_grad()
    def merge(self):
        if self.merged:
            return
        
        # (OUT_DIM, RANK) @ (RANK, IN_DIM) --> (OUT_DIM, IN_DIM)
        delta_w = F.linear(self.up.t(), self.down, bias=None)
        self.base_weight.add_(
            delta_w.to(self.base_weight.dtype), alpha=self.scaling
        )
        self.merged = True
    
    @torch.no_grad()
    def unmerge(self):
        if not self.merged:
            return
        
        # (OUT_DIM, RANK) @ (RANK, IN_DIM) --> (OUT_DIM, IN_DIM)
        delta_w = F.linear(self.up.t(), self.down, bias=None)        
        self.base_weight.sub_(
            delta_w.to(self.base_weight.dtype), alpha=self.scaling
        )
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base(x)

        # (N_BATCHES, SEQ_LEN, IN_DIM) @ (IN_DIM, RANK) --> (N_BATCHES, SEQ_LEN, RANK)
        r = F.linear(self.dropout(x), self.down.t(), bias=None)
        
        # (N_BATCHES, SEQ_LEN, RANK) @ (RANK, OUT_DIM) --> (N_BATCHES, SEQ_LEN, OUT_DIM)
        r = F.linear(r, self.up.t(), bias=None)
        
        return self.base(x) + self.scaling * r
    
    @staticmethod
    def get_lora_param_names( targets: dict):
        new_params = []        
        for submodule_name, data in targets.items():
            if data["type"] == 'ModuleList':
                for idx  in data['indices']:
                    for target in data['submodules']:
                        new_params.append(f"{submodule_name}.{idx}.{target}.up")
                        new_params.append(f"{submodule_name}.{idx}.{target}.down")
            elif data["type"] == 'Module':
                for target in data['submodules']:
                    new_params.append(f"{submodule_name}.{target}.up")
                    new_params.append(f"{submodule_name}.{target}.down")
            else:
                raise ValueError(f"Unknown type: {data['type']}")
        return new_params

    @staticmethod
    def apply_lora(model: nn.Module, targets: dict, rank: int, alpha: float, dropout: float):
        params = set(tuple(p.split(".")[:-1]) for p in LoRAdapter.get_lora_param_names(targets))
        for param in params:
            layer_name, parent_module_name = param[-1], ".".join(param[:-1])
            
            module = model.get_submodule(parent_module_name)
            linear_layer = getattr(module, layer_name)
            
            if not isinstance(linear_layer, nn.Linear):
                raise ValueError(f"{layer_name} must be nn.Linear")
            
            adapter = LoRAdapter(
                linear_layer,
                rank=rank,
                alpha=alpha,
                dropout=dropout
            )
            setattr(module, layer_name, adapter)
