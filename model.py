import math
import torch
import torch.nn as nn
from config import D_MODEL, DEVICE, DFF, DROPOUT, HEADS, N_BLOCKS, SEQ_LEN


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, D_MODEL)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, 1) --> (batch, seq_len, d_model)
        return self.embedding.forward(x) * math.sqrt(D_MODEL)
    
    
class PositionalEncoding(nn.Module):
    def __init__(self) -> None:
        super().__init__()        
        self.dropout = nn.Dropout(DROPOUT)
        
        # (seq_len, d_model)
        pe = torch.zeros(SEQ_LEN, D_MODEL)
        
        # (seq_len, 1)
        pos = torch.arange(0, SEQ_LEN, dtype=torch.float).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D_MODEL, 2).float() * -(math.log(10000.0) / D_MODEL))

        # PE(pos, 2i) = sin(pos / (10000 ^ (2i/d_model)))
        pe[:, 0::2] = torch.sin(pos * div_term)

        # PE(pos, 2i + 1) = cos(pos / (10000 ^ (2i/d_model)))
        pe[:, 1::2] = torch.cos(pos * div_term)
        
        # (1, seq_len, d_model)
        pe = pe.unsqueeze(0) 
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        assert x.shape[1] <= SEQ_LEN, f"Input sequence length exceeds the position encoder's max sequence length  `{SEQ_LEN}`"

        # (batch, seq_len, d_model) + (1, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.dropout(x + self.pe[:, :x.shape[1], :].requires_grad_(False))
    
    
class FeedForwardBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(D_MODEL, DFF).to(DEVICE)
        self.dropout = nn.Dropout(DROPOUT)
        self.linear_2 = nn.Linear(DFF, D_MODEL).to(DEVICE)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, d_model) -> (batch, seq_len, dff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self) -> None:
        assert D_MODEL % HEADS == 0, "d_model is not divisible by heads"
        
        super().__init__()
        
        self.d_k = D_MODEL // HEADS
        
        self.W_q = nn.Linear(D_MODEL, D_MODEL, bias=False).to(DEVICE)
        self.W_k = nn.Linear(D_MODEL, D_MODEL, bias=False).to(DEVICE)
        self.W_v = nn.Linear(D_MODEL, D_MODEL, bias=False).to(DEVICE)

        self.W_o = nn.Linear(D_MODEL, D_MODEL, bias=False).to(DEVICE)
        self.dropout = nn.Dropout(DROPOUT)
    
    @staticmethod
    def attention(
        # (batch, head, seq_len, d_k)
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
        dropout: nn.Dropout=None, 
        mask: torch.Tensor=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = query.shape[-1]
        
        # (batch, head, seq_len, d_k) @ (batch, head, d_k, seq_len) --> (batch, head, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)        

        # The mask passed has two components:
        # 1. A lookback mask that makes sure the output at a certain position can only depend on the tokens on from previous positions. (USED ONLY ON THE DECODER)
        # 2. An ignore mask so that attention score for the padding token [PAD] is zero. (USED BOTH ON THE DECODER AND THE ENCODER)
        # If a mask is passed then some of the attention scores are set to zero based on the mask.
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e09)
            
        # (batch, head, seq_len, seq_len) which applies softmax to the last dimension
        # so that the sum of the probabilities along this dimension equals 1
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # (batch, head, seq_len, seq_len) @ (batch, head, seq_len, d_k) --> (batch, head, seq_len, d_k)
        return (attention_scores @ value), attention_scores
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, d_model) @ (d_model, d_model) --> (batch, seq_len, d_model)
        query: torch.Tensor = self.W_q(q) 

        # (batch, seq_len, d_model) @ (d_model, d_model) --> (batch, seq_len, d_model)
        key: torch.Tensor = self.W_k(k)   
        
        # (batch, seq_len, d_model) @ (d_model, d_model) --> (batch, seq_len, d_model)
        value: torch.Tensor = self.W_v(v) 
        
        # (batch, seq_len, d_model) --> (batch, seq_len, head, d_k) --> (batch, head, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], HEADS, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], HEADS, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], HEADS, self.d_k).transpose(1, 2)
        
        # Here has shape x = (batch, head, seq_len, d_k)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, self.dropout, mask)
        
        # (batch, head, seq_len, d_k) --> (batch, seq_len, head, d_k)
        x = x.transpose(1, 2)
        
        # (batch, seq_len, head, d_k) --> (batch, seq_len, d_model)
        x = x.contiguous().view(x.shape[0], -1, HEADS * self.d_k)
        
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.W_o(x)
    
    
class ResidualConnection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(DROPOUT)
        self.norm = nn.LayerNorm(D_MODEL, device=DEVICE)
        
    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))
    
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection() for _ in range(2)])
    
    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, input_mask))
        return self.residual_connections[1](x, self.feed_forward_block)
    
    
class Encoder(nn.Module):
    def __init__(self, encoder_blocks: nn.ModuleList) -> None:
        super().__init__()
        self.encoder_blocks = encoder_blocks
        self.norm = nn.LayerNorm(D_MODEL, device=DEVICE)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for block in self.encoder_blocks:
            x = block(x, mask)
        return self.norm(x)
    
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(DROPOUT)
        self.residual_connections = nn.ModuleList([ResidualConnection() for _ in range(3)])
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, input_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, input_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
       
        
class Decoder(nn.Module):
    def __init__(self, decoder_blocks: nn.ModuleList) -> None:
        super().__init__()
        self.decoder_blocks = decoder_blocks
        self.norm = nn.LayerNorm(D_MODEL, device=DEVICE)
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, input_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder_blocks:
            x = layer(x, encoder_output, input_mask, target_mask)
        return self.norm(x)
    
    
class ProjectionLayer(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(D_MODEL, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    
    
class MtTransformerModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, input_embed: WordEmbedding, target_embed: WordEmbedding, input_pos: PositionalEncoding, target_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_embed = input_embed
        self.target_embed = target_embed
        self.input_pos = input_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer
        
    def encode(self, input: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        input = self.input_embed(input)
        input = self.input_pos(input)
        return self.encoder(input, input_mask)
    
    def decode(self, encoder_output: torch.Tensor, input_mask: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        target = self.target_embed(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, input_mask, target_mask)
    
    def project(self, x: torch.Tensor):
        return self.projection_layer(x)

    @staticmethod
    def build(
        vocab_size: int,
        state: dict = None
    ):
        # Create the embedding layers
        embed = WordEmbedding(vocab_size)
        
        # Create the positional encoding layers
        input_pos = PositionalEncoding()
        target_pos = PositionalEncoding()
        
        # Create N_BLOCKS number of encoders
        encoder_blocks = []
        for _ in range(N_BLOCKS):
            self_attention_block = MultiHeadAttentionBlock()
            feed_forward_block = FeedForwardBlock()
            
            encoder_blocks.append(
                EncoderBlock(self_attention_block, feed_forward_block)
            )
            
        # Create N_BLOCKS number of decoders
        decoder_blocks = []
        for _ in range(N_BLOCKS):
            self_attention_block = MultiHeadAttentionBlock()
            cross_attention_block = MultiHeadAttentionBlock()
            feed_forward_block = FeedForwardBlock()
            
            decoder_blocks.append(
                DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block)
            )
            
        # Create the encoder and the decoder
        encoder = Encoder(nn.ModuleList(encoder_blocks))
        decoder = Decoder(nn.ModuleList(decoder_blocks))
        
        # Create the projection layer
        projection_layer = ProjectionLayer(vocab_size)
        
        # Create the transformer
        transformer = MtTransformerModel(encoder, decoder, embed, embed, input_pos, target_pos, projection_layer)
        
        # Initialize the parameters
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if state:
            transformer.load_state_dict(state)

        return transformer