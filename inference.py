import time
import torch
import argparse
import traceback
from config import *
from model import GPTmodel
from typing import Iterator
import sentencepiece as spm
from collections import Counter
import torch.nn.functional as F
from dataset import TextDataset
from cache import SlidingKVCache
from preprocessor import AmharicPreprocessor


class GptInferenceEngine:
    def __init__(self, model: GPTmodel, tokenizer: spm.SentencePieceProcessor, config: InferenceConfig):
        
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessor = AmharicPreprocessor()
        self.max_len = self.model.config.seq_len
        self.use_kv_cache = config.kv_cache_size > 0
        self.kv_caches = [
            SlidingKVCache(config.kv_cache_size) for _ in range(self.model.config.n_blocks) \
            if self.use_kv_cache
        ]
        
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.temperature = min(config.temperature, config.max_temp) + 1e-5

        self.pad_token = self.tokenizer.PieceToId("[PAD]")
        self.unk_token = self.tokenizer.PieceToId("[UNK]")
        self.bot_token = self.tokenizer.PieceToId("[BOT]")
        self.stop_token = self.tokenizer.PieceToId("[STOP]")
        self.user_token = self.tokenizer.PieceToId("[USER]")
        self.system_token = self.tokenizer.PieceToId("[SYSTEM]")

        self.rep_window = config.rep_window
        self.freq_penalty = config.freq_penalty
        self.presence_penalty = config.presence_penalty
        self.repetition_penalty = config.repetition_penalty
        self.no_repeat_ngram_size = config.no_repeat_ngram_size

        self.model.eval()
        
    def get_tokens(self, text: str) -> list[int]:
        text = self.preprocessor.execute(text)
        return self.tokenizer.Encode(text, out_type=int)

    def _ban_tokens(self, logits: torch.Tensor, banned_token_ids: list[int]) -> None:
        for token_id in banned_token_ids:
            if token_id >= 0:
                logits[..., token_id] = -float("inf")
                
    def _no_repeat_ngrams_ids(self, history: list[int], n: int) -> list[int]:
        if n <= 1 or len(history) < n-1:
            return []
        prefix = tuple(history[-(n-1):])      # last n-1 tokens
        bans = []
        for i in range(len(history) - n + 1):
            if tuple(history[i:i+n-1]) == prefix:
                bans.append(history[i+n-1])   # the token that completed that n-gram before
        return list(set(bans))

    def _apply_penalties(self, logits: torch.Tensor, history: list[int]) -> None:
        if not history: 
            return
        counts = Counter(history[-self.rep_window:])
        fp = self.freq_penalty
        pp = self.presence_penalty
        rp = self.repetition_penalty
        if rp <= 1.0 and fp <= 0.0 and pp <= 0.0:
            return

        for token_id, count in counts.items():
            if token_id < 0: 
                continue
            
            val = logits[..., token_id]
            # HF-style repetition penalty
            logits[..., token_id] = torch.where(val > 0, val / rp, val * rp)
            
            # frequency + presence penalties (OpenAI-style)
            logits[..., token_id] = logits[..., token_id] - (fp * float(count) + pp)

    @torch.no_grad()
    def complete(self, token_ids: list[int]) -> Iterator[int]:
        while token_ids and len(token_ids) < self.max_len:
            decoder_input = torch.tensor(
                token_ids,
                dtype=torch.int64
            ).to(DEVICE).unsqueeze(0)
                        
            decoder_mask = TextDataset.lookback_mask(len(token_ids)).to(DEVICE)
                        
            with torch.autocast(device_type=DEVICE.type, enabled=MIXED_PRECISION_ENABLED):
                # (1, SEQ_LEN, VOCAB_SIZE)
                logits = self.model(decoder_input, decoder_mask, self.use_kv_cache, self.kv_caches)

            # (1, VOCAB_SIZE)
            # Take logits for the last position and apply temperature scaling
            next_token_logits = logits[:, -1, :] / self.temperature

            # Ban control tokens and no-repeat n-gram causing tokens
            self._ban_tokens(
                logits=next_token_logits, 
                banned_token_ids=[
                    self.unk_token, self.user_token, self.bot_token, self.system_token, self.pad_token,
                    *self._no_repeat_ngrams_ids(token_ids, self.no_repeat_ngram_size)
                ]
            )

            # Apply presence, frequency and repetition penalties based on history
            self._apply_penalties(next_token_logits, token_ids)

            # (1, VOCAB_SIZE)
            probs = F.softmax(next_token_logits, dim=-1)

            # Top-k filtering (on probs, keeping your style)
            if self.top_k and self.top_k > 0:
                top_k_probs, top_k_idx = torch.topk(probs, k=min(self.top_k, probs.size(-1)), dim=-1)
                probs = torch.zeros_like(probs).scatter(-1, top_k_idx, top_k_probs)

            # Top-p (nucleus) filtering
            if self.top_p and self.top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cprobs = torch.cumsum(sorted_probs, dim=-1)
                mask = cprobs <= self.top_p
                mask[..., 0] = True  # ensure at least one token
                filtered = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
                probs = torch.zeros_like(probs).scatter(-1, sorted_idx, filtered)

            # Re-normalize; if everything got masked, fall back to argmax
            denom = probs.sum(dim=-1, keepdim=True)
            if torch.all(denom <= 0):
                print("WARNING: all probabilities are zero, falling back to argmax")
                predicted_token = torch.argmax(next_token_logits, dim=-1).item()
            else:
                probs = probs / denom.clamp_min(1e-12)
                predicted_token = torch.multinomial(probs, num_samples=1).item()
            
            if predicted_token == self.stop_token:
                break
            
            token_ids.append(predicted_token)
            yield predicted_token


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Conduct inference on model")
    parser.add_argument("--top-k", type=int, default=DEFAULT_INFERENCE_CONFIG.top_k, help="Top k tokens to sample from (set to 0 to disable)")
    parser.add_argument("--top-p", type=float, default=DEFAULT_INFERENCE_CONFIG.top_p, help="Top p (nucleus) sampling probability (set to 0.0 to disable)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_INFERENCE_CONFIG.temperature, help="Sampling temperature (t=1.0 for normal sampling, 0<t<1.0 for less random, t>1.0 for more random sampling)")
    parser.add_argument("--repetition-penalty", type=float, default=DEFAULT_INFERENCE_CONFIG.repetition_penalty, help="Repetition penalty strength")
    parser.add_argument("--presence-penalty", type=float, default=DEFAULT_INFERENCE_CONFIG.presence_penalty, help="Presence penalty strength")
    parser.add_argument("--frequency-penalty", type=float, default=DEFAULT_INFERENCE_CONFIG.freq_penalty, help="Frequency penalty strength")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=DEFAULT_INFERENCE_CONFIG.no_repeat_ngram_size, help="No repeat n-gram size")
    parser.add_argument("--repeat-window", type=int, default=DEFAULT_INFERENCE_CONFIG.rep_window, help="Repeat window size")
    parser.add_argument("--kv-cache-size", type=int, default=DEFAULT_INFERENCE_CONFIG.kv_cache_size, help="KV cache size")
    parser.add_argument("--checkpoint", type=str, required=True, help="File path to load saved checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True, help="File path to load SentencePiece tokenizer")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint) and not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"File {args.checkpoint} does not exist")
    
    if not os.path.exists(args.tokenizer) and not os.path.isfile(args.tokenizer):
        raise FileNotFoundError(f"File {args.tokenizer} does not exist")
    
    LOGGER.info(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint: dict = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)

    model_config: ModelConfig = checkpoint["model_config"]
    inference_config: InferenceConfig = InferenceConfig(**args.__dict__)

    model = GPTmodel.build(model_config, checkpoint["weights"]).to(DEVICE)

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f"Device: {DEVICE}")
    LOGGER.info(f"Total Parameters: {total_params}")
    LOGGER.info(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    LOGGER.info(f"Model Size(MB): {total_params * 4 / (1024 ** 2):.2f}MB")
    LOGGER.info(f"Initiating inference with {'mixed-precision' if MIXED_PRECISION_ENABLED else 'single-precision'}...")
    
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.LoadFromFile(args.tokenizer)
    inference_engine = GptInferenceEngine(model, tokenizer, inference_config)

    while True:
        user_input = input("Input: ")
        if user_input.lower() == 'exit':
            LOGGER.info("Exiting the program.")
            break

        try:
            LOGGER.info(f"Response: {user_input}", extra={"partial": True})
            tokens = inference_engine.get_tokens(user_input)
            for token_id in inference_engine.complete(tokens):
                token: str = tokenizer.IdToPiece(token_id)
                LOGGER.info(token.replace("▁", " "), extra={"partial": True})
                time.sleep(0.05)
            print()
        except Exception as e:
            traceback.print_exc()
            LOGGER.error(f"Error during inference: {e}")
