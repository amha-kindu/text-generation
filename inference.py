import time
import torch
import argparse
import traceback
from config import *
from model import GPTmodel
from typing import Iterator
import sentencepiece as spm
from dataset import TextDataset
from preprocessor import AmharicPreprocessor


class GptInferenceEngine:
    def __init__(self, model: GPTmodel, tokenizer: spm.SentencePieceProcessor, top_k: int = 50, top_p: float = 0.9, temperature: float = 1.0) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_len = self.model.config.seq_len
        self.preprocessor = AmharicPreprocessor()
        self.stop_token = self.tokenizer.PieceToId("[STOP]")

        self.model.eval()
        
    def get_tokens(self, text: str) -> list[int]:
        text = self.preprocessor.execute(text)
        return self.tokenizer.Encode(text, out_type=int)

    @torch.no_grad()
    def complete(self, token_ids: list[int]) -> Iterator[int]:
        predicted_token = None
        while token_ids and len(token_ids) < self.max_len:
            decoder_input = torch.tensor(
                token_ids,
                dtype=torch.int64
            ).to(DEVICE).unsqueeze(0)
                        
            decoder_mask = TextDataset.lookback_mask(len(token_ids)).to(DEVICE)
                        
            with torch.autocast(device_type=DEVICE.type, enabled=MIXED_PRECISION_ENABLED):
                # (1, SEQ_LEN, VOCAB_SIZE)
                logits = self.model(decoder_input, decoder_mask)

            # (1, VOCAB_SIZE)
            # Take logits for the last position and apply temperature scaling
            next_token_logits = logits[:, -1, :] / self.temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(next_token_logits, dim=-1)

            # Top-k filtering
            if self.top_k > 0:
                top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
                probs = torch.zeros_like(probs).scatter(-1, top_k_indices, top_k_probs)

            # Top-p (nucleus) filtering
            if self.top_p > 0.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs <= self.top_p
                
                mask[..., 0] = True     # Ensure at least one token is selected
                filtered_probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
                probs = torch.zeros_like(probs).scatter(-1, sorted_indices, filtered_probs)

            # Re-normalize probabilities and sample
            probs = probs / probs.sum()
            predicted_token = torch.multinomial(probs, 1).item()
            
            if predicted_token == self.stop_token:
                break
            
            token_ids.append(predicted_token)
            yield predicted_token


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Conduct inference on model")
    parser.add_argument("--top-k", type=int, default=0, help="Top k tokens to sample from (set to 0 to disable)")
    parser.add_argument("--top-p", type=float, default=0.0, help="Top p (nucleus) sampling probability (set to 0.0 to disable)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (t=1.0 for normal sampling, 0<t<1.0 for less random, t>1.0 for more random sampling)")
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
    inference_engine = GptInferenceEngine(model, tokenizer, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature)

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
