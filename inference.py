import time
import torch
import argparse
import traceback
from config import *
from model import GPTmodel
from typing import Iterator
from dataset import TextDataset
from preprocessor import AmharicPreprocessor
from tokenizer import SentencePieceProcessor


class GptInferenceEngine:
    def __init__(self, model: GPTmodel, tokenizer: SentencePieceProcessor, top_k: int = 50, top_p: float = 0.9, temperature: float = 1.0) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.pad_id = self.tokenizer.pad_id()
        self.eos_id = self.tokenizer.eos_id()
        self.preprocessor = AmharicPreprocessor()

        self.model.eval()

    @torch.no_grad()
    def complete(self, text: str) -> Iterator[int]:
        text = self.preprocessor.execute(text)
        token_ids: list[int] = self.tokenizer.Encode(text, out_type=int)

        predicted_token = None
        while token_ids and len(token_ids) < self.tokenizer.max_len and predicted_token != self.eos_id:
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
            prob_dist = torch.softmax(next_token_logits, dim=-1)

            # Filter the top k tokens based on their probabilities
            top_k = min(self.top_k, prob_dist.size(-1))
            top_k_probs, top_k_indices = torch.topk(prob_dist, top_k, dim=-1)

            # Sort top_k by descending prob, do top-p within that subset
            sorted_probs, sorted_indices = torch.sort(top_k_probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = (cumulative_probs <= self.top_p)

            # Ensure at least one token is kept: only set the *highest-prob* token true
            mask[0, 0] = True

            filtered_probs = sorted_probs[mask]
            filtered_indices = sorted_indices[mask]

            # If top-p filtering zeroed everything (very rare), revert to top_k
            if filtered_probs.numel() == 0:
                filtered_probs = top_k_probs[0]
                filtered_indices = top_k_indices[0]
            else:
                # Otherwise, map sorted_indices back to the original top_k_indices:
                # sorted_indices are indices *within* top_k => we find actual token IDs:
                filtered_indices = top_k_indices[0, filtered_indices]

            # Re-normalize probabilities and sample
            filtered_probs = filtered_probs / filtered_probs.sum()
            chosen_idx = torch.multinomial(filtered_probs, 1).item()
            predicted_token = filtered_indices[chosen_idx].item()

            token_ids.append(predicted_token)

            yield predicted_token


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument("--checkpoint", type=str, required=True, help="File path to load saved weights")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint) and not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"File {args.checkpoint} does not exist")
    
    LOGGER.info(f"Preloading model weights {args.checkpoint}...")
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
    
    tokenizer = SentencePieceProcessor(model_config.seq_len)
    tokenizer.LoadFromFile(
        f"{WORKING_DIR}/tokenizers/amharic-bpe-tokenizer-{model_config.vocab_size // 1000}k.model"
    )
    inference_engine = GptInferenceEngine(model, tokenizer, top_k=10, top_p=0.9, temperature=1)

    while True:
        user_input = input("Input: ")
        if user_input.lower() == 'exit':
            LOGGER.info("Exiting the program.")
            break

        try:
            LOGGER.info(f"Response: {user_input}", extra={"partial": True})
            for token_id in inference_engine.complete(user_input):
                if token_id != tokenizer.eos_id():
                    token: str = tokenizer.IdToPiece(token_id)
                    LOGGER.info(token.replace("▁", " "), extra={"partial": True})
                    time.sleep(0.05)
            print()
        except Exception as e:
            traceback.print_exc()
            LOGGER.error(f"Error during inference: {e}")
