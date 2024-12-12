import torch
import argparse
from config import *
from model import GPTmodel
from preprocessor import AmharicPreprocessor
from tokenizer import SentencePieceProcessor


class GptInferenceEngine:

    def __init__(self, model: GPTmodel, tokenizer: SentencePieceProcessor, top_k: int= 1, nucleus_threshold=10) -> None:
        self.model = model
        self.top_k = top_k
        self.tokenizer = tokenizer
        self.nucleus_threshold = nucleus_threshold
        self.pad_id = self.tokenizer.pad_id()
        self.eos_id = self.tokenizer.eos_id()
        self.preprocessor = AmharicPreprocessor(tokenizer)

        self.model.eval()

    @torch.no_grad()
    def complete(self, text: str, max_len: int) -> str:
        text = self.preprocessor.execute(text)
        token_ids = self.tokenizer.Encode(text, out_type=int)
        padding = model.config.seq_len - len(token_ids)

        # (1, SEQ_LEN)
        decoder_input = torch.concat([
            torch.tensor(token_ids, dtype=torch.int64),
            torch.tensor([self.pad_id] * padding, dtype=torch.int64)
        ]).unsqueeze(0).to(DEVICE)

        predicted_token = None
        non_pad_tokens = len(token_ids)
        while non_pad_tokens > 0 and non_pad_tokens < max_len and predicted_token != self.eos_id:
            with torch.autocast(device_type=DEVICE.type, enabled=MIXED_PRECISION_ENABLED):
                # (1, SEQ_LEN, VOCAB_SIZE)
                logits = self.model(decoder_input)

            # (1, VOCAB_SIZE)
            next_token_logits = logits[:, non_pad_tokens - 1]

            # Evaluate the probability distribution across the VOCAB_SIZE
            # dimension using softmax - (1, VOCAB_SIZE)
            probab_distribution = torch.softmax(next_token_logits, dim=1)

            # Get the top 5 tokens with the highest probabilities
            _, top_k_tokens = torch.topk(probab_distribution, k=self.top_k, dim=1)

            # Randomly pick from the top 5 most probable tokens
            predicted_token = top_k_tokens[0, torch.randint(0, self.top_k, (1,)).item()]

            # Add the predicted token to the decoder input for the subsequent iterations
            decoder_input[:, non_pad_tokens] = predicted_token.item()

            non_pad_tokens += 1

        # Remove the batch dimension
        # (1, SEQ_LEN) ---> (SEQ_LEN,)
        decoder_input = decoder_input.squeeze(0)

        return self.tokenizer.Decode(decoder_input.detach().cpu().tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument("--preload-weights", type=str, required=True, help="File path to load saved weights")

    args = parser.parse_args()

    if not os.path.exists(args.preload_weights):
        raise FileNotFoundError(f"File {args.preload_weights} does not exist")
    
    LOGGER.info(f"Preloading model weights {args.preload_weights}...")
    checkpoint: dict = torch.load(args.preload_weights, map_location=DEVICE)

    model_config: ModelConfig = checkpoint["model_config"]

    model = GPTmodel.build(model_config, checkpoint["model_state_dict"]).to(DEVICE)

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f"Device: {DEVICE}")
    LOGGER.info(f"Total Parameters: {total_params}")
    LOGGER.info(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    LOGGER.info(f"Model Size(MB): {total_params * 4 / (1024 ** 2):.2f}MB")
    LOGGER.info("Using Mixed Precision (FP16 and FP32) Training" if MIXED_PRECISION_ENABLED else "Using Single Precision (FP32) Training")
    
    tokenizer = SentencePieceProcessor(model_config.seq_len)
    tokenizer.LoadFromFile(
        f"{WORKING_DIR}/tokenizers/amharic-bpe-tokenizer-{model_config.vocab_size // 1000}k.model"
    )
    inference_engine = GptInferenceEngine(model, tokenizer)

    user_input = input("Enter amharic text to complete: ")
    LOGGER.info(
        inference_engine.complete(user_input, model_config.seq_len)
    )