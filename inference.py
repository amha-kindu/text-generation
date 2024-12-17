import torch
import argparse
import traceback
from config import *
from dataset import TextDataset
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
        self.preprocessor = AmharicPreprocessor()

        self.model.eval()

    @torch.no_grad()
    def complete(self, text: str, max_len: int) -> str:
        text = self.preprocessor.execute(text)
        token_ids = self.tokenizer.Encode(text, out_type=int)

        # (1, len(token_ids))
        decoder_input = torch.tensor([token_ids], dtype=torch.int64, device=DEVICE)

        predicted_token = None
        while token_ids and decoder_input.size(1) < max_len and predicted_token != self.eos_id:
            # (1, len(token_ids), len(token_ids))
            decoder_mask = TextDataset.lookback_mask(decoder_input.size(1)).to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, enabled=MIXED_PRECISION_ENABLED):
                # (1, SEQ_LEN, VOCAB_SIZE)
                logits = self.model(decoder_input, decoder_mask)

            # (1, VOCAB_SIZE)
            next_token_logits = logits[:, -1]

            # Evaluate the probability distribution across the VOCAB_SIZE
            # dimension using softmax - (1, VOCAB_SIZE)
            probab_distribution = torch.softmax(next_token_logits, dim=1)

            # Get the token with the highest probabilities
            _, predicted_token = torch.max(probab_distribution, dim=1)

            # Add the predicted token to the decoder input for the subsequent iterations
            decoder_input = torch.concat([
                decoder_input,
                predicted_token.unsqueeze(1)
            ], dim=1)

        # Remove the batch dimension
        # (1, SEQ_LEN) ---> (SEQ_LEN,)
        decoder_input = decoder_input.squeeze(0)

        return self.tokenizer.Decode(decoder_input.detach().cpu().tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument("--load-checkpoint", type=str, required=True, help="File path to load saved weights")

    args = parser.parse_args()

    if not os.path.exists(args.load_checkpoint) and not os.path.isfile(args.load_checkpoint):
        raise FileNotFoundError(f"File {args.load_checkpoint} does not exist")
    
    LOGGER.info(f"Preloading model weights {args.load_checkpoint}...")
    checkpoint: dict = torch.load(args.load_checkpoint, map_location=DEVICE)

    model_config: ModelConfig = checkpoint["model_config"]

    model = GPTmodel.build(model_config, checkpoint["model_state_dict"]).to(DEVICE)

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
    inference_engine = GptInferenceEngine(model, tokenizer)

    while True:
        user_input = input("Enter Amharic text to complete (or type 'exit' to stop): ")

        if user_input.lower() == 'exit':
            LOGGER.info("Exiting the program.")
            break

        try:
            completed_text = inference_engine.complete(user_input, model_config.seq_len)
            LOGGER.info(completed_text)
        except Exception as e:
            traceback.print_exc()
            LOGGER.error(f"Error during inference: {e}")