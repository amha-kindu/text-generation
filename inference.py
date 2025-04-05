import time
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

        # (len(token_ids), )
        decoder_input = torch.tensor(token_ids + [self.pad_id] * (max_len - len(token_ids)), dtype=torch.int64).to(DEVICE)
        
        # (SEQ_LEN,) != (1,) --> (SEQ_LEN,) --> (1, SEQ_LEN) --> (1, SEQ_LEN) & (1, SEQ_LEN, SEQ_LEN) --> (1, SEQ_LEN, SEQ_LEN)
        decoder_mask = (decoder_input != self.pad_id).unsqueeze(0).int().to(DEVICE) & TextDataset.lookback_mask(max_len).to(DEVICE)
        
        count = len(token_ids)
        predicted_token = None
        while token_ids and count + 1 < self.tokenizer.max_len and predicted_token != self.eos_id:
            with torch.autocast(device_type=DEVICE.type, enabled=MIXED_PRECISION_ENABLED):
                # (1, SEQ_LEN, VOCAB_SIZE)
                logits = self.model(decoder_input, decoder_mask)

            # (1, VOCAB_SIZE)
            next_token_logits = logits[:, -1]

            # Evaluate the probability distribution across the VOCAB_SIZE
            # dimension using softmax - (1, VOCAB_SIZE)
            probab_distribution = torch.softmax(next_token_logits, dim=1)

            # Randomly pick one of the top 5 tokens
            top_k_probs, top_k_indices = torch.topk(probab_distribution, self.top_k, dim=1)
            chosen_index = torch.multinomial(top_k_probs, 1)
            predicted_token = top_k_indices[0, chosen_index]

            # Add the predicted token to the decoder input for the subsequent iterations
            decoder_input[count] = predicted_token.item()
            
            count += 1
            yield predicted_token.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument("--checkpoint", type=str, required=True, help="File path to load saved weights")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint) and not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"File {args.checkpoint} does not exist")
    
    LOGGER.info(f"Preloading model weights {args.checkpoint}...")
    checkpoint: dict = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)

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
    inference_engine = GptInferenceEngine(model, tokenizer, top_k=5)

    while True:
        user_input = input("Input: ")
        if user_input.lower() == 'exit':
            LOGGER.info("Exiting the program.")
            break

        try:
            LOGGER.info(f"Response: {user_input}", extra={"partial": True})
            for token in inference_engine.complete(user_input, model_config.seq_len):
                LOGGER.info(tokenizer.DecodeIds([token]), extra={"partial": True})
                time.sleep(0.05)
            print()
        except Exception as e:
            traceback.print_exc()
            LOGGER.error(f"Error during inference: {e}")