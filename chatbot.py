import time
import torch
import argparse
import traceback
from config import *
from model import GPTmodel
from typing import Iterator
import sentencepiece as spm
from dataset import Conversation
from inference import GptInferenceEngine


class ChatBot(GptInferenceEngine):
    def __init__(self, model: GPTmodel, tokenizer: spm.SentencePieceProcessor, top_k: int = 50, top_p: float = 0.9, temperature: float = 1.0, system_prompt: str = "") -> None:
        super().__init__(model, tokenizer, top_k, top_p, temperature)
        self.user_input = ""
        self.conv = Conversation(system_prompt)
        self.bot_token = self.tokenizer.PieceToId("[BOT]")
        self.user_token = self.tokenizer.PieceToId("[USER]")
        self.system_token = self.tokenizer.PieceToId("[SYSTEM]")
        
        self.system_tokens = []
        if self.conv.system_text:
            self.conv.system_text = self.preprocessor.execute(self.conv.system_text)
            self.system_tokens.extend([
                self.system_token,
                *self.tokenizer.Encode(self.conv.system_text, out_type=int)
            ])
        
    def get_tokens(self, text: str) -> list[int]:
        input_ids: list[int] = []
        if self.conv.system_text:
            input_ids.extend(self.system_tokens)
        
        exchanges = []
        for exchange in self.conv.exchanges:
            exchanges.extend([
                self.user_token,
                *exchange["input"],
                self.bot_token,
                *exchange["output"]
            ])
        
        text = self.preprocessor.execute(text)
        self.user_input = self.tokenizer.Encode(text, out_type=int)
        exchanges.extend([
            self.user_token,
            *self.user_input,
            self.bot_token
        ])
        
        # Discard tokens of the earlier exchanges if input_ids gets too long(exceeds max_len)
        if len(input_ids) + len(exchanges) > self.max_len:
            input_ids.extend(exchanges[len(input_ids) + len(exchanges) - self.max_len:])
        else:
            input_ids.extend(exchanges)
            
        return input_ids
    
    def complete(self, token_ids: list[int]) -> Iterator[int]:
        bot_output = []
        for prediction_token in super().complete(token_ids):
            bot_output.append(prediction_token)
            yield prediction_token
        self.conv.add_exchange(self.user_input, bot_output)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chat with a finetuned GPT model")
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
    
    system_prompt = """አንቺ በአምሃ ክንዱ የተፈጠርሽ ኤአይ (አርቲፊሻል ኢንተለጀንስ) ነሽ፣ ስምሽም ሰላም ይባላል። አንቺ አክብሮት የምታሳይ፣ ሐቀኛ፣ ተግባቢ እና ሙያዊ የኤአይ ረዳት ስለሆንሽ፣ ስለ ኢትዮጵያ—የበለፀገ ታሪኳ፣ የተለያዩ ባህሎቿ፣ ቋንቋዎቿ እና የጊዜው ዘመናዊ እድገቶቿ—ትክክለኛ እና ባህላዊ ግንዛቤ ያለው መረጃ ለመስጠት የተነደፉሽ ነሽ። ለሚቀርብልሽ ጥያቄዎች የምሰጪው መልሶች ምንም ዓይነት ጎጂ፣ ሥነ ምግባር የጎደለው፣ ዘረኛ፣ ወሲብ፣ መርዛማ፣ አደገኛ ወይም ሕገ ወጥ ይዘት ሊኖራቸው አይገባም። በተጨማሪም—የፋይናንስ (ለምሳሌ፣ የኢንቨስትመንት ምክሮች)፣ የህክምና (ለምሳሌ፣ ምርመራዎች)፣ ፖለቲካ፣ ሃይማኖት ወይም የህግ ምክሮች አንዳሰጪ ነገር ግን በትህትና ስለነዚ ርዕሶች ማውራት አልችልም ብለሽ መልሺ።"""
    bot = ChatBot(model, tokenizer, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature, system_prompt=system_prompt)

    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            LOGGER.info("Exiting the program.")
            break
        
        try:
            LOGGER.info(f"Bot: ", extra={"partial": True})
            tokens = bot.get_tokens(user_input)
            for token_id in bot.complete(tokens):
                token: str = tokenizer.IdToPiece(token_id)
                token = token.replace("▁", " ")\
                        .replace(bot.preprocessor.newline_placeholder, "\n")\
                        .replace(bot.preprocessor.tab_placeholder, "\t")
                LOGGER.info(token, extra={"partial": True})
                time.sleep(0.05)
            print()
        except Exception as e:
            traceback.print_exc()
            LOGGER.error(f"Error during inference: {e}")