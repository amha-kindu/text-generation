import ijson
import torch
from config import *
from typing import Iterator
from preprocessor import AmharicPreprocessor
from tokenizer import SentencePieceProcessor
from torch.utils.data import Dataset, DataLoader, Sampler


class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: SentencePieceProcessor) -> None:
        super().__init__()
        self.file = open(file_path, 'r', encoding='utf-8')
        self.tokenizer = tokenizer

        self.pad_token = torch.tensor([self.tokenizer.pad_id()], dtype=torch.int64)
        self.eos_token = torch.tensor([self.tokenizer.eos_id()], dtype=torch.int64)
        self.bos_token = torch.tensor([self.tokenizer.bos_id()], dtype=torch.int64)

        self.sentences = []
        preprocessor = AmharicPreprocessor()
        file_name = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            LOGGER.info(f"\033[93mLoading data from {file_name}...\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
            for sentence in ijson.items(f, "item"):
                preprocessed_sentence = preprocessor.execute(sentence)
                if preprocessed_sentence:
                    self.sentences.append(preprocessed_sentence)
                    if self.sentences and len(self.sentences) % 100000 == 0:
                        LOGGER.info(f"\033[93mLoaded {len(self.sentences)} sentences from {file_name}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
        LOGGER.info(f"\033[92mDone! Loaded {len(self.sentences)} sentences from {file_name}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None

    def __len__(self) -> int:
        return len(self.sentences)

    def batch_iterator(self, batch_size: int, sampler: Sampler=None) -> DataLoader:
        return DataLoader(self, batch_size, shuffle=(sampler is None), sampler=sampler)

    @staticmethod
    def lookback_mask(size: int) -> torch.Tensor:
        # Lower triangular matrix
        # [[
        #   [1, 0, ... , 0],
        #   [1, 1, ... , 0],
        #   [1, 1, ... , 0],
        #   [1, 1, ... , 1]
        # ]]
        # 1 x size x size
        return torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int) == 0

    def __getitem__(self, index) -> Iterator[dict]:
        text = self.sentences[index]
        
        # Preprocess and tokenize
        token_ids = self.tokenizer.Encode(text, out_type=int)
        padding = self.tokenizer.max_len - len(token_ids) - 1

        # (SEQ_LEN,)
        decoder_input = torch.concat([
            # (len(token_ids),)
            torch.tensor(token_ids, dtype=torch.int64),

            # (1,)
            torch.tensor([self.eos_token.item()], dtype=torch.int64),

            # (padding,)
            torch.tensor([self.pad_token.item()] * padding, dtype=torch.int64)
        ])[:self.tokenizer.max_len]

        # (SEQ_LEN,)
        label = torch.concat([
            # (len(token_ids) - 1,)
            torch.tensor(token_ids[1:], dtype=torch.int64),

            # (1,)
            torch.tensor([self.eos_token.item()], dtype=torch.int64),

            # (padding,)
            torch.tensor([self.pad_token.item()] * (padding + 1), dtype=torch.int64)
        ])[:decoder_input.size(0)]

        return {
            # (SEQ_LEN,)
            "decoder_input": decoder_input,

            # (SEQ_LEN,) != (1,) --> (SEQ_LEN,) --> (1, SEQ_LEN) --> (1, SEQ_LEN) & (1, SEQ_LEN, SEQ_LEN) --> (1, SEQ_LEN, SEQ_LEN)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & self.lookback_mask(self.tokenizer.max_len),

            # (SEQ_LEN,)
            "label": label
        }