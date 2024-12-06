from typing import Iterator
import torch
import json
from config import *
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from preprocessor import AmharicPreprocessor


class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: spm.SentencePieceProcessor) -> None:
        super().__init__()
        self.file = open(file_path, 'r', encoding='utf-8')
        self.tokenizer = tokenizer
        self.preprocessor = AmharicPreprocessor(tokenizer)

        self.pad_token = torch.tensor([self.tokenizer.pad_id()], dtype=torch.int64)
        self.eos_token = torch.tensor([self.tokenizer.eos_id()], dtype=torch.int64)

        with open(file_path, 'r', encoding='utf-8') as f:
            self.sentences = json.loads(f.read())

    def __len__(self) -> int:
        return len(self.sentences)

    def batch_iterator(self, batch_size: int, sampler: DistributedSampler) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, sampler=sampler)

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
        token_ids = self.preprocessor.preprocess(text)
        token_ids = token_ids[:SEQ_LEN]
        padding = SEQ_LEN - len(token_ids)

        # (SEQ_LEN,)
        decoder_input = torch.concat([
            # (len(token_ids),)
            torch.tensor(token_ids, dtype=torch.int64),

            # (padding,)
            torch.tensor([self.pad_token] * padding, dtype=torch.int64)
        ])

        # (SEQ_LEN,)
        label = torch.concat([
            # (len(token_ids) - 1,)
            torch.tensor(token_ids[1:], dtype=torch.int64),

            # (1, )
            torch.tensor([self.eos_token], dtype=torch.int64),

            # (padding,)
            torch.tensor([self.pad_token] * padding, dtype=torch.int64)
        ])[:SEQ_LEN]

        return {
            # (SEQ_LEN,)
            "decoder_input": decoder_input,

            # (SEQ_LEN,) != (1,) --> (SEQ_LEN,) --> (1, SEQ_LEN) --> (1, SEQ_LEN) & (1, SEQ_LEN, SEQ_LEN) --> (1, SEQ_LEN, SEQ_LEN)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & self.lookback_mask(SEQ_LEN),

            # (SEQ_LEN,)
            "label": label
        }