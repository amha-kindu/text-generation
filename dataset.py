from os import SEEK_SET
from typing import Iterator
import torch
from config import *
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from preprocessor import AmharicPreprocessor
from torch.utils.data.sampler import RandomSampler


class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: spm.SentencePieceProcessor) -> None:
        super().__init__()
        self.file = open(file_path, 'r', encoding='utf-8')
        self.tokenizer = tokenizer
        self.preprocessor = AmharicPreprocessor(tokenizer)

        self.pad_token = torch.tensor([self.tokenizer.pad_id()], dtype=torch.int64)
        self.sos_token = torch.tensor([self.tokenizer.bos_id()], dtype=torch.int64)
        self.eos_token = torch.tensor([self.tokenizer.eos_id()], dtype=torch.int64)
        
        offset = 0
        self.offsets = []
        for line in self.file:
            self.offsets.append(offset)
            offset += len(line.encode('utf-8')) + 1

        self.file.seek(0, SEEK_SET)

    def __del__(self):
        self.file.close()

    def __len__(self) -> int:
        return len(self.offsets)

    def batch_iterator(self, batch_size: int) -> DataLoader:
        return DataLoader(self, batch_size, shuffle=True)

    def random_samples(self, batch_size: int, count: int) -> DataLoader:
        return DataLoader(self, batch_size, sampler=RandomSampler(self, replacement=True, num_samples=count))

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
        self.file.seek(self.offsets[index], SEEK_SET)
        text = self.file.readline()
        
        # Preprocess and tokenize
        token_ids = self.preprocessor.preprocess(text)
        padding = SEQ_LEN - len(token_ids) - 2

        # (SEQ_LEN,)
        decoder_input = torch.concat([
            # (1,)
            self.sos_token,

            # (len(token_ids),)
            torch.tensor(token_ids, dtype=torch.int64),

            # (1, )
            self.eos_token,

            # (padding,)
            torch.tensor([self.pad_token] * padding, dtype=torch.int64)
        ])

        # (SEQ_LEN,)
        label = torch.concat([
            # (len(token_ids),)
            torch.tensor(token_ids[1:], dtype=torch.int64),

            # (2, )
            torch.tensor([self.eos_token] * 2, dtype=torch.int64),

            # (padding,)
            torch.tensor([self.pad_token] * padding, dtype=torch.int64)
        ])

        return {
            # (SEQ_LEN,)
            "decoder_input": decoder_input,

            # (SEQ_LEN,) != (1,) --> (SEQ_LEN,) --> (1, SEQ_LEN) --> (1, SEQ_LEN) & (1, SEQ_LEN, SEQ_LEN) --> (1, SEQ_LEN, SEQ_LEN)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & self.lookback_mask(SEQ_LEN),

            # (SEQ_LEN,)
            "label": label
        }