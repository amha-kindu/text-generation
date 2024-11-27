import torch
from config import *
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from preprocessor import AmharicPreprocessor
from torch.utils.data.sampler import RandomSampler


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer: Tokenizer) -> None:
        super().__init__()
        self.texts: list[str] = texts

        self.tokenizer = tokenizer
        self.preprocessor = AmharicPreprocessor(tokenizer)

        # (1,)
        self.pad_token = torch.tensor([self.tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.texts)

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

    def __getitem__(self, index) -> dict:
        token_ids = self.preprocessor.preprocess(self.texts[index])
        padding = SEQ_LEN - len(token_ids) + 1

        # (SEQ_LEN,)
        decoder_input = torch.concat([
            # (len(token_ids) - 1,)
            torch.tensor(token_ids[:-1], dtype=torch.int64),

            # (padding,)
            torch.tensor([self.pad_token] * padding, dtype=torch.int64)
        ])

        # (SEQ_LEN,)
        label = torch.concat([
            # (len(token_ids) - 1,)
            torch.tensor(token_ids[1:], dtype=torch.int64),

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