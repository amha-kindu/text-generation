import json
import torch
from abc import ABC, abstractmethod
from config import *
from typing import Iterator
from preprocessor import AmharicPreprocessor
import sentencepiece as spm
from torch.utils.data import get_worker_info
from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler


def get_chunks(token_ids, chunk_size, overlap=20):
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(token_ids) - overlap, step):
        chunk = token_ids[i : i + chunk_size]
        chunks.append(chunk)
    return chunks


class IDataset(Dataset, ABC):
    def __init__(self, file_path: str, tokenizer: spm.SentencePieceProcessor, max_len: int) -> None:
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.pad_token = torch.tensor([self.tokenizer.pad_id()], dtype=torch.int64)
        self.eos_token = torch.tensor([self.tokenizer.eos_id()], dtype=torch.int64)
        self.bos_token = torch.tensor([self.tokenizer.bos_id()], dtype=torch.int64)
            
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

    @abstractmethod
    def batch_iterator(self, batch_size: int, sampler: Sampler=None) -> DataLoader:
        raise NotImplementedError("Subclass must implement the 'batch_iterator' abstractmethod!")
    
    def datapoint(self, token_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        padding = self.max_len - len(token_ids)
        
        # (SEQ_LEN,)
        decoder_input: torch.Tensor = torch.concat([
            # (len(token_ids),)
            torch.tensor(token_ids, dtype=torch.int64),
            
            # (padding,)
            torch.tensor([self.pad_token] * padding, dtype=torch.int64)
        ])[:self.max_len]

        # (SEQ_LEN,)
        target = torch.concat([
            # (len(token_ids) - 1,)
            torch.tensor(token_ids[1:], dtype=torch.int64),
            
            # (1,)
            torch.tensor([self.eos_token], dtype=torch.int64),
            
            # (padding,)
            torch.tensor([self.pad_token] * padding, dtype=torch.int64)
        ])[:self.max_len]
        
        return decoder_input, target


class TextDataset(IDataset):
    def __init__(self, file_path: str, tokenizer: spm.SentencePieceProcessor, max_len: int) -> None:
        super().__init__(file_path, tokenizer, max_len)

        self.texts = []
        self.file = open(file_path, 'r', encoding='utf-8')
        preprocessor = AmharicPreprocessor()
        file_name = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            LOGGER.info(f"\033[93mLoading data from {file_name}...\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
            for line in f:
                # Parse each line as a separate JSON object
                text = json.loads(line.strip())
                preprocessed_text = preprocessor.execute(text)
                if preprocessed_text:
                    self.texts.append(preprocessed_text)
                    if self.texts and len(self.texts) % 100000 == 0:
                        LOGGER.info(f"\033[93mLoaded {len(self.texts)} texts from {file_name}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
        LOGGER.info(f"\033[92mDone! Loaded {len(self.texts)} texts from {file_name}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None

    def __len__(self) -> int:
        return len(self.texts)

    def batch_iterator(self, batch_size: int, sampler: Sampler=None) -> DataLoader:
        return DataLoader(self, batch_size, shuffle=(sampler is None), sampler=sampler)

    def __getitem__(self, index) -> Iterator[dict]:
        text = self.texts[index]
        
        token_ids = self.tokenizer.Encode(text, out_type=int)
        for chunk in get_chunks(token_ids, self.max_len):
            if not chunk:
                continue
            
            decoder_input, target = self.datapoint(chunk)

            yield {
                # (SEQ_LEN,)
                "decoder_input": decoder_input,

                # (SEQ_LEN,) != (1,) --> (SEQ_LEN,) --> (1, SEQ_LEN) --> (1, SEQ_LEN) & (1, SEQ_LEN, SEQ_LEN) --> (1, SEQ_LEN, SEQ_LEN)
                "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0) & self.lookback_mask(self.max_len),

                # (SEQ_LEN,)
                "label": target
            }


class IStreamDataset(IterableDataset, IDataset):

    @staticmethod
    @abstractmethod
    def collate_fn(batch):
        # Assumes all items are same shape already (fixed-length seqs)
        return {
            "decoder_input": torch.stack([item["decoder_input"] for item in batch]),
            "decoder_mask": torch.stack([item["decoder_mask"] for item in batch]),
            "label": torch.stack([item["label"] for item in batch])
        }


class StreamingTextDataset(IStreamDataset):
    def __init__(self, file_path: str, tokenizer: spm.SentencePieceProcessor, max_len: int) -> None:
        super().__init__(file_path, tokenizer, max_len)
    
    @staticmethod
    def collate_fn(batch):
        return {
            "decoder_input": torch.stack([item["decoder_input"] for item in batch]),
            "decoder_mask": torch.stack([item["decoder_mask"] for item in batch]),
            "label": torch.stack([item["label"] for item in batch])
        }
        
    def batch_iterator(self, batch_size: int, sampler: Sampler=None) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            num_workers=LOCAL_WORLD_SIZE,
            pin_memory=True
        )

    def __iter__(self) -> Iterator[dict]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        effective_rank = GLOBAL_RANK * num_workers + worker_id
        effective_world_size = WORLD_SIZE * num_workers
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # Shard the stream
                if i % effective_world_size != effective_rank:
                    continue
                
                # Parse each line as a separate JSON object
                text = json.loads(line.strip())
                
                # Tokenize text(split into chunks if text is bigger than max_len)
                token_ids = self.tokenizer.Encode(text, out_type=int)
                for chunk in get_chunks(token_ids, self.max_len):
                    if not chunk:
                        continue

                    decoder_input, target = self.datapoint(chunk)

                    yield {
                        # (SEQ_LEN,)
                        "decoder_input": decoder_input,
                        
                        # (SEQ_LEN,) != (1,) --> (SEQ_LEN,) --> (1, SEQ_LEN) --> (1, SEQ_LEN) & (1, SEQ_LEN, SEQ_LEN) --> (1, SEQ_LEN, SEQ_LEN) --> (1, 1, SEQ_LEN, SEQ_LEN)
                        "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0) & self.lookback_mask(self.max_len),
                        
                        # (SEQ_LEN,)
                        "label": target
                    }
