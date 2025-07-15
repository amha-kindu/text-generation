import json
import torch
from config import *
from typing import Iterator
import sentencepiece as spm
from abc import ABC, abstractmethod
from preprocessor import AmharicPreprocessor
from torch.utils.data import get_worker_info
from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler


class IDataset(Dataset, ABC):
    def __init__(self, file_path: str, tokenizer: spm.SentencePieceProcessor, max_len: int) -> None:
        self.max_len = max_len
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.preprocessor = AmharicPreprocessor()

        self.pad_token = torch.tensor([self.tokenizer.pad_id()], dtype=torch.int64)
    
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
    
    def get_io_tensors(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        # Text:               A B C D E
        # Input Structure:    A B C D $ $ $ $
        # Output Structure:   B C D E $ $ $ $

        text = self.preprocessor.execute(text)
        token_ids = self.tokenizer.Encode(text, out_type=int)[:self.max_len]        
        padding = self.max_len - len(token_ids) + 1
        
        # (SEQ_LEN,)
        input: torch.Tensor = torch.concat([
            # (len(token_ids) - 1,)
            torch.tensor(token_ids[:-1], dtype=torch.int64),
            
            # (padding,)
            torch.tensor([self.pad_token] * padding, dtype=torch.int64)
        ])[:self.max_len]

        # (SEQ_LEN,)
        output = torch.concat([
            # (len(token_ids) - 1,)
            torch.tensor(token_ids[1:], dtype=torch.int64),
            
            # (padding,)
            torch.tensor([self.pad_token] * padding, dtype=torch.int64)
        ])[:self.max_len]

        return input, output


class TextDataset(IDataset):
    def __init__(self, file_path: str, tokenizer: spm.SentencePieceProcessor, max_len: int) -> None:
        super().__init__(file_path, tokenizer, max_len)

        self.texts = []
        self.file = open(file_path, 'r', encoding='utf-8')
        file_name = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            LOGGER.info(f"\033[93mLoading data from {file_name}...\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
            for line in f:
                # Parse each line as a separate JSON object
                text = json.loads(line.strip())
                preprocessed_text = self.preprocessor.execute(text)
                if preprocessed_text:
                    self.texts.append(preprocessed_text)
                    if self.texts and len(self.texts) % 100000 == 0:
                        LOGGER.info(f"\033[93mLoaded {len(self.texts)} samples from {file_name}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
        LOGGER.info(f"\033[92mDone! Loaded {len(self.texts)} samples from {file_name}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None

    def __len__(self) -> int:
        return len(self.texts)

    def batch_iterator(self, batch_size: int, sampler: Sampler=None) -> DataLoader:
        return DataLoader(self, batch_size, shuffle=(sampler is None), sampler=sampler)

    def __getitem__(self, index) -> dict:
        text = self.texts[index]
        
        input_tensor, output_tensor = self.get_io_tensors(text)
        return {
            # (SEQ_LEN,)
            "decoder_input": input_tensor,
            
            # (SEQ_LEN,) != (1,) --> (SEQ_LEN,) --> (1, SEQ_LEN) --> (1, SEQ_LEN) & (1, SEQ_LEN, SEQ_LEN) --> (1, SEQ_LEN, SEQ_LEN)
            "decoder_mask": (input_tensor != self.pad_token).unsqueeze(0) & self.lookback_mask(self.max_len),
            
            # (SEQ_LEN,)
            "label": output_tensor
        }
    

class StreamingTextDataset(IterableDataset, IDataset):
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
                
                input_tensor, output_tensor = self.get_io_tensors(text)
                yield {
                    # (SEQ_LEN,)
                    "decoder_input": input_tensor,
                    
                    # (SEQ_LEN,) != (1,) --> (SEQ_LEN,) --> (1, SEQ_LEN) --> (1, SEQ_LEN) & (1, SEQ_LEN, SEQ_LEN) --> (1, SEQ_LEN, SEQ_LEN)
                    "decoder_mask": (input_tensor != self.pad_token).unsqueeze(0) & self.lookback_mask(self.max_len),
                    
                    # (SEQ_LEN,)
                    "label": output_tensor
                }


class Conversation:
    def __init__(self, system_text=None):
        self.exchanges = []
        self.system_text = system_text
    
    def add_exchange(self, input_text: str, output_text: str):
        self.exchanges.append({
            "input": input_text,
            "output": output_text
        })


class FineTuningDataset(IDataset):
    def __init__(self, file_path: str, tokenizer: spm.SentencePieceProcessor, max_len: int) -> None:
        
        super().__init__(file_path, tokenizer, max_len)        
        self.bot_token = torch.tensor([self.tokenizer.PieceToId("[BOT]")], dtype=torch.int64)
        self.user_token = torch.tensor([self.tokenizer.PieceToId("[USER]")], dtype=torch.int64)
        self.stop_token = torch.tensor([self.tokenizer.PieceToId("[STOP]")], dtype=torch.int64)
        self.system_token = torch.tensor([self.tokenizer.PieceToId("[SYSTEM]")], dtype=torch.int64)

        skips = 0
        self.samples = []
        self.file = open(file_path, 'r', encoding='utf-8')
        file_name = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            LOGGER.info(f"\033[93mLoading data from {file_name}...\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
            data = json.load(f)
            for sample in data:
                system_prompt = sample.get("system", "")
                if system_prompt:
                    system_prompt = self.preprocessor.execute(system_prompt)
                conv = Conversation(system_prompt)
                for exchange in sample["exchanges"]:
                    try:
                        conv.add_exchange(
                            self.preprocessor.execute(exchange["input"]),
                            self.preprocessor.execute(exchange["output"])
                        )
                    except Exception as e:
                        LOGGER.error('File must be in JSON format [{"system": ..., "exchanges": [{"input": ..., "output": ...}, ...}] ')
                        exit(1)
                try:
                    input, output = self.get_io_tensors(conv)
                    self.samples.append((input, output))
                except ValueError:
                    skips += 1

                if self.samples and len(self.samples) % 30000 == 0:
                    LOGGER.info(f"\033[93mLoaded {len(self.samples)} samples from {file_name}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
        LOGGER.info(f"\033[92mDone! Loaded {len(self.samples)} samples from {file_name}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
        LOGGER.info(f"\033[93mSkipped {skips} samples from {file_name}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
        LOGGER.info(f"\033[93mUsing {len(self.samples)} samples from {file_name}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def batch_iterator(self, batch_size: int, sampler: Sampler=None) -> DataLoader:
        return DataLoader(self, batch_size, shuffle=(sampler is None), sampler=sampler)

    def __getitem__(self, index) -> dict:
        input_tensor, output_tensor = self.samples[index]
        
        return {
            # (SEQ_LEN,)
            "decoder_input": input_tensor,
                        
            # (SEQ_LEN,) != (1,) --> (SEQ_LEN,) --> (1, SEQ_LEN) --> (1, SEQ_LEN) & (1, SEQ_LEN, SEQ_LEN) --> (1, SEQ_LEN, SEQ_LEN)
            "decoder_mask": (input_tensor != self.pad_token).unsqueeze(0) & self.lookback_mask(self.max_len),
            
            # (SEQ_LEN,)
            "label": output_tensor
        }
    
    def get_io_tensors(self, conv: Conversation) -> tuple[torch.Tensor, torch.Tensor]:
        # System:              K L M N O
        # Conversation:        User: A B C D E 
        #                      Bot:  F G H I J
        #                      ...
        #                      User: Q R S T U
        #                      Bot:  V W X Y Z
        # Input Structure:     [SYSTEM] K L M N O [USER] A B C D E [BOT] F G H I J ... [USER] Q R S T U [BOT] V W X Y   Z    $ $ $
        # Output Structure:        $    $ $ $ $ $    $   $ $ $ $ $   $   $ $ $ $ $ ...    $   $ $ $ $ $    V  W X Y Z [STOP] $ $ $
        
        input_ids: list[int] = []
        if conv.system_text:
            input_ids.append(self.system_token)
            conv.system_text = self.preprocessor.execute(conv.system_text)
            input_ids.extend(self.tokenizer.Encode(conv.system_text, out_type=int))
        
        exchanges = []
        for exchange in conv.exchanges:
            input_text = self.preprocessor.execute(exchange["input"])
            output_text = self.preprocessor.execute(exchange["output"])
            input_token_ids = self.tokenizer.Encode(input_text, out_type=int)
            output_token_ids = self.tokenizer.Encode(output_text, out_type=int)
            exchanges.extend([
                self.user_token,
                *input_token_ids,
                self.bot_token,
                *output_token_ids
            ])
        
        # Discard tokens of the earlier exchanges if input_ids gets too long(exceeds max_len)
        if len(input_ids) + len(exchanges) > self.max_len:
            input_ids.extend(exchanges[len(input_ids) + len(exchanges) - self.max_len:])
        else:
            input_ids.extend(exchanges)
        
        input_suffix_padding = self.max_len - len(input_ids)
        last_bot_token_idx = max(i for i, v in enumerate(input_ids) if v == self.bot_token)        
        
        # (SEQ_LEN,)
        input: torch.Tensor = torch.concat([
            # (len(input_ids),)
            torch.tensor(input_ids, dtype=torch.int64),
            
            # (input_suffix_padding,)
            torch.tensor([self.pad_token] * input_suffix_padding, dtype=torch.int64)
        ])[:self.max_len]
        
        # (SEQ_LEN,)
        output: torch.Tensor = torch.concat([
            # (last_bot_token_idx,)
            torch.tensor([self.pad_token] * last_bot_token_idx, dtype=torch.int64),
            
            # (len(input_ids) - last_bot_token_idx - 1,)
            torch.tensor(input_ids[last_bot_token_idx + 1:], dtype=torch.int64),
            
            # (1,)
            torch.tensor([self.stop_token], dtype=torch.int64),
            
            # (input_suffix_padding,)
            torch.tensor([self.pad_token] * input_suffix_padding, dtype=torch.int64),
        ])[:self.max_len]
        
        return input, output
