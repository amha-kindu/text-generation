import json
import torch
import random
from config import *
from typing import Iterator
import sentencepiece as spm
from bisect import bisect_left
from abc import ABC, abstractmethod
from preprocessor import AmharicPreprocessor
from torch.utils.data import get_worker_info
from torch.utils.data import Dataset, Subset, IterableDataset, DataLoader, Sampler, ConcatDataset, SequentialSampler


class IDataset(Dataset, ABC):
    ignore_index = -100
    
    def __init__(self, file_path: str, tokenizer: spm.SentencePieceProcessor, max_len: int) -> None:
        self.max_len = max_len
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.preprocessor = AmharicPreprocessor()

        self.pad_token = self.tokenizer.pad_id()
    
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
    def get_loader(self, batch_size: int, sampler: Sampler=None) -> DataLoader:
        raise NotImplementedError("Subclass must implement the 'get_loader' abstractmethod!")
    
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
            torch.tensor([self.ignore_index] * padding, dtype=torch.int64)
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

    def get_loader(self, batch_size: int, sampler: Sampler=None) -> DataLoader:
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
        
    def get_loader(self, batch_size: int, sampler: Sampler=None) -> DataLoader:
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
    def __init__(self, type: str, system_text=None) -> None:
        self.type = type
        self.exchanges = []
        self.system_text = system_text
    
    def add_exchange(self, input_text: str, output_text: str):
        self.exchanges.append({
            "input": input_text,
            "output": output_text
        })


class FineTuningDataset(IDataset):
    def __init__(self, intent: str, file_path: str, tokenizer: spm.SentencePieceProcessor, max_len: int) -> None:
        super().__init__(file_path, tokenizer, max_len)
        self.bot_token = self.tokenizer.PieceToId("[BOT]")
        self.user_token = self.tokenizer.PieceToId("[USER]")
        self.stop_token = self.tokenizer.PieceToId("[STOP]")
        self.system_token = self.tokenizer.PieceToId("[SYSTEM]")

        skips = 0
        self.samples = []
        self.file = open(file_path, 'r', encoding='utf-8')
        file_name = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            LOGGER.info(f"\033[93mLoading data from {file_name}{'/' + intent if intent else ''}...\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
            data = json.load(f)
            for sample in data:
                if intent and sample['type'] != intent:
                    continue
                system_prompt = sample.get("system", "")
                if system_prompt:
                    system_prompt = self.preprocessor.execute(system_prompt)
                conv = Conversation(sample['type'], system_prompt)
                for exchange in sample["exchanges"]:
                    try:
                        conv.add_exchange(
                            self.preprocessor.execute(exchange["input"]),
                            self.preprocessor.execute(exchange["output"])
                        )
                    except Exception as e:
                        LOGGER.error('File must be in JSON format [{"system": ..., "exchanges": [{"input": ..., "output": ...}, ...}]]')
                        exit(1)
                try:
                    input, output = self.get_io_tensors(conv)
                    self.samples.append((input, output))
                except ValueError:
                    skips += 1
                    continue

                if self.samples and len(self.samples) % 30000 == 0:
                    LOGGER.info(f"\033[93mLoaded {len(self.samples)} samples from {file_name}{'/' + intent if intent else ''}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None    
        LOGGER.info(f"\033[92mDone! Loaded {len(self.samples)} samples from {file_name}{'/' + intent if intent else ''}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
        LOGGER.info(f"\033[93mSkipped {skips} samples from {file_name}{'/' + intent if intent else ''}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
        LOGGER.info(f"\033[93mUsing {len(self.samples)} samples from {file_name}{'/' + intent if intent else ''}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def get_loader(self, batch_size: int, sampler: Sampler=None) -> DataLoader:
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
        # Input Structure:     [SYSTEM] K L M N O [USER] A B C D E [BOT] F G H I   J    ... [USER] Q R S T U [BOT] V W X Y   Z    $ $ $
        # Output Structure:        $    $ $ $ $ $    $   $ $ $ $ $   F   G H I J [STOP] ...    $   $ $ $ $ $   V   W X Y Z [STOP] $ $ $
        
        input_ids: list[int] = []
        output_ids: list[int] = []
        if conv.system_text:
            input_ids.extend([
                self.system_token,
                *self.tokenizer.Encode(conv.system_text, out_type=int)
            ])
            output_ids.extend([self.ignore_index] * len(input_ids))
        
        exchanges_ipt, exchanges_opt = [], []
        for exchange in reversed(conv.exchanges):
            input_token_ids = self.tokenizer.Encode(exchange["input"], out_type=int)
            output_token_ids = self.tokenizer.Encode(exchange["output"], out_type=int)
            
            if len(input_ids) + len(exchanges_ipt) + len(input_token_ids) + len(output_token_ids) + 2 > self.max_len:
                break
            
            # [USER] A B C ... H I J [BOT] K L M ... X Y   Z   
            #   $    $ $ $ ... $ $ $   K   L M O ... Y Z [STOP]
            if input_token_ids and output_token_ids:
                exchanges_ipt = [
                    self.user_token,
                    *input_token_ids,
                    self.bot_token,
                    *output_token_ids
                ] + exchanges_ipt
                exchanges_opt = [
                    *[self.ignore_index] * (len(input_token_ids) + 1),
                    *output_token_ids,
                    self.stop_token
                ] + exchanges_opt
                
        if not exchanges_ipt:
            raise ValueError("Input text too long(or no exchanges)!")
        
        input_ids.extend(exchanges_ipt)
        output_ids.extend(exchanges_opt)

        padding = self.max_len - len(input_ids)
        
        # (SEQ_LEN,)
        input: torch.Tensor = torch.concat([
            # (len(input_ids),)
            torch.tensor(input_ids, dtype=torch.int64),
            
            # (padding,)
            torch.tensor([self.pad_token] * padding, dtype=torch.int64)
        ])[:self.max_len]
        
        # (SEQ_LEN,)
        output: torch.Tensor = torch.concat([
            # (len(output_ids),)
            torch.tensor(output_ids, dtype=torch.int64),
            
            # (padding,)
            torch.tensor([self.ignore_index] * padding, dtype=torch.int64),
        ])[:self.max_len]
        
        return input, output


class MultiTaskDataset(Dataset):
    ignore_index = IDataset.ignore_index
    
    def __init__(self, datasets: dict[str, IDataset]) -> None:
        self.tokenizer = datasets[list(datasets.keys())[0]].tokenizer
        self.task_names = list(datasets.keys())
        self.datasets = [datasets[k] for k in self.task_names]
        self.lengths = [len(ds) for ds in self.datasets]
        self.offsets = []
        total = 0
        for L in self.lengths:
            self.offsets.append(total)
            total += L
        self.total_len = total

    def __len__(self):
        return self.total_len
    
    def get_loader(self, batch_size: int, sampler: Sampler=None) -> DataLoader:
        return DataLoader(self, batch_size, shuffle=(sampler is None), sampler=sampler)

    def __getitem__(self, global_index: int):
        task_id = bisect_left(self.offsets, global_index)
        if task_id == len(self.offsets) or global_index != self.offsets[task_id]:
            task_id -= 1
        return self.datasets[task_id][global_index - self.offsets[task_id]]


class TemperatureMixSampler(Sampler[int]):
    def __init__(self, mt_dataset: MultiTaskDataset, alpha: float = 0.5,
                 steps_per_epoch: int | None = None):
        self.alpha = alpha
        self.mt = mt_dataset
        self.lengths = torch.tensor(self.mt.lengths, dtype=torch.float)
        self.offsets = torch.tensor(self.mt.offsets, dtype=torch.long)
        
        weights = (self.lengths.clamp(min=1.0)) ** alpha # probs ∝ n^alpha
        self.task_probs = (weights / weights.sum()).tolist()
        self.steps_per_epoch = steps_per_epoch or sum(self.mt.lengths)

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        probs = torch.tensor(self.task_probs, dtype=torch.float)
        for _ in range(self.steps_per_epoch):
            task = torch.multinomial(probs, 1).item()
            local_len = int(self.lengths[task].item())
            j = random.randrange(local_len)
            global_index = self.offsets[task].item() + j
            yield global_index


class RollingShardsDataset:  
    def __init__(self, datasets: dict[str, Dataset], num_shards: int):
        assert num_shards >= 1
        self._round = 0
        self.num_shards = num_shards
        self.names = list(datasets.keys())
        self.datasets = [datasets[n] for n in self.names]

        self._splits = []
        for ds in self.datasets:
            self._splits.append(
                self._even_splits(len(ds), num_shards)
            )

    @staticmethod
    def _even_splits(n: int, k: int):
        base, rem = divmod(n, k)
        splits, start = [], 0
        for i in range(k):
            end = start + base + (1 if i < rem else 0)
            splits.append((start, end))
            start = end
        return splits

    def _dataset_for(self, shard_idx: int) -> Dataset:
        k = shard_idx % self.num_shards
        parts = []
        for ds, splits in zip(self.datasets, self._splits):
            s, e = splits[k]
            if e > s:
                idx = [i for i in range(s, e)]
                parts.append(Subset(ds, idx))
        return ConcatDataset(parts) if parts else ConcatDataset([])

    def get_loader(self, batch_size: int, sampler: Sampler=None, shard_idx: int = 0) -> DataLoader:
        ds = self._dataset_for(shard_idx)
        if sampler is None:
            sampler = SequentialSampler(ds)
        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False
        )
    
    def next_loader(self, batch_size: int) -> DataLoader:
        shard_idx = self._round % self.num_shards
        self._round += 1
        return self.get_loader(batch_size, shard_idx=shard_idx)
