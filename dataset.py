import torch
from config import *
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader

from preprocessor import AmharicPreprocessor

class TextDataset(Dataset):
    def __init__(self, dataset: list[dict], tokenizer: Tokenizer) -> None:
        super().__init__()
        self.dataset: list[dict] = dataset

        self.tokenizer = tokenizer
        self.preprocessor = AmharicPreprocessor(tokenizer)
        
        # (1,)
        self.pad_token = torch.tensor([self.tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

        # (1, )
        self.continue_token = torch.tensor([self.tokenizer.token_to_id("[CONT]")], dtype=torch.int64)
        
    def __len__(self):
        return len(self.dataset)
    
    def batch_iterator(self, batch_size: int) -> DataLoader:
        return DataLoader(self, batch_size, shuffle=True)

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
        text_pairs = self.dataset[index]
        input_text = text_pairs["input"]
        target_text = text_pairs["target"]
                
        input_token_ids = self.preprocessor.preprocess(input_text)
        target_token_ids = self.preprocessor.preprocess(target_text)

        input_padding = SEQ_LEN - len(input_token_ids)
        target_padding = SEQ_LEN - len(target_token_ids) - 1
                
        # (seq_len,)
        encoder_input = torch.concat([
            # (len(input_token_ids),)
            torch.tensor(input_token_ids, dtype=torch.int64),

            # (input_padding,)
            torch.tensor([self.pad_token] * input_padding, dtype=torch.int64)
        ])     
        
        # (seq_len,)
        decoder_input = torch.concat([
            # (1, )
            self.continue_token,
            
            # (len(target_token_ids),)
            torch.tensor(target_token_ids, dtype=torch.int64),

            # (target_padding,)
            torch.tensor([self.pad_token] * target_padding, dtype=torch.int64)
        ])                    
        
        # (seq_len,)
        label = torch.concat([
            # (len(target_token_ids),)
            torch.tensor(target_token_ids, dtype=torch.int64),

            # (target_padding,)
            torch.tensor([self.pad_token] * target_padding, dtype=torch.int64)
        ])     
        
        return {
            # (seq_len,)
            "encoder_input": encoder_input, 
            
            # (seq_len,)                                    
            "decoder_input": decoder_input,    
                                             
            # (seq_len,) != (1,) --> (seq_len,) --> (1, 1, seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
                            
            # (seq_len,) != (1,) --> (seq_len,) --> (1, 1, seq_len) --> (1, seq_len) & (1, seq_len, seq_len) --> (1, seq_len, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & self.lookback_mask(SEQ_LEN),  
            
            # (seq_len,)         
            "label": label
        }