import re
import sentencepiece as spm
from abc import ABC, abstractmethod


class PreprocessingPipeline(ABC):   
    def __init__(self, tokenizer: spm.SentencePieceProcessor) -> None:
        super().__init__()
        self.tokenizer = tokenizer
    
    @abstractmethod
    def preprocess(self, text: str, encode=True) -> str:
        pass

    
class AmharicPreprocessor(PreprocessingPipeline):
    def __init__(self, tokenizer: spm.SentencePieceProcessor) -> None:
        super().__init__(tokenizer)
        self.extra_whitespace = re.compile(r'\s{2,}')
        self.normalization_patterns = [
            (re.compile('[ሃኅኃሐሓኻ]'), 'ሀ'),
            (re.compile('[ሑኁዅ]'), 'ሁ'),
            (re.compile('[ኂሒኺ]'), 'ሂ'),
            (re.compile('[ኌሔዄ]'), 'ሄ'),
            (re.compile('[ሕኅ]'), 'ህ'),
            (re.compile('[ኆሖኾ]'), 'ሆ'),
            (re.compile('[ሠ]'), 'ሰ'),
            (re.compile('[ሡ]'), 'ሱ'),
            (re.compile('[ሢ]'), 'ሲ'),
            (re.compile('[ሣ]'), 'ሳ'),
            (re.compile('[ሤ]'), 'ሴ'),
            (re.compile('[ሥ]'), 'ስ'),
            (re.compile('[ሦ]'), 'ሶ'),
            (re.compile('[ዓኣዐ]'), 'አ'),
            (re.compile('[ዑ]'), 'ኡ'),
            (re.compile('[ዒ]'), 'ኢ'),
            (re.compile('[ዔ]'), 'ኤ'),
            (re.compile('[ዕ]'), 'እ'),
            (re.compile('[ዖ]'), 'ኦ'),
            (re.compile('[ጸ]'), 'ፀ'),
            (re.compile('[ጹ]'), 'ፁ'),
            (re.compile('[ጺ]'), 'ፂ'),
            (re.compile('[ጻ]'), 'ፃ'),
            (re.compile('[ጼ]'), 'ፄ'),
            (re.compile('[ጽ]'), 'ፅ'),
            (re.compile('[ጾ]'), 'ፆ'),
            (re.compile('(ሉ[ዋአ])'), 'ሏ'),
            (re.compile('(ሙ[ዋአ])'), 'ሟ'),
            (re.compile('(ቱ[ዋአ])'), 'ቷ'),
            (re.compile('(ሩ[ዋአ])'), 'ሯ'),
            (re.compile('(ሱ[ዋአ])'), 'ሷ'),
            (re.compile('(ሹ[ዋአ])'), 'ሿ'),
            (re.compile('(ቁ[ዋአ])'), 'ቋ'),
            (re.compile('(ቡ[ዋአ])'), 'ቧ'),
            (re.compile('(ቹ[ዋአ])'), 'ቿ'),
            (re.compile('(ሁ[ዋአ])'), 'ኋ'),
            (re.compile('(ኑ[ዋአ])'), 'ኗ'),
            (re.compile('(ኙ[ዋአ])'), 'ኟ'),
            (re.compile('(ኩ[ዋአ])'), 'ኳ'),
            (re.compile('(ዙ[ዋአ])'), 'ዟ'),
            (re.compile('(ጉ[ዋአ])'), 'ጓ'),
            (re.compile('(ደ[ዋአ])'), 'ዷ'),
            (re.compile('(ጡ[ዋአ])'), 'ጧ'),
            (re.compile('(ጩ[ዋአ])'), 'ጯ'),
            (re.compile('(ጹ[ዋአ])'), 'ጿ'),
            (re.compile('(ፉ[ዋአ])'), 'ፏ'),
            (re.compile('[ቊ]'), 'ቁ'),
            (re.compile('[ኵ]'), 'ኩ')
        ]
    
    def preprocess(self, text: str, encode=True) -> str:
        # Remove leading and trailing spaces
        text = text.strip()

        # Character level mismatch
        text = self.normalize_char_level_missmatch(text)

        # Remove extra whitespace
        text = self.extra_whitespace.sub(' ', text)
        
        if encode:
            return self.tokenizer.Encode(
                text,
                out_type=int
            )
        else:
            return text

    def normalize_char_level_missmatch(self, text: str) -> str:
        for pattern, replacement in self.normalization_patterns:
            text = pattern.sub(replacement, text)
        return text