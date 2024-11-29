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
        self.punc_and_special_chars = re.compile(r'[!@#$%^«»&*()…\[\]{};“”›’‘"\':,.‹/<>\?\|\`\´~\-=+፡።፤;፦፥፧፨፠፣፩፪፫፬፭፮፮፰፱፲፳፴፵፵፷፸፹፺፻01-9]')
        self.extra_whitespace = re.compile(r'\s{2,}')
        self.ascii_and_numbers = re.compile('[A-Za-z0-9]')
        self.non_amharic_chars=re.compile('[^\u1200-\u137F\s]+')
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

        # Remove punctuations and special characters
        text = self.remove_punc_and_special_chars(text)

        # Remove non-amharic character
        text = self.remove_ascii_and_numbers(text)
        
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
    
    def remove_punc_and_special_chars(self, text: str) -> str:
        text = self.punc_and_special_chars.sub(' ', text)
        text = self.extra_whitespace.sub(' ', text)

        return text

    def remove_ascii_and_numbers(self, text: str) -> str:
        rm_num_and_ascii = self.ascii_and_numbers.sub('', text)
        return self.non_amharic_chars.sub('', rm_num_and_ascii)
