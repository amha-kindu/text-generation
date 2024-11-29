import os
from config import *
import sentencepiece as spm
from typing import Generator, Iterator
from preprocessor import AmharicPreprocessor


class SentenceIterator(Iterator):
    def __init__(self, file_paths: list[str]):
        self.current_file = 0
        self.file_paths = file_paths
        self.preprocessor = AmharicPreprocessor(None)
        self.generator = self.__gen__()

    def __iter__(self) -> Iterator:
        return self
    
    def __gen__(self) -> Generator:
        with open(self.file_paths[self.current_file], 'r', encoding='utf-8') as f:
            for line in f:
                yield self.preprocessor.preprocess(line, encode=False)
            self.current_file += 1
        yield None

    def __next__(self):
        item = next(self.generator)
        if item is not None:
            return item
        raise StopIteration


if __name__ == "__main__":
    iterator = SentenceIterator([
        TRAINING_DATA_FILEPATH, TEST_DATA_FILEPATH, VALIDATION_DATA_FILEPATH
    ])

    spm.SentencePieceTrainer.Train(
        sentence_iterator=iterator,
        model_prefix=os.path.join('tokenizers', f"amharic-bpe-tokenizer-{VOCAB_SIZE // 1000}k"),
        vocab_size=VOCAB_SIZE,
        character_coverage=0.9995,
        model_type='bpe',
        unk_id=0, pad_id=1, bos_id=2, eos_id=3,
        unk_piece='[UNK]', pad_piece='[PAD]', bos_piece='[SOS]', eos_piece='[EOS]',
        train_extremely_large_corpus=True
    )