import os
import json
import argparse
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
            sentences = json.loads(f.read())
            for sentence in sentences:
                yield self.preprocessor.preprocess(sentence, encode=False)
            self.current_file += 1
        yield None

    def __next__(self):
        item = next(self.generator)
        if item is not None:
            return item
        raise StopIteration


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument("--tokenizer-name", type=str, required=True, help="Name of the tokenizer to use")
    parser.add_argument("--training-data", type=str, default=DEFAULT_TRAINING_CONFIG.training_data, help="Path to the training dataset")
    parser.add_argument("--validation-data", type=str, default=DEFAULT_TRAINING_CONFIG.validation_data, help="Path to the validation dataset")
    parser.add_argument("--testing-data", type=str, default=DEFAULT_TRAINING_CONFIG.testing_data, help="Path to the testing dataset")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_MODEL_CONFIG.vocab_size, help="Vocabulary size to use")

    args = parser.parse_args()
    config = TrainingConfig(args)

    iterator = SentenceIterator([
        config.training_data, config.testing_data, config.validation_data
    ])

    spm.SentencePieceTrainer.Train(
        sentence_iterator=iterator,
        model_prefix=os.path.join('tokenizers', f"amharic-bpe-tokenizer-{args.vocab_size // 1000}k"),
        vocab_size=args.vocab_size,
        character_coverage=0.9995,
        model_type='bpe',
        unk_id=0, pad_id=1, bos_id=2, eos_id=3,
        unk_piece='[UNK]', pad_piece='[PAD]', bos_piece='[SOS]', eos_piece='[EOS]',
        train_extremely_large_corpus=True
    )