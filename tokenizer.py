import sentencepiece as spm


class SentencePieceProcessor(spm.SentencePieceProcessor):
    def __init__(self, max_len: int, *args, **kwargs):
        super().__init__()
        self.max_len: int = max_len

    def Encode(self, input, out_type=None, add_bos=None, add_eos=None, reverse=None, emit_unk_piece=None, enable_sampling=None, nbest_size=None, alpha=None, num_threads=None):
        token_ids = super().Encode(input, out_type, add_bos, add_eos, reverse, emit_unk_piece, enable_sampling, nbest_size, alpha, num_threads)
        return token_ids[:self.max_len]
