import torch, random, numpy

torch.manual_seed(3000)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.cuda.manual_seed_all(3000)
else:
    DEVICE = torch.device('cpu')
random.seed(3000)
numpy.random.seed(3000)

VOCAB_SIZE=25000
BATCH_SIZE = 64
EPOCHS = 100
INIT_LR = 2e-04
SEQ_LEN = 256
D_MODEL = 512
N_BLOCKS = 6
HEADS = 16
DROPOUT = 0.1
DFF = 2048
MODELS_FOLDER = "models"
PRELOAD_MODEL_FILEPATH = ""
TOKENIZER_FILEPATH = f"tokenizers/amharic-bpe-tokenizer-{VOCAB_SIZE // 1000}k.model"
TB_LOG_DIR = "logs/gpt_model"
TRAINING_DATA_FILEPATH="data/train_chunk_size_255.json"
VALIDATION_DATA_FILEPATH="data/validate_chunk_size_255.json"
TEST_DATA_FILEPATH="data/test_chunk_size_255.json"