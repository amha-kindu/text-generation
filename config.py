import torch, random, numpy

torch.manual_seed(3000)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.cuda.manual_seed_all(3000)
else:
    DEVICE = torch.device('cpu')
random.seed(3000)
numpy.random.seed(3000)

WORKING_DIR="/workspace/text-generation"
VOCAB_SIZE=32000
BATCH_SIZE = 64
EPOCHS = 100
INIT_LR = 2e-04
SEQ_LEN = 256
D_MODEL = 768
N_BLOCKS = 6
HEADS = 16
DROPOUT = 0.1
DFF = 3078
MODELS_FOLDER = f"{WORKING_DIR}/models"
PRELOAD_MODEL_FILEPATH = ""
TOKENIZER_FILEPATH = f"{WORKING_DIR}/tokenizers/amharic-bpe-tokenizer-{VOCAB_SIZE // 1000}k.model"
TB_LOG_DIR = f"{WORKING_DIR}/logs/gpt_model"
TRAINING_DATA_FILEPATH=f"{WORKING_DIR}/data/train_chunk_size_255.json"
VALIDATION_DATA_FILEPATH=f"{WORKING_DIR}/data/validate_chunk_size_255.json"
TEST_DATA_FILEPATH=f"{WORKING_DIR}/data/test_chunk_size_255.json"