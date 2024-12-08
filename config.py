import logging
import torch, random, numpy


torch.manual_seed(3000)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.cuda.manual_seed_all(3000)
else:
    DEVICE = torch.device('cpu')
random.seed(3000)
numpy.random.seed(3000)

LOGGER = logging.getLogger(str(DEVICE).upper())
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

WORKING_DIR = "/workspace/text-generation"
GLOBAL_RANK = 0
LOCAL_RANK = 0
VOCAB_SIZE = 32000
BATCH_SIZE = 64
EPOCHS = 100
INIT_LR = 2e-04
SEQ_LEN = 50
D_MODEL = 768
N_BLOCKS = 6
HEADS = 16
DROPOUT = 0.1
DFF = 3072
GLOBAL_RANK = 0
MASTER_RANK = 0
WEIGHTS_DIRECTORY = f"{WORKING_DIR}/weights"
PRELOAD_WEIGHTS_FILEPATH = ""
TB_LOG_DIR = f"{WORKING_DIR}/logs"
TEST_DATA_FILEPATH=f"{WORKING_DIR}/data/test_chunk_size_256.json"
TRAINING_DATA_FILEPATH=f"{WORKING_DIR}/data/train_chunk_size_256.json"
VALIDATION_DATA_FILEPATH=f"{WORKING_DIR}/data/validate_chunk_size_256.json"
TOKENIZER_FILEPATH = f"{WORKING_DIR}/tokenizers/amharic-bpe-tokenizer-{VOCAB_SIZE // 1000}k.model"