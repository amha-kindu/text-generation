import torch, random, nltk

nltk.download('stopwords')

torch.manual_seed(3000)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.cuda.manual_seed_all(3000)
else:
    DEVICE = torch.device('cpu')
random.seed(3000)

BATCH_SIZE = 64
EPOCHS = 100
INIT_LR = 2e-04
SEQ_LEN = 52
D_MODEL = 512
N_BLOCKS = 6
HEADS = 16
DROPOUT = 0.1
DFF = 2048
MODEL_FOLDER = "models"
PRELOAD_MODEL_SUFFIX = ""
TOKENIZER_FILEPATH = "tokenizers/amharic-bpe-tokenizer-v1-25k.json"
TB_LOG_DIR = "logs"
DATASET_PATH = "data/parallel-corpus-en-am-v3.5.json"