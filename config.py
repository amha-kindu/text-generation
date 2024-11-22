import torch, random, nltk

nltk.download('stopwords')

torch.manual_seed(3000)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.cuda.manual_seed_all(3000)
else:
    DEVICE = torch.device('cpu')
random.seed(3000)

VOCAB_SIZE=25000
BATCH_SIZE = 64
EPOCHS = 100
INIT_LR = 2e-04
SEQ_LEN = 52
D_MODEL = 512
N_BLOCKS = 6
HEADS = 16
DROPOUT = 0.1
DFF = 2048
MODELS_FOLDER = "models"
PRELOAD_MODEL_FILEPATH = ""
TOKENIZER_FILEPATH = f"tokenizers/amharic-bpe-tokenizer-v1-{VOCAB_SIZE // 1000}k.json"
TB_LOG_DIR = "logs/gpt_model"
DATASET_PATH = "data/amharic-texts.txt"