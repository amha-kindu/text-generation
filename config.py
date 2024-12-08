import os, logging
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

GLOBAL_RANK = 0
LOCAL_RANK = 0
MASTER_RANK = 0
WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
WEIGHTS_DIRECTORY = os.path.join(WORKING_DIR, "weights")

class Config:
    def to_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"


class ModelConfig(Config):
    def __init__(self, **kwargs):
        self.d_model: int = kwargs.get("d_model", 512)
        self.n_blocks: int = kwargs.get("n_blocks", 6)
        self.vocab_size: int = kwargs.get("vocab_size", 50)
        self.dff: int = kwargs.get("dff", 2048)
        self.heads: int = kwargs.get("heads", 8)
        self.dropout: float = kwargs.get("dropout", 0.1)
        self.seq_len: int = kwargs.get("seq_len", 50)


class TrainingConfig(Config):
    def __init__(self, **kwargs):
        self.epochs: int = kwargs.get("epochs", 10)
        self.batch_size: int = kwargs.get("batch_size", 64)
        self.init_lr: float = kwargs.get("init_lr", 2e-04)
        self.tb_log_dir: str = kwargs.get("tb_log_dir", "logs")
        self.validation_samples: int = kwargs.get("validation_samples", 20)
        self.training_data: str = kwargs.get("training_data", os.path.join(WORKING_DIR, "data", "train_chunk_size_256.json"))
        self.validation_data: str = kwargs.get("training_data", os.path.join(WORKING_DIR, "data", "validate_chunk_size_256.json"))
        self.testing_data: str = kwargs.get("training_data", os.path.join(WORKING_DIR, "data", "test_chunk_size_256.json"))

DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()