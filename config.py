import os, sys, logging
import torch, random, numpy

random.seed(4321)
torch.manual_seed(4321)
numpy.random.seed(4321)

class PartialLineHandler(logging.StreamHandler):
    def emit(self, record):
        # Determine whether this record is 'partial'
        partial = getattr(record, 'partial', False)

        try:
            msg = self.format(record)
            if partial:
                # Write the message without a newline
                self.stream.write(msg)
            else:
                # Normal behavior: newline at the end
                self.stream.write(msg + "\n")
            self.flush()
        except Exception:
            self.handleError(record)

class PartialFormatter(logging.Formatter):
    def format(self, record):
        if getattr(record, 'partial', False):
            return record.getMessage()
        else:
            return super().format(record)
        
COORDINATOR_RANK = 0
GLOBAL_RANK = int(os.getenv("RANK", "0"))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
WEIGHTS_DIRECTORY = os.path.join(WORKING_DIR, "weights")

DEVICE = torch.device('cpu')
MIXED_PRECISION_ENABLED = False
LOGGER = logging.getLogger(f"CPU {GLOBAL_RANK}")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(3000)
    LOGGER = logging.getLogger(f"GPU {GLOBAL_RANK}")
    DEVICE = torch.device(f'cuda:{LOCAL_RANK}')
    torch.cuda.set_device(DEVICE)
    MIXED_PRECISION_ENABLED = torch.amp.autocast_mode.is_autocast_available(DEVICE.type)

LOGGER.setLevel(logging.INFO)
stream_handler = PartialLineHandler(sys.stdout)
stream_handler.setFormatter(PartialFormatter(
    fmt="\033[95m%(asctime)s\033[0m - \033[94m%(levelname)s\033[0m - \033[96m%(name)s\033[0m - \033[93m%(message)s\033[0m"
))
LOGGER.addHandler(stream_handler)


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
        self.embed_dim: int = kwargs.get("embed_dim", 512)
        self.n_blocks: int = kwargs.get("n_blocks", 6)
        self.vocab_size: int = kwargs.get("vocab_size", 25000)
        self.ff_dim: int = kwargs.get("ff_dim", 2048)
        self.heads: int = kwargs.get("heads", 8)
        self.dropout: float = kwargs.get("dropout", 0.1)
        self.seq_len: int = kwargs.get("seq_len", 50)


class TrainingConfig(Config):
    def __init__(self, **kwargs):
        self.epochs: int = kwargs.get("epochs", 10)
        self.batch_size: int = kwargs.get("batch_size", 64)
        self.init_lr: float = kwargs.get("init_lr", 2e-04)
        self.tb_log_dir: str = kwargs.get("tb_log_dir", "logs")
        self.checkpoint: str = kwargs.get("checkpoint", "amharic-gpt")
        self.validation_samples: int = kwargs.get("validation_samples", 20)
        self.training_data: str = kwargs.get("training_data", os.path.join(WORKING_DIR, "data", "train_chunk_size_256.json"))
        self.validation_data: str = kwargs.get("validation_data", os.path.join(WORKING_DIR, "data", "validate_chunk_size_256.json"))
        self.testing_data: str = kwargs.get("testing_data", os.path.join(WORKING_DIR, "data", "test_chunk_size_256.json"))

DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()