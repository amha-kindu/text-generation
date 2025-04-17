import fcntl
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
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
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


def get_line_count(file_path):
    cache_file_path = f"{file_path}.lc"
    
    # Try to read from cache (with shared lock)
    if os.path.exists(cache_file_path):
        if GLOBAL_RANK == COORDINATOR_RANK:
            LOGGER.info("Using cached sample count")
        with open(cache_file_path, 'r') as f:
            fcntl.flock(f, fcntl.LOCK_SH)  # Shared lock for reading
            try:
                val = f.read()
                if val:
                    return int(val)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    # Compute line count and write to cache (with exclusive lock)
    with open(cache_file_path, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock for writing
        try:
            # Compute actual line count
            if GLOBAL_RANK == COORDINATOR_RANK:
                LOGGER.info("Counting samples in training data...")
            with open(file_path, 'r') as src_file:
                count = sum(1 for _ in src_file)
            
            # Write to cache
            with open(cache_file_path, 'w') as cache_f:
                cache_f.write(str(count))
            
            return count
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

class TrainingConfig(Config):
    def __init__(self, **kwargs):
        self.epochs: int = kwargs.get("epochs", 10)
        self.batch_size: int = kwargs.get("batch_size", 64)
        self.grad_accum_steps: int = kwargs.get("grad_accum_steps", 1)
        self.init_lr: float = kwargs.get("init_lr", 2e-04)
        self.weight_decay: float = kwargs.get("weight_decay", 0.01)
        self.max_norm: float = kwargs.get("max_norm", 1.0)
        self.ema_alpha: float = kwargs.get("ema_alpha", 0.9)
        self.label_smoothing: float = kwargs.get("label_smoothing", 0.1)
        self.es_min_delta: float = kwargs.get("es_min_delta", 0.01)
        self.es_patience: float = kwargs.get("es_patience", 10000)
        self.tb_log_dir: str = kwargs.get("tb_log_dir", "logs")
        self.checkpoint: str = kwargs.get("checkpoint", "amharic-gpt")
        self.save_every: int = kwargs.get("save_every", 1000)
        self.validate_every: int = kwargs.get("validate_every", 100)
        self.validation_samples: int = kwargs.get("validation_samples", 20)
        self.training_data: str = kwargs.get("training_data", os.path.join(WORKING_DIR, "pretraining-corpus", "train.jsonl"))
        self.validation_data: str = kwargs.get("validation_data", os.path.join(WORKING_DIR, "pretraining-corpus", "val.jsonl"))
        self.testing_data: str = kwargs.get("testing_data", os.path.join(WORKING_DIR, "pretraining-corpus", "test.jsonl"))
        
        if not os.path.isfile(self.training_data):
            raise FileNotFoundError(f"File '{self.training_data}' does not exist")

        if kwargs:
            samples = get_line_count(self.training_data)
            self.samples_per_epoch = samples // (self.batch_size * WORLD_SIZE)
            self.updates_per_epoch = samples // (self.batch_size * self.grad_accum_steps * WORLD_SIZE)
            # Set warmup steps to 1% of the steps per epoch
            self.warmup_steps = int(0.01 * self.updates_per_epoch)
            if GLOBAL_RANK == COORDINATOR_RANK:
                numerical_configs = {k: v for k, v in self.to_dict().items() if not isinstance(v, str)}
                LOGGER.info(f"Total training samples: {samples}")
                LOGGER.info(f"Using training config: {numerical_configs}")
        
        if not os.path.isfile(self.validation_data):
            raise FileNotFoundError(f"File '{self.validation_data}' does not exist")

        if not os.path.isfile(self.testing_data):
            raise FileNotFoundError(f"File '{self.testing_data}' does not exist")

DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()