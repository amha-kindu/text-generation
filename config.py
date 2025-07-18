import os
import sys
import time
import torch
import numpy
import random
import logging
import tempfile

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
CHECKPOINTS_DIR = os.path.join(WORKING_DIR, "checkpoints")

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
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)

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
    lock_path = cache_file_path + ".lock"
    
    while True:
        if os.path.exists(cache_file_path):
            if GLOBAL_RANK == COORDINATOR_RANK:
                LOGGER.info("Using cached sample count")
            with open(cache_file_path, 'r') as f:
                val = f.read().strip()
                if val:
                    return int(val)
            # If empty, continue to compute
        
        # Try to acquire lock to compute
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                # Check again after acquiring lock
                if os.path.exists(cache_file_path):
                    with open(cache_file_path, 'r') as f:
                        val = f.read().strip()
                        if val:
                            return int(val)
                
                # Compute line count
                if GLOBAL_RANK == COORDINATOR_RANK:
                    LOGGER.info("Counting samples in training data...")
                with open(file_path, 'r') as src_file:
                    count = sum(1 for _ in src_file)
                
                # Write to temp file and rename atomically
                dir_name = os.path.dirname(cache_file_path)
                temp_fd, temp_path = tempfile.mkstemp(prefix='cache.', suffix='.tmp', dir=dir_name)
                try:
                    with os.fdopen(temp_fd, 'w') as temp_f:
                        temp_f.write(str(count))
                    os.rename(temp_path, cache_file_path)
                except:
                    os.unlink(temp_path)
                    raise
                
                return count
            finally:
                os.close(fd)
                os.unlink(lock_path)
        except FileExistsError:
            time.sleep(0.1)


class TrainingConfig(Config):
    def __init__(self, **kwargs):
        kwargs = { k: v for k, v in kwargs.items() if v is not None}
        self.samples_per_epoch = None
        self.updates_per_epoch = None
        self.epochs: int = kwargs.get("epochs", 10)
        self.batch_size: int = kwargs.get("batch_size", 64)
        self.grad_accum_steps: int = kwargs.get("grad_accum_steps", 1)
        self.init_lr: float = kwargs.get("init_lr", 2e-04)
        self.final_lr: float = kwargs.get("final_lr", 0)
        self.weight_decay: float = kwargs.get("weight_decay", 0.01)
        self.beta1: float = kwargs.get("beta1", 0.9)
        self.beta2: float = kwargs.get("beta2", 0.999)
        self.epsilon: float = kwargs.get("epsilon", 1e-08)
        self.max_norm: float = kwargs.get("max_norm", 1.0)
        self.ema_alpha: float = kwargs.get("ema_alpha", 0.9)
        self.label_smoothing: float = kwargs.get("label_smoothing", 0)
        self.es_min_delta: float = kwargs.get("es_min_delta", 0.01)
        self.es_patience: float = kwargs.get("es_patience", 10000)
        self.tb_log_dir: str = kwargs.get("tb_log_dir", "logs")
        self.checkpoint: str = kwargs.get("checkpoint", "amharic-gpt")
        self.max_checkpoints_to_keep: int = kwargs.get("max_checkpoints_to_keep", 5)
        self.warmup_steps: int = kwargs.get("warmup_steps", 1000)
        self.save_every: int = kwargs.get("save_every", 1000)
        self.validate_every: int = kwargs.get("validate_every", 100)
        self.validation_samples: int = kwargs.get("validation_samples", 20)
        self.training_data: str = kwargs.get("training_data", None)
        self.validation_data: str = kwargs.get("validation_data", None)
        
        if self.training_data and not os.path.isfile(self.training_data):
            raise FileNotFoundError(f"File '{self.training_data}' does not exist")
        
        if self.validation_data and not os.path.isfile(self.validation_data):
            raise FileNotFoundError(f"File '{self.validation_data}' does not exist")


DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Check a model training checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint) and not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"File {args.checkpoint} does not exist")
    
    LOGGER.info(f"Loading checkpoint from '{args.checkpoint}'...")
    checkpoint: dict = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)

    model_config: ModelConfig = checkpoint["model_config"]
    training_config: TrainingConfig = checkpoint["training_config"]
    training_state: dict = checkpoint["training_state"]
    
    training_state = {k: v for k, v in training_state.items() if k not in ["optimizer_state", "lr_scheduler_state"]}
    
    LOGGER.info(f"Model config: {model_config}")
    LOGGER.info(f"Training config: {training_config}")
    LOGGER.info(f"Training state: {training_state}")