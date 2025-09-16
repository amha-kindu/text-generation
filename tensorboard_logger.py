from torch.utils.tensorboard import SummaryWriter

from config import GLOBAL_RANK, COORDINATOR_RANK


class TensorboardLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir) if self.is_coordinator() else None

    def is_coordinator(self):
        return GLOBAL_RANK == COORDINATOR_RANK

    def log_scalar(self, tag, value, step):
        if self.is_coordinator():
            self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, value, step):
        if self.is_coordinator():
            self.writer.add_histogram(tag, value, step)

    def log_scalars(self, tag, value_dict, step):
        if self.is_coordinator():
            self.writer.add_scalars(tag, value_dict, step)

    def log_text(self, tag, text, step):
        if self.is_coordinator():
            self.writer.add_text(tag, text, step)

    def close(self):
        if self.is_coordinator():
            self.writer.close()
