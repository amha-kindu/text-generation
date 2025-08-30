import torch
import queue
import threading
from torch.utils.tensorboard import SummaryWriter

from config import LOGGER, GLOBAL_RANK, COORDINATOR_RANK


class TensorboardLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir) if self.is_coordinator() else None
        self.queue = queue.Queue()
        if self.is_coordinator():
            self._stop_signal = threading.Event()
            self.thread = threading.Thread(target=self._write_loop)
            self.thread.start()

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

    def log_named_gradients(self, named_params, step):
        if self.is_coordinator():
            for name, param in named_params:
                if param.grad is not None:
                    grad_norm = torch.linalg.vector_norm(param.grad.view(-1)).item()
                    self.queue.put((f"GradNorm/{name}", grad_norm, step))

    def log_gradients(self, params, step):
        if self.is_coordinator():
            grad_vector = torch.cat([p.grad.view(-1) for p in params if p.grad is not None])
            grad_norm = torch.linalg.vector_norm(grad_vector).item()
            self.queue.put(("GradNorm/Overall", grad_norm, step))

    def _write_loop(self):
        while not self._stop_signal.is_set():
            try:
                name, value, step = self.queue.get(timeout=1)
                self.writer.add_scalar(name, value, step)
            except queue.Empty:
                continue
            except Exception as e:
                LOGGER.error(f"Exception in TensorboardLogger: {e}")

    def close(self):
        if self.is_coordinator():
            self._stop_signal.set()
            self.thread.join()
            self.writer.close()
