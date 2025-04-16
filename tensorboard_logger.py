import threading
from queue import Queue, Empty
from torch.utils.tensorboard import SummaryWriter
import torch

class TensorboardLogger:
    def __init__(self, log_dir: str, is_distributed: bool, global_rank: int, coordinator_rank: int = 0):
        self.queue = Queue()
        self.writer = SummaryWriter(log_dir) if global_rank == coordinator_rank else None
        self.global_rank = global_rank
        self.coordinator_rank = coordinator_rank
        self.is_distributed = is_distributed
        self._stop_signal = threading.Event()
        if self.is_main():
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def is_main(self):
        return not self.is_distributed or self.global_rank == self.coordinator_rank

    def _run(self):
        while not self._stop_signal.is_set():
            try:
                item = self.queue.get(timeout=1)
                if item is None:
                    continue
                self._process(item)
            except Empty:
                continue

    def _process(self, item):
        tag, value, step, mode = item
        if mode == 'scalar':
            self.writer.add_scalar(tag, value, step)
        elif mode == 'histogram':
            self.writer.add_histogram(tag, value, step)
        elif mode == 'scalars':
            self.writer.add_scalars(tag, value, step)
        elif mode == 'text':
            self.writer.add_text(tag, value, step)
        elif mode == 'figure':
            self.writer.add_figure(tag, value, step)
        self.writer.flush()

    def log_scalar(self, tag, value, step):
        if self.is_main():
            self.queue.put((tag, value, step, 'scalar'))

    def log_histogram(self, tag, value, step):
        if self.is_main():
            self.queue.put((tag, value, step, 'histogram'))

    def log_scalars(self, tag, value_dict, step):
        if self.is_main():
            self.queue.put((tag, value_dict, step, 'scalars'))

    def log_text(self, tag, text, step):
        if self.is_main():
            self.queue.put((tag, text, step, 'text'))

    def log_figure(self, tag, fig, step):
        if self.is_main():
            self.queue.put((tag, fig, step, 'figure'))
    
    def log_gradients(self, named_params, step):
        grad_vector = None
        for name, param in named_params:
            if param.grad is not None:
                grad = param.grad.detach()
                grad_norm = grad.norm().item()
                grad_vector = torch.cat([grad.view(-1)])
                self.log_scalar(f"GradNorm/{name}", grad_norm, step)
                self.log_histogram(f"GradHistogram/{name}", grad, step)
        grad_norm_pre_clip = torch.linalg.vector_norm(grad_vector).item()
        self.log_scalar("GradNorm/Overall", grad_norm_pre_clip, step)

    def close(self):
        if self.is_main():
            self._stop_signal.set()
            self.thread.join()
            self.writer.close()
