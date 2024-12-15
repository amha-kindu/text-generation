import os
import torch
import argparse
from config import *
import torch.nn as nn
from tqdm import tqdm
from model import GPTmodel
from datetime import datetime
from dataset import TextDataset
import torch.distributed as dist
from tokenizer import SentencePieceProcessor
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


@torch.no_grad()
def test(config: TrainingConfig, model: GPTmodel, test_dataset: TextDataset, is_distributed: bool=False):
    model.eval()

    loss_func = nn.CrossEntropyLoss(ignore_index=test_dataset.tokenizer.pad_id(), label_smoothing=0.1).to(DEVICE)

    sampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(), rank=LOCAL_RANK) if is_distributed else None
    batch_iterator = tqdm(test_dataset.batch_iterator(config.batch_size, sampler=sampler), desc=f"\033[95m{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]}\033[0m - \033[94mINFO\033[0m - \033[96m{LOGGER.name}\033[0m - \033[93mModel Evaluation", disable = GLOBAL_RANK != COORDINATOR_RANK)

    evaluation_loss = 0
    for index, batch in enumerate(batch_iterator):
        # (N_BATCHES, SEQ_LEN)
        decoder_input = batch["decoder_input"].to(DEVICE)

        # (1, SEQ_LEN, SEQ_LEN)
        decoder_mask = batch["decoder_mask"].to(DEVICE)

        # (N_BATCHES, SEQ_LEN)
        label: torch.Tensor = batch['label'].to(DEVICE)

        with torch.autocast(DEVICE.type, enabled=MIXED_PRECISION_ENABLED):
            # (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
            logits: torch.Tensor = model(decoder_input, decoder_mask)

            # Compute the training loss
            test_loss: torch.Tensor = loss_func(
                # (N_BATCHES, SEQ_LEN, VOCAB_SIZE) --> (N_BATCHES * SEQ_LEN, VOCAB_SIZE)
                logits.view(-1, model.config.vocab_size),

                # (N_BATCHES, SEQ_LEN) --> (N_BATCHES * SEQ_LEN, )
                label.view(-1)
            )

        batch_iterator.set_postfix({"loss": f"{evaluation_loss / (index + 1):6.3f}"})

        evaluation_loss += test_loss.item()
    
    
@torch.no_grad()
def validate(model: GPTmodel, val_batch_iterator: DataLoader, loss_func: nn.CrossEntropyLoss):
    model.eval()

    val_loss = 0
    for batch in val_batch_iterator:
        # (N_BATCHES, SEQ_LEN)
        decoder_input = batch["decoder_input"].to(DEVICE)

        # (1, SEQ_LEN, SEQ_LEN)
        decoder_mask = batch["decoder_mask"].to(DEVICE)

        # (N_BATCHES, SEQ_LEN)
        label: torch.Tensor = batch['label'].to(DEVICE)

        with torch.autocast(DEVICE.type, enabled=MIXED_PRECISION_ENABLED):
            # (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
            logits: torch.Tensor = model(decoder_input, decoder_mask)

            # Compute the cross-entropy loss
            loss: torch.Tensor = loss_func(
                # (N_BATCHES, SEQ_LEN, VOCAB_SIZE) --> (N_BATCHES * SEQ_LEN, VOCAB_SIZE)
                logits.view(-1, model.config.vocab_size),

                # (N_BATCHES, SEQ_LEN) --> (N_BATCHES * SEQ_LEN, )
                label.view(-1)
            ) 
        val_loss += loss.item()

    return val_loss / len(val_batch_iterator)


def train(config: TrainingConfig, model: GPTmodel, train_dataset: TextDataset, val_dataset: TextDataset, is_distributed: bool = False, state: dict = {}) -> None:
    writer = SummaryWriter(config.tb_log_dir)

    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[LOCAL_RANK])
    
    scaler = torch.GradScaler(device=DEVICE.type) if MIXED_PRECISION_ENABLED else None

    early_stopping = EarlyStopping(patience=3, min_delta=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.init_lr, weight_decay=1e-2)
    
    initial_epoch = 0
    global_step = 0
    training_loss = 0
    validation_loss = 0
    if state:
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        training_loss = state["training_loss"]
        validation_loss = state["validation_loss"]
        optimizer.load_state_dict(state["optimizer_state_dict"])

    loss_func = nn.CrossEntropyLoss(ignore_index=train_dataset.tokenizer.pad_id(), label_smoothing=0.1).to(DEVICE)

    sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=LOCAL_RANK, shuffle=True) if is_distributed else None
    batch_iterator = train_dataset.batch_iterator(config.batch_size, sampler=sampler)

    val_sampler = RandomSampler(val_dataset, replacement=True, num_samples=config.validation_samples)
    val_batch_iterator = val_dataset.batch_iterator(config.batch_size, sampler=val_sampler)

    avg_train_loss, avg_val_loss = 0, 0
    for epoch in range(initial_epoch, config.epochs):
        if is_distributed:
            sampler.set_epoch(epoch)
        
        batch_iterator = tqdm(batch_iterator, desc=f"\033[95m{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]}\033[0m - \033[94mINFO\033[0m - \033[96m{LOGGER.name}\033[0m - \033[93mProcessing epoch {epoch: 02d}", disable = GLOBAL_RANK != COORDINATOR_RANK)
        for batch in batch_iterator:
            model.train() 
                 
            # (N_BATCHES, SEQ_LEN)
            decoder_input: torch.Tensor = batch["decoder_input"].to(DEVICE)

            # (1, SEQ_LEN, SEQ_LEN)
            decoder_mask: torch.Tensor = batch["decoder_mask"].to(DEVICE)
            
            # (N_BATCHES, SEQ_LEN)
            label: torch.Tensor = batch['label'].to(DEVICE)

            with torch.autocast(device_type=DEVICE.type, enabled=MIXED_PRECISION_ENABLED):
                # (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
                logits: torch.Tensor = model(decoder_input, decoder_mask)

                # Compute the cross-entropy loss
                batch_loss = loss_func.forward(
                    # (N_BATCHES, SEQ_LEN, VOCAB_SIZE) --> (N_BATCHES * SEQ_LEN, VOCAB_SIZE)
                    logits.view(-1, train_dataset.tokenizer.vocab_size()),

                    # (N_BATCHES, SEQ_LEN) --> (N_BATCHES * SEQ_LEN, )
                    label.view(-1)
                )
            
            training_loss += batch_loss.item()
            avg_train_loss = training_loss / (global_step + 1)

            if GLOBAL_RANK == COORDINATOR_RANK:
                if global_step % 200 == 0:
                    validation_loss += validate(model.module if is_distributed else model, val_batch_iterator, loss_func)
                    avg_val_loss = validation_loss / ((global_step + 1) // 200 + 1)

                    if early_stopping(avg_val_loss):
                        return
                    
                    writer.add_scalars(
                        "Loss", 
                        { 
                            "Validation": avg_val_loss
                        },
                        global_step
                    )
                writer.add_scalars(
                    "Loss",
                    {
                        "Training": avg_train_loss
                    },
                    global_step
                )
                writer.flush()

            batch_iterator.set_postfix({
                "train_loss": f"{avg_train_loss:6.3f}", 
                "val_loss": f"{avg_val_loss:6.3f}"
            })

            # Scale the loss for numerical stability if using mixed-precision and
            # Perform the backward pass on the computation graph built during the forward pass, 
            # in order to calculate the grad for each of the intermediate and leaf tensors on the computation graph
            if MIXED_PRECISION_ENABLED:
                scaler.scale(batch_loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)                
                optimizer.step()

            
            # Zero the gradients of the model parameters to prevent gradient accumulation 
            optimizer.zero_grad()

            global_step += 1
        
        if GLOBAL_RANK == COORDINATOR_RANK:        
            torch.save({
                "epoch": epoch,
                "batch_size": config.batch_size,
                "initial_learning_rate": config.init_lr,
                "model_state_dict": model.module.state_dict() if is_distributed else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "model_config": model.module.config if is_distributed else model.config
            }, os.path.join(WEIGHTS_DIRECTORY, f"{config.checkpoint}.pt"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument("--is-distributed", action="store_true", help="Device to train the model on")
    parser.add_argument("--training-data", type=str, default=DEFAULT_TRAINING_CONFIG.training_data, help="Path to the training dataset")
    parser.add_argument("--validation-data", type=str, default=DEFAULT_TRAINING_CONFIG.validation_data, help="Path to the validation dataset")
    parser.add_argument("--testing-data", type=str, default=DEFAULT_TRAINING_CONFIG.testing_data, help="Path to the testing dataset")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAINING_CONFIG.batch_size, help="Batch size used during training")
    parser.add_argument("--init-lr", type=float, default=DEFAULT_TRAINING_CONFIG.init_lr, help="Initial learning rate")
    parser.add_argument("--tb-log-dir", type=str, default=DEFAULT_TRAINING_CONFIG.tb_log_dir, help="Initial learning rate")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_TRAINING_CONFIG.checkpoint, help="Filename for the model checkpoint")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_CONFIG.epochs, help="Number of epochs to train the model")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_MODEL_CONFIG.seq_len, help="Sequence length of the input")
    parser.add_argument("--d-model", type=int, default=DEFAULT_MODEL_CONFIG.d_model, help="Dimensionality of the model")
    parser.add_argument("--n-blocks", type=int, default=DEFAULT_MODEL_CONFIG.n_blocks, help="Number of decoder blocks")
    parser.add_argument("--heads", type=int, default=DEFAULT_MODEL_CONFIG.heads, help="Number of attention heads")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_MODEL_CONFIG.vocab_size, help="Vocabulary size to use")
    parser.add_argument("--dropout", type=float, default=DEFAULT_MODEL_CONFIG.dropout, help="Dropout probability")
    parser.add_argument("--dff", type=int, default=DEFAULT_MODEL_CONFIG.dff, help="Dimensionality of the feed forward layer")
    parser.add_argument("--dist-backend", type=str, default="nccl", help="Distributed backend")
    parser.add_argument("--preload-weights", type=str, default="", help="File path to load saved weights")
    parser.add_argument("--validation-samples", type=int, default=DEFAULT_TRAINING_CONFIG.validation_samples, help="Number of samples to use for a single validation run")

    args = parser.parse_args()

    if args.is_distributed:
        assert torch.cuda.device_count() > 1, "Must have more than one CUDA supporting GPUs to initiate distributed training"
        assert args.dist_backend in ["nccl", "gloo" "mpi", "ucc"], "Distributed backend must be one of the following: nccl, gloo, mpi or ucc"

        dist.init_process_group(backend=args.dist_backend)

    assert args.d_model % args.heads == 0, "d_model must be divisible by heads"

    training_config = TrainingConfig(**args.__dict__)
    model_config = ModelConfig(**args.__dict__)

    os.makedirs(WEIGHTS_DIRECTORY, exist_ok=True)
    tokenizer = SentencePieceProcessor(max_len=model_config.seq_len)
    tokenizer.LoadFromFile(
        os.path.join(WORKING_DIR, os.path.join("tokenizers", f"amharic-bpe-tokenizer-{model_config.vocab_size // 1000}k.model"))
    )

    train_dataset = TextDataset(training_config.training_data, tokenizer)
    val_dataset = TextDataset(training_config.validation_data, tokenizer)
    test_dataset = TextDataset(training_config.testing_data, tokenizer)

    state, weights = {}, {}
    if args.preload_weights:
        if not os.path.exists(args.preload_weights):
            raise FileNotFoundError(f"File {args.preload_weights} does not exist")
        
        LOGGER.info(f"Preloading model weights {args.preload_weights}...")
        state: dict = torch.load(args.preload_weights, map_location=DEVICE)
        weights = state["model_state_dict"]
        state.pop("model_state_dict")

    model = GPTmodel.build(model_config, weights).to(DEVICE)
    
    if GLOBAL_RANK == COORDINATOR_RANK:
        LOGGER.info(f"Initiating training with {'mixed-precision' if MIXED_PRECISION_ENABLED else 'single-precision'}...")
        LOGGER.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train(training_config, model, train_dataset, val_dataset, args.is_distributed, state)

    test(training_config, model, test_dataset, args.is_distributed)

    if args.is_distributed:
        dist.destroy_process_group()