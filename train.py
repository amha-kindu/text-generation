import os
import math
import torch
import argparse
from config import *
import torch.nn as nn
from tqdm import tqdm
from model import GPTmodel
import sentencepiece as spm
from datetime import datetime
import torch.distributed as dist
from tensorboard_logger import TensorboardLogger
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from dataset import StreamingTextDataset, TextDataset, IDataset


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.counter = 0
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


@torch.no_grad()
def test(config: TrainingConfig, model: GPTmodel, test_dataset: IDataset, is_distributed: bool=False):
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

    val_loss = torch.tensor(0, dtype=torch.float32, device=DEVICE)
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

            loss: torch.Tensor = loss_func(
                # (N_BATCHES, SEQ_LEN, VOCAB_SIZE) --> (N_BATCHES * SEQ_LEN, VOCAB_SIZE)
                logits.view(-1, model.config.vocab_size),

                # (N_BATCHES, SEQ_LEN) --> (N_BATCHES * SEQ_LEN, )
                label.view(-1)
            ) 

        val_loss += loss

    return val_loss / len(val_batch_iterator)


def train(config: TrainingConfig, model: GPTmodel, train_dataset: IDataset, val_dataset: IDataset, is_distributed: bool = False, state: dict = {}) -> None:
    tb_logger = TensorboardLogger(config.tb_log_dir, is_distributed, GLOBAL_RANK, COORDINATOR_RANK)

    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[LOCAL_RANK])

    # Learning rate scheduler with warmup and cosine decay
    def lr_lambda(current_step: int):
        min_ratio = config.final_lr / config.init_lr
        if current_step < config.warmup_steps:
            return float(current_step) / float(max(1, config.warmup_steps))
        
        progress = (current_step - config.warmup_steps) / float(max(1, config.updates_per_epoch * config.epochs - config.warmup_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

        return min_ratio + (1.0 - min_ratio) * cosine_decay

    
    scaler = torch.GradScaler(device=DEVICE.type) if MIXED_PRECISION_ENABLED else None

    early_stopping = EarlyStopping(patience=config.es_patience, min_delta=config.es_min_delta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    accum_loss = 0
    global_step = 0
    initial_epoch = 0
    training_loss = 0
    validation_loss = 0
    if state:
        training_state = state["training_state"]
        global_step = training_state["global_step"] + 1
        initial_epoch = math.floor(training_state["global_step"] / config.updates_per_epoch)
        training_loss = training_state["training_loss"]
        validation_loss = training_state["validation_loss"]
        early_stopping.best_loss = training_state["best_val_loss"]
        optimizer.load_state_dict(training_state["optimizer_state"])
        scheduler.load_state_dict(training_state["lr_scheduler_state"])

    loss_func = nn.CrossEntropyLoss(ignore_index=train_dataset.tokenizer.pad_id(), label_smoothing=config.label_smoothing).to(DEVICE)
    batch_iterator = train_dataset.batch_iterator(config.batch_size)

    val_sampler = RandomSampler(val_dataset, replacement=True, num_samples=config.validation_samples)
    val_batch_iterator = val_dataset.batch_iterator(config.batch_size, sampler=val_sampler)

    for epoch in range(initial_epoch, config.epochs):
        batch_iterator = tqdm(batch_iterator, desc=f"\033[95m{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]}\033[0m - \033[94mINFO\033[0m - \033[96m{LOGGER.name}\033[0m - \033[93mEpoch {epoch+1}/{config.epochs}", disable = GLOBAL_RANK != COORDINATOR_RANK, total=config.samples_per_epoch)
        for i, batch in enumerate(batch_iterator):
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
                batch_loss: torch.Tensor = loss_func(
                    # (N_BATCHES, SEQ_LEN, VOCAB_SIZE) --> (N_BATCHES * SEQ_LEN, VOCAB_SIZE)
                    logits.view(-1, train_dataset.tokenizer.vocab_size()),

                    # (N_BATCHES, SEQ_LEN) --> (N_BATCHES * SEQ_LEN, )
                    label.view(-1)
                )
            
            loss_tensor = batch_loss.detach().clone()
            if is_distributed:
                torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            accum_loss += loss_tensor.item() / WORLD_SIZE

            accum_grad = (i + 1) % config.grad_accum_steps != 0 and i + 1 != config.samples_per_epoch

            if MIXED_PRECISION_ENABLED:
                scaler.scale(batch_loss).backward()
                if not accum_grad:
                    scaler.unscale_(optimizer)
                    tb_logger.log_gradients(model.parameters(), global_step)
                    tb_logger.log_named_gradients(model.named_parameters(), global_step)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                batch_loss.backward()
                if not accum_grad:
                    tb_logger.log_gradients(model.parameters(), global_step)
                    tb_logger.log_named_gradients(model.named_parameters(), global_step)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            if not accum_grad:
                accum_steps = config.grad_accum_steps if (i + 1) % config.grad_accum_steps == 0 else (i + 1) % config.grad_accum_steps
                if training_loss == 0:
                    training_loss = (accum_loss / accum_steps)
                else:
                    training_loss = config.ema_alpha * training_loss + (1 - config.ema_alpha) * (accum_loss / accum_steps)
                accum_loss = 0.0
                
                if global_step % config.validate_every == 0:
                    val_loss = validate(model.module if is_distributed else model, val_batch_iterator, loss_func)
                    if is_distributed:
                        torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)
                    avg_val_loss = val_loss.item() / WORLD_SIZE
                    
                    if validation_loss == 0:
                        validation_loss = avg_val_loss
                    else:
                        validation_loss = config.ema_alpha * validation_loss + (1 - config.ema_alpha) * avg_val_loss
                    
                    if early_stopping(validation_loss):
                        LOGGER.info(f"Early stopping triggered at epoch {epoch + 1}; avg val loss {early_stopping.best_loss:.4f} did not decrease significantly for {early_stopping.patience} consecutive weight updates")
                        if is_distributed:
                            dist.barrier()
                            dist.destroy_process_group()
                            exit(-1)
                        else:
                            break                        
                
                if GLOBAL_RANK == COORDINATOR_RANK:
                    if global_step % config.validate_every == 0:
                        tb_logger.log_scalars("Loss", {"Validation": validation_loss}, global_step)
                        tb_logger.log_scalar('Loss Gap', validation_loss - training_loss, global_step)
                    
                    tb_logger.log_scalars("Loss", {"Training": training_loss}, global_step)
                    tb_logger.log_scalar("Learning Rate", scheduler.get_last_lr()[0], global_step)
                    
                    batch_iterator.set_postfix({
                        "train_loss": f"{training_loss:6.3f}",
                        "val_loss": f"{validation_loss:6.3f}"
                    })
                    
                    if global_step % 2000 == 0:
                        top5_avg_probs = torch.topk(torch.softmax(logits, dim=-1), k=5, dim=-1)[0].mean(dim=-1).flatten().cpu().detach()
                        tb_logger.log_histogram("Top-5 Prediction Confidence Distribution", top5_avg_probs.numpy(), global_step)
                    
                    if global_step and global_step % config.save_every == 0:
                        torch.save({
                            "weights": model.module.state_dict() if is_distributed else model.state_dict(),
                            "model_config": model.module.config if is_distributed else model.config,
                            "training_state": {
                                "epoch": epoch,
                                "global_step": global_step,
                                "training_loss": training_loss,
                                "validation_loss": validation_loss,
                                "best_val_loss": early_stopping.best_loss,
                                "optimizer_state": optimizer.state_dict(),
                                "lr_scheduler_state": scheduler.state_dict()
                            },
                            "training_config": config
                        }, os.path.join(config.checkpoint))

                global_step += 1

    if is_distributed:
        dist.barrier()
    tb_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument("--is-distributed", action="store_true", help="Device to train the model on")
    parser.add_argument("--training-data", type=str, help="Path to the training dataset")
    parser.add_argument("--validation-data", type=str, help="Path to the validation dataset")
    parser.add_argument("--testing-data", type=str, help="Path to the testing dataset")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--grad-accum-steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, help="Number of warmup steps")
    parser.add_argument("--save-every", type=int, help="Number of weight updates between checkpoints")
    parser.add_argument("--validate-every", type=int, help="Number of weight updates between validations")
    parser.add_argument("--init-lr", type=float, help="Initial learning rate")
    parser.add_argument("--final-lr", type=float, help="Final learning rate")
    parser.add_argument("--weight-decay", type=float, help="L2 regularization coefficient")
    parser.add_argument("--max-norm", type=float, help="Gradient clipping threshold")
    parser.add_argument("--ema-alpha", type=float, help="Exponential moving average parameter")
    parser.add_argument("--label-smoothing", type=float, help="Label smoothing factor")
    parser.add_argument("--es-patience", type=int, help="Early stopping patience(number of steps)")
    parser.add_argument("--es-min-delta", type=float, help="Early stopping min delta")
    parser.add_argument("--tb-log-dir", type=str, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train the model")
    parser.add_argument("--seq-len", type=int, help="Sequence length of the input")
    parser.add_argument("--embed-dim", type=int, help="Dimensionality of the model")
    parser.add_argument("--n-blocks", type=int, help="Number of decoder blocks")
    parser.add_argument("--heads", type=int, help="Number of attention heads")
    parser.add_argument("--vocab-size", type=int, help="Vocabulary size to use")
    parser.add_argument("--dropout", type=float, help="Dropout probability")
    parser.add_argument("--ff-dim", type=int, help="Dimensionality of the feed forward layer")
    parser.add_argument("--dist-backend", type=str, default="nccl", help="Distributed backend")
    parser.add_argument("--tokenizer", type=str, required=True, help="The path to the trained tokenizer model")
    parser.add_argument("--resume", default=False, action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--validation-samples", type=int, help="Number of samples to use for a single validation run")

    args = parser.parse_args()
    
    if args.is_distributed:
        assert torch.cuda.device_count() > 1, "Must have more than one CUDA supporting GPUs to initiate distributed training"
        assert args.dist_backend in ["nccl", "gloo" "mpi", "ucc"], "Distributed backend must be one of the following: nccl, gloo, mpi or ucc"

        dist.init_process_group(backend=args.dist_backend)

    if args.embed_dim and args.heads:
        assert args.embed_dim % args.heads == 0, "embed_dim must be divisible by heads"

    training_config = TrainingConfig(**args.__dict__)
    model_config = ModelConfig(**args.__dict__)

    state, weights = {}, {}
    if args.resume and args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"File {args.checkpoint} does not exist")

        LOGGER.info(f"Loading checkpoint from '{args.checkpoint}'...")
        state: dict = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        weights = state["weights"]
        model_config: ModelConfig = state["model_config"]
        training_config: TrainingConfig = state["training_config"]
        training_config.update(**args.__dict__)
        state.pop("weights")
    
    samples = get_line_count(training_config.training_data)
    training_config.samples_per_epoch = math.ceil(samples / (training_config.batch_size * WORLD_SIZE))
    training_config.updates_per_epoch = math.ceil(training_config.samples_per_epoch / training_config.grad_accum_steps)
    if GLOBAL_RANK == COORDINATOR_RANK:
        numerical_configs = {k: v for k, v in training_config.to_dict().items() if not isinstance(v, str)}
        LOGGER.info(f"Total training samples: {samples}")
        LOGGER.info(f"Using training config: {numerical_configs}")
        LOGGER.info(f"Using model config: {model_config}")
        
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.LoadFromFile(args.tokenizer)
       
    if os.path.getsize(training_config.training_data) > 200 * 1024 * 1024:
        if GLOBAL_RANK == COORDINATOR_RANK:
            LOGGER.info(f"File '{os.path.basename(training_config.training_data)}' too large! streaming file...")
        train_dataset = StreamingTextDataset(training_config.training_data, tokenizer, model_config.seq_len)
    else:
        train_dataset = TextDataset(training_config.training_data, tokenizer, model_config.seq_len)
    
    if os.path.getsize(training_config.testing_data) > 200 * 1024 * 1024:
        if GLOBAL_RANK == COORDINATOR_RANK:
            LOGGER.info(f"File '{os.path.basename(training_config.testing_data)}' too large! streaming file...")
        test_dataset = StreamingTextDataset(training_config.testing_data, tokenizer, model_config.seq_len)
    else:
        test_dataset = TextDataset(training_config.testing_data, tokenizer, model_config.seq_len)

    val_dataset = TextDataset(training_config.validation_data, tokenizer, model_config.seq_len)

    model = GPTmodel.build(model_config, weights).to(DEVICE)
    
    if GLOBAL_RANK == COORDINATOR_RANK:
        LOGGER.info(f"Initiating training with {'mixed-precision' if MIXED_PRECISION_ENABLED else 'single-precision'}...")
        LOGGER.info(f"Model size: {sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2):.2f}MB")
        LOGGER.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train(training_config, model, train_dataset, val_dataset, args.is_distributed, state)

    test(training_config, model, test_dataset, args.is_distributed)

    if args.is_distributed:
        dist.destroy_process_group()