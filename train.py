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
from dataset import StreamingTextDataset, TextDataset, FineTuningDataset, IDataset


def log_confidence_metrics(logits: torch.Tensor, targets: torch.Tensor, tb_logger: TensorboardLogger, global_step: int, log_interval: int=2000):
    if global_step % log_interval != 0:
        # Ensure inputs are on CPU and detached for logging
        logits = logits.cpu().detach()
        targets = targets.cpu().detach()

        # Compute softmax probabilities
        probs = torch.softmax(logits, dim=-1)

        # 1. Top-1 Probability (confidence in the most likely token)
        top1_probs = probs.max(dim=-1)[0].mean().item()
        tb_logger.log_scalar("Confidence/Top-1 Probability", top1_probs, global_step)

        # 2. Top-5 Average Probability
        top5_probs = torch.topk(probs, k=5, dim=-1)[0].mean(dim=-1).flatten()
        tb_logger.log_histogram("Confidence/Top-5 Probability Distribution", top5_probs.numpy(), global_step)

        # 3. Entropy (measures uncertainty; lower is better)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
        tb_logger.log_scalar("Confidence/Entropy", entropy, global_step)

        # 4. Perplexity (based on cross-entropy loss)
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = nn.functional.nll_loss(log_probs.view(-1, log_probs.size(-1)), targets.view(-1), reduction='mean')
        perplexity = torch.exp(nll).item()
        tb_logger.log_scalar("Confidence/Perplexity", perplexity, global_step)


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

    # Learning rate scheduler with linear warmup for first warmup_steps
    # followed by inverse square root decay(scaled by the model's dimensionality)
    def lr_lambda(current_step: int):
        # Avoid division by zero
        current_step = max(1, current_step)
        
        # Scale factor
        scale = model.config.embed_dim ** -0.5
        
        # Linear warmup for first warmup_steps, then inverse square root decay
        lr = scale * min(
            current_step ** -0.5,
            current_step * (config.warmup_steps ** -1.5)
        )
        return lr / config.init_lr
    
    scaler = torch.GradScaler(device=DEVICE.type) if MIXED_PRECISION_ENABLED else None

    early_stopping = EarlyStopping(patience=config.es_patience, min_delta=config.es_min_delta)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.init_lr,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon
    )
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

            # (1, 1, SEQ_LEN, SEQ_LEN)
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
                    
                    # Log the confidence metrics                        
                    log_confidence_metrics(logits, label, tb_logger, global_step)
                    
                    if global_step and global_step % config.save_every == 0:
                        # Delete the oldest checkpoint(only keeps the last 'config.max_checkpoints_to_keep' checkpoints)
                        if global_step > config.max_checkpoints_to_keep * config.save_every:
                            os.remove(
                                config.checkpoint.replace(".pt", f"-{(global_step - config.max_checkpoints_to_keep * config.save_every) // 1000 }K_steps.pt")
                            )
                        torch.save(
                            {
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
                            },
                            os.path.join(
                                config.checkpoint.replace(".pt", f"-{global_step // 1000}K_steps.pt")
                            )
                        )

                global_step += 1

    if is_distributed:
        dist.barrier()
    tb_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument("--is-distributed", action="store_true", help="Device to train the model on")
    parser.add_argument("--training-data", type=str, default=DEFAULT_TRAINING_CONFIG.training_data, help="Path to the training dataset")
    parser.add_argument("--validation-data", type=str, default=DEFAULT_TRAINING_CONFIG.validation_data, help="Path to the validation dataset")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAINING_CONFIG.batch_size, help="Batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=DEFAULT_TRAINING_CONFIG.grad_accum_steps, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_TRAINING_CONFIG.warmup_steps, help="Number of warmup steps")
    parser.add_argument("--save-every", type=int, default=DEFAULT_TRAINING_CONFIG.save_every, help="Number of weight updates between checkpoints")
    parser.add_argument("--validate-every", type=int, default=DEFAULT_TRAINING_CONFIG.validate_every, help="Number of weight updates between validations")
    parser.add_argument("--init-lr", type=float, default=DEFAULT_TRAINING_CONFIG.init_lr, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_TRAINING_CONFIG.weight_decay, help="L2 regularization coefficient")
    parser.add_argument("--beta1", type=float, default=DEFAULT_TRAINING_CONFIG.beta1, help="Adam optimizer beta1")
    parser.add_argument("--beta2", type=float, default=DEFAULT_TRAINING_CONFIG.beta2, help="Adam optimizer beta2")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_TRAINING_CONFIG.epsilon, help="Adam optimizer epsilon")
    parser.add_argument("--max-norm", type=float, default=DEFAULT_TRAINING_CONFIG.max_norm, help="Gradient clipping threshold")
    parser.add_argument("--ema-alpha", type=float, default=DEFAULT_TRAINING_CONFIG.ema_alpha, help="Exponential moving average parameter")
    parser.add_argument("--label-smoothing", type=float, default=DEFAULT_TRAINING_CONFIG.label_smoothing, help="Label smoothing factor")
    parser.add_argument("--es-patience", type=int, default=DEFAULT_TRAINING_CONFIG.es_patience, help="Early stopping patience(number of steps)")
    parser.add_argument("--es-min-delta", type=float, default=DEFAULT_TRAINING_CONFIG.es_min_delta, help="Early stopping min delta")
    parser.add_argument("--tb-log-dir", type=str, default=DEFAULT_TRAINING_CONFIG.tb_log_dir, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_CONFIG.epochs, help="Number of epochs to train the model")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_MODEL_CONFIG.seq_len, help="Sequence length of the input")
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_MODEL_CONFIG.embed_dim, help="Dimensionality of the model")
    parser.add_argument("--n-blocks", type=int, default=DEFAULT_MODEL_CONFIG.n_blocks, help="Number of decoder blocks")
    parser.add_argument("--heads", type=int, default=DEFAULT_MODEL_CONFIG.heads, help="Number of attention heads")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_MODEL_CONFIG.vocab_size, help="Vocabulary size to use")
    parser.add_argument("--dropout", type=float, default=DEFAULT_MODEL_CONFIG.dropout, help="Dropout probability")
    parser.add_argument("--ff-dim", type=int, default=DEFAULT_MODEL_CONFIG.ff_dim, help="Dimensionality of the feed forward layer")
    parser.add_argument("--dist-backend", type=str, default="nccl", help="Distributed backend")
    parser.add_argument("--tokenizer", type=str, required=True, help="The path to the trained tokenizer model")
    parser.add_argument("--resume", default=False, action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--finetune", default=False, action="store_true", help="Finetune the model on conversational dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--max-checkpoints-to-keep", type=int, help="Maximum number of checkpoints to keep")
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
    if args.resume or args.finetune:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"File {args.checkpoint} does not exist")
        LOGGER.info(f"Loading checkpoint from '{args.checkpoint}'...")
        state: dict = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        weights = state["weights"]
        model_config: ModelConfig = state["model_config"]
        if args.resume:
            training_config: TrainingConfig = state["training_config"]
            training_config.update(**args.__dict__)
        state.pop("weights")        
        if args.finetune and not args.resume:
            state = {}
    
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.LoadFromFile(args.tokenizer)
    
    if args.finetune:
        train_dataset = FineTuningDataset(training_config.training_data, tokenizer, model_config.seq_len)
        val_dataset = FineTuningDataset(training_config.validation_data, tokenizer, model_config.seq_len)
        samples = len(train_dataset)    
    elif os.path.getsize(training_config.training_data) > 200 * 1024 * 1024:
        if GLOBAL_RANK == COORDINATOR_RANK:
            LOGGER.info(f"File '{os.path.basename(training_config.training_data)}' too large! streaming file...")
        train_dataset = StreamingTextDataset(training_config.training_data, tokenizer, model_config.seq_len)
        val_dataset = TextDataset(training_config.validation_data, tokenizer, model_config.seq_len)
        samples = get_line_count(training_config.training_data)
    else:
        train_dataset = TextDataset(training_config.training_data, tokenizer, model_config.seq_len)
        val_dataset = TextDataset(training_config.validation_data, tokenizer, model_config.seq_len)
        samples = len(train_dataset)
    
    training_config.samples_per_epoch = math.ceil(samples / (training_config.batch_size * WORLD_SIZE))
    training_config.updates_per_epoch = math.ceil(training_config.samples_per_epoch / training_config.grad_accum_steps)
    
    model = GPTmodel.build(model_config, weights).to(DEVICE)
    
    if GLOBAL_RANK == COORDINATOR_RANK:
        numerical_configs = {k: v for k, v in training_config.to_dict().items() if not isinstance(v, str)}
        LOGGER.info(f"Total training samples: {samples}")
        LOGGER.info(f"Using training config: {numerical_configs}")
        LOGGER.info(f"Initiating training with {'mixed-precision' if MIXED_PRECISION_ENABLED else 'single-precision'}...")
        LOGGER.info(f"Using model config: {model_config}")
        LOGGER.info(f"Model size: {sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2):.2f}MB")
        LOGGER.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train(training_config, model, train_dataset, val_dataset, args.is_distributed, state)

    if args.is_distributed:
        dist.destroy_process_group()