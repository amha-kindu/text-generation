import os
import json
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
import sentencepiece as spm
from datetime import datetime
import torch.distributed as dist
from torch.utils.data import DistributedSampler, RandomSampler
from torch.nn.parallel import DistributedDataParallel

from config import *
from model import GPTmodel
from tensorboard_logger import TensorboardLogger
from lr_schedulers import LRScheduler, get_lr_scheduler
from dataset import TextStreamDataset, TextDataset, NLPDataset
from utils import EarlyStopping, log_confidence_metrics, save_checkpoint, validate


def train(config: TrainingConfig, model: GPTmodel, train_dataset: NLPDataset, val_dataset: NLPDataset, is_distributed: bool = False, training_state: TrainingState | None = None) -> None:
    tb_logger = TensorboardLogger(config.tb_log_dir)
    
    tb_logger.log_text("TrainingConfig", f"```json\n{json.dumps(config.__dict__, indent=2)}\n```", step=0)
    tb_logger.log_text("ModelConfig", f"```json\n{json.dumps(model.config.__dict__, indent=2)}\n```", step=0)
    tb_logger.log_text("Environment", f"```json\n{json.dumps(ENV, indent=2)}\n```", step=0)

    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[LOCAL_RANK])
    
    scaler = torch.GradScaler(device=DEVICE.type) if MIXED_PRECISION_ENABLED else None

    early_stopping = EarlyStopping(patience=config.es_patience, min_delta=config.es_min_delta)

    optimizer = torch.optim.AdamW(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=config.init_lr,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon
    )
    
    scheduler = get_lr_scheduler(optimizer, config, model.config.embed_dim)
    
    accum_loss = 0
    global_step = 0
    initial_epoch = 0
    training_loss = 0
    validation_loss = 0
    if training_state:
        global_step = training_state.global_step + 1
        initial_epoch = int(training_state.global_step / config.steps_per_epoch)
        training_loss = training_state.training_loss
        validation_loss = training_state.validation_loss
        early_stopping.best_loss = training_state.best_val_loss
        optimizer.load_state_dict(training_state.optimizer_state)
        scheduler.load_state_dict(training_state.lr_scheduler_state)

    loss_func = nn.CrossEntropyLoss(ignore_index=train_dataset.ignore_index, label_smoothing=config.label_smoothing).to(DEVICE)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=WORLD_SIZE, rank=GLOBAL_RANK, shuffle=True, drop_last=True) if is_distributed else None
    data_loader = train_dataset.get_loader(config.batch_size, sampler=train_sampler)
    
    val_sampler = RandomSampler(val_dataset, replacement=True, num_samples=config.batch_size * int(config.vt_ratio * config.validate_every * config.grad_accum_steps))
    val_loader = val_dataset.get_loader(config.batch_size, sampler=val_sampler)

    for epoch in range(initial_epoch, config.epochs):
        data_loader = tqdm(data_loader, desc=f"\033[95m{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]}\033[0m - \033[94mINFO\033[0m - \033[96m{LOGGER.name}\033[0m - \033[93mEpoch {epoch+1}/{config.epochs}", disable = GLOBAL_RANK != COORDINATOR_RANK, total=config.batches_per_epoch)
        for i, batch in enumerate(data_loader):
            model.train()
            
            if is_distributed:
                train_sampler.set_epoch(epoch)
            
            # (N_BATCHES, SEQ_LEN)
            decoder_input: torch.Tensor = batch[0].to(DEVICE)
            label: torch.Tensor         = batch[1].to(DEVICE)
            
            # (N_BATCHES, SEQ_LEN, SEQ_LEN)
            decoder_mask: torch.Tensor  = batch[2].to(DEVICE)

            with torch.autocast(device_type=DEVICE.type, enabled=MIXED_PRECISION_ENABLED):
                # (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
                logits: torch.Tensor = model(decoder_input, decoder_mask)

                # Compute the cross-entropy loss
                batch_loss: torch.Tensor = loss_func(
                    # (N_BATCHES, SEQ_LEN, VOCAB_SIZE) --> (N_BATCHES * SEQ_LEN, VOCAB_SIZE)
                    logits.view(-1, model.config.vocab_size),

                    # (N_BATCHES, SEQ_LEN) --> (N_BATCHES * SEQ_LEN, )
                    label.view(-1)
                )
            
            loss_tensor = batch_loss.detach().clone()
            if is_distributed:
                torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            accum_loss += loss_tensor.item() / WORLD_SIZE

            accum_grad = (i + 1) % config.grad_accum_steps != 0 and i + 1 != config.batches_per_epoch

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
                    val_loss = validate(
                        model=model.module if is_distributed else model,
                        data_loader=val_loader,
                        loss_func=loss_func
                    )
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
                    
                    data_loader.set_postfix({
                        "train_loss": f"{training_loss:6.3f}",
                        "val_loss": f"{validation_loss:6.3f}"
                    })
                    
                    log_confidence_metrics(logits.detach(), label.detach(), tb_logger, global_step, train_dataset.ignore_index, config.steps_per_epoch // 2)
                    
                    if global_step and global_step % config.save_every == 0:
                        save_checkpoint(
                            model=model.module if is_distributed else model,
                            global_step=global_step,
                            config=config,
                            training_state=TrainingState(
                                epoch=epoch,
                                global_step=global_step,
                                training_loss=training_loss,
                                validation_loss=validation_loss,
                                best_val_loss=early_stopping.best_loss,
                                optimizer_state=optimizer.state_dict(),
                                lr_scheduler_state=scheduler.state_dict()
                            )
                        )

                global_step += 1

    if is_distributed:
        dist.barrier()
    tb_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument("--is-distributed", action="store_true", help="Device to train the model on")
    parser.add_argument("--training-data", required=True, type=str, help="Path to the training dataset")
    parser.add_argument("--validation-data", required=True, type=str, help="Path to the validation dataset")
    parser.add_argument("--tokenizer", type=str, required=True, help="The path to the trained tokenizer model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--init-weights", type=str, help="Path to checkpoint with weights for initialization")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--grad-accum-steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, help="Number of warmup steps")
    parser.add_argument("--save-every", type=int, help="Number of weight updates between checkpoints")
    parser.add_argument("--validate-every", type=int, help="Number of weight updates between validations")
    parser.add_argument("--init-lr", type=float, help="Initial learning rate")
    parser.add_argument("--min-lr", type=float, help="Minimum learning rate")
    parser.add_argument("--lr-scheduler", type=str, choices=[LRScheduler.WARMUP_CONSTANT.value, LRScheduler.WARMUP_LINEAR.value, LRScheduler.WARMUP_COSINE.value, LRScheduler.INVERSE_SQRT.value], help="Learning rate scheduler(default: warmup_linear)")
    parser.add_argument("--weight-decay", type=float, help="L2 regularization coefficient")
    parser.add_argument("--beta1", type=float, help="Adam optimizer beta1")
    parser.add_argument("--beta2", type=float, help="Adam optimizer beta2")
    parser.add_argument("--epsilon", type=float, help="Adam optimizer epsilon")
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
    parser.add_argument("--resume", default=False, action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--max-checkpoints-to-keep", type=int, help="Maximum number of checkpoints to keep")
    parser.add_argument("--dl-workers", type=int, help="Number of subprocesses to use for data loading")
    parser.add_argument("--stream", default=False, action="store_true", help="Stream data from disk")

    args = parser.parse_args()
    
    if args.is_distributed:
        assert torch.cuda.device_count() > 1, "Must have more than one CUDA supporting GPUs to initiate distributed training"
        assert args.dist_backend in ["nccl", "gloo" "mpi", "ucc"], "Distributed backend must be one of the following: nccl, gloo, mpi or ucc"

        dist.init_process_group(backend=args.dist_backend)

    if args.embed_dim and args.heads:
        assert args.embed_dim % args.heads == 0, "embed_dim must be divisible by heads"

    model_config: ModelConfig = DEFAULT_MODEL_CONFIG
    model_config.update(**args.__dict__)
    
    training_config: TrainingConfig = DEFAULT_TRAINING_CONFIG
    training_config.update(**args.__dict__)

    training_state, weights = None, {}
    if args.init_weights:
        if not os.path.exists(args.init_weights):
            raise FileNotFoundError(f"File {args.init_weights} does not exist")
        LOGGER.info(f"Loading initial weights from '{args.init_weights}'...")
        checkpoint = torch.load(args.init_weights, map_location=DEVICE, weights_only=True)
        weights = checkpoint['weights']
    elif args.resume:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"File {args.checkpoint} does not exist")
        LOGGER.info(f"Loading checkpoint from '{args.checkpoint}'...")
        checkpoint: dict = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        weights = checkpoint["weights"]
        
        model_config: ModelConfig = checkpoint["model_config"]
        model_config.update(dropout=args.dropout)
        
        training_config: TrainingConfig = checkpoint["training_config"]
        training_config.update(skip=['checkpoint', 'training_data', 'validation_data'], **args.__dict__)
        
        training_state = checkpoint["training_state"]
    
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.LoadFromFile(args.tokenizer)
    
    if args.stream or os.path.getsize(training_config.training_data) > 200 * 1024 * 1024:
        if GLOBAL_RANK == COORDINATOR_RANK:
            LOGGER.info(f"File '{os.path.basename(training_config.training_data)}' too large! streaming file...")
        train_dataset = TextStreamDataset(training_config.training_data, tokenizer, model_config.seq_len, training_config.workers)
    else:
        train_dataset = TextDataset(training_config.training_data, tokenizer, model_config.seq_len, training_config.workers)
            
    if args.stream or os.path.getsize(training_config.validation_data) > 200 * 1024 * 1024:
        if GLOBAL_RANK == COORDINATOR_RANK:
            LOGGER.info(f"File '{os.path.basename(training_config.validation_data)}' too large! streaming file...")
        val_dataset = TextStreamDataset(training_config.validation_data, tokenizer, model_config.seq_len)
    else:
        val_dataset = TextDataset(training_config.validation_data, tokenizer, model_config.seq_len)
    
    training_config.batches_per_epoch = int(len(train_dataset) / (training_config.batch_size * WORLD_SIZE))
    training_config.steps_per_epoch = int(training_config.batches_per_epoch / training_config.grad_accum_steps)
    
    model = GPTmodel.build(model_config, weights).to(DEVICE)
    
    if GLOBAL_RANK == COORDINATOR_RANK:
        numerical_configs = {k: v for k, v in training_config.to_dict().items() if not isinstance(v, str)}
        LOGGER.info(f"Total training samples: {len(train_dataset)}")
        if train_dataset.tokens > 0:
            LOGGER.info(f"Total training tokens: {train_dataset.tokens}")
            LOGGER.info(f"Average tokens per sample: {train_dataset.tokens / len(train_dataset):.2f}")
        LOGGER.info(f"Using training config: {numerical_configs}")
        LOGGER.info(f"Initiating training with {'mixed-precision' if MIXED_PRECISION_ENABLED else 'single-precision'}...")
        LOGGER.info(f"Using model config: {model_config}")
        LOGGER.info(f"Model size: {sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2):.2f}MB")
        LOGGER.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train(training_config, model, train_dataset, val_dataset, args.is_distributed, training_state)

    if args.is_distributed:
        dist.destroy_process_group()