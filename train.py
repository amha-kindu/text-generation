import os
import torch
import argparse
from config import *
import torch.nn as nn
from tqdm import tqdm
from model import GPTmodel
from dataset import TextDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
import sentencepiece as spm
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel


def get_tokenizer() -> spm.SentencePieceProcessor:
    tokenizer: spm.SentencePieceProcessor = spm.SentencePieceProcessor()
    tokenizer.LoadFromFile(TOKENIZER_FILEPATH)
    
    return tokenizer

def get_dataset() -> tuple[TextDataset, TextDataset, TextDataset]:
    tokenizer = get_tokenizer()

    train_dataset = TextDataset(TRAINING_DATA_FILEPATH, tokenizer)
    val_dataset = TextDataset(VALIDATION_DATA_FILEPATH, tokenizer)
    test_dataset = TextDataset(TEST_DATA_FILEPATH, tokenizer)
    
    return train_dataset, val_dataset, test_dataset


@torch.no_grad()
def test(model: GPTmodel, test_dataset: TextDataset, is_distributed: bool=False):
    model.eval()
    IS_MASTER = GLOBAL_RANK == MASTER_RANK

    loss_func = nn.CrossEntropyLoss(ignore_index=test_dataset.tokenizer.pad_id(), label_smoothing=0.1).to(DEVICE)

    sampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(), rank=LOCAL_RANK) if is_distributed else None
    batch_iterator = tqdm(test_dataset.batch_iterator(BATCH_SIZE,sampler=sampler), desc=f"Evaluating model on test dataset", disable=not IS_MASTER)

    evaluation_loss = 0
    for index, batch in enumerate(batch_iterator):
        # (N_BATCHES, SEQ_LEN)
        decoder_input = batch["decoder_input"].to(DEVICE)

        # (1, SEQ_LEN, SEQ_LEN)
        decoder_mask = batch["decoder_mask"].to(DEVICE)

        # (N_BATCHES, SEQ_LEN)
        label: torch.Tensor = batch['label'].to(DEVICE)

        # (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
        logits: torch.Tensor = model.forward(decoder_input, decoder_mask)

        # Compute the training loss
        test_loss: torch.Tensor = loss_func(
            # (N_BATCHES, SEQ_LEN, VOCAB_SIZE) --> (N_BATCHES * SEQ_LEN, VOCAB_SIZE)
            logits.view(-1, VOCAB_SIZE),

            # (N_BATCHES, SEQ_LEN) --> (N_BATCHES * SEQ_LEN, )
            label.view(-1)
        )

        batch_iterator.set_postfix({"avg test_loss": f"{test_loss.item() / (index + 1):6.3f}"})

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

        # (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
        logits: torch.Tensor = model(decoder_input, decoder_mask)

        # Compute the cross-entropy loss
        loss: torch.Tensor = loss_func(
            # (N_BATCHES, SEQ_LEN, VOCAB_SIZE) --> (N_BATCHES * SEQ_LEN, VOCAB_SIZE)
            logits.view(-1, VOCAB_SIZE),

            # (N_BATCHES, SEQ_LEN) --> (N_BATCHES * SEQ_LEN, )
            label.view(-1)
        ) 
        val_loss += loss.item()

    return val_loss / len(val_batch_iterator)


def train(model: GPTmodel, train_dataset: TextDataset, val_dataset: TextDataset, is_distributed: bool = False, state: dict = {}) -> None:   
    IS_MASTER = GLOBAL_RANK == MASTER_RANK
    writer = SummaryWriter(TB_LOG_DIR)

    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[LOCAL_RANK])

    optimizer = torch.optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=1e-2)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
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

    sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    batch_iterator = train_dataset.batch_iterator(BATCH_SIZE, sampler=sampler)

    val_sampler = RandomSampler(val_dataset, replacement=True, num_samples=VALIDATION_SAMPLES)
    val_batch_iterator = val_dataset.batch_iterator(BATCH_SIZE, sampler=val_sampler)

    for epoch in range(initial_epoch, EPOCHS):
        if is_distributed:
            sampler.set_epoch(epoch)
        
        batch_iterator = tqdm(batch_iterator, desc=f"{LOGGER.name}: Processing epoch {epoch: 02d}", disable = not IS_MASTER)
        for batch in batch_iterator:
            model.train() 
                 
            # (N_BATCHES, SEQ_LEN)
            decoder_input = batch["decoder_input"].to(DEVICE)

            # (1, SEQ_LEN, SEQ_LEN)
            decoder_mask = batch["decoder_mask"].to(DEVICE)
            
            # (N_BATCHES, SEQ_LEN)
            label: torch.Tensor = batch['label'].to(DEVICE)

            # (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
            logits: torch.Tensor = model(decoder_input, decoder_mask)
    
            # Compute the cross-entropy loss
            batch_loss = loss_func.forward(
                # (N_BATCHES, SEQ_LEN, VOCAB_SIZE) --> (N_BATCHES * SEQ_LEN, VOCAB_SIZE)
                logits.view(-1, VOCAB_SIZE),

                # (N_BATCHES, SEQ_LEN) --> (N_BATCHES * SEQ_LEN, )
                label.view(-1)
            )
            training_loss += batch_loss.item()

            if IS_MASTER and global_step % 200 == 0:
                validation_loss += validate(model.module, val_batch_iterator, loss_func)

                writer.add_scalars(
                    "Loss", 
                    { 
                        "Training": training_loss / (global_step + 1), 
                        "Validation": validation_loss / ((global_step + 1) // 200 + 1)
                    },
                    global_step
                )
            else:
                    writer.add_scalars(
                        "Loss",
                        {
                            "Training": training_loss / (global_step + 1)
                        },
                        global_step
                    )
     
            writer.flush()

            batch_iterator.set_postfix({
                "train_loss": f"{training_loss / (global_step + 1):6.3f}", 
                "val_loss": validation_loss / ((global_step + 1) // 200 + 1)
            })

            # Perform the backward pass on the computation graph built during the forward pass, 
            # in order to calculate the grad for each of the intermediate and leaf tensors on the computation graph
            batch_loss.backward()
            
            # Update the model parameters
            optimizer.step()
            
            # Zero the gradients of the model parameters to prevent gradient accumulation 
            optimizer.zero_grad()

            global_step += 1
        
        if IS_MASTER:
            model_filename = f"{WEIGHTS_DIRECTORY}/amharic-gpt-base-model-v1.pt"        
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "model_hyperparams":{
                    "D_MODEL": D_MODEL,
                    "N_BLOCKS": N_BLOCKS,
                    "HEADS": HEADS,
                    "DROPOUT": DROPOUT,
                    "DFF": DFF,
                    "BATCH_SIZE": BATCH_SIZE,
                    "INIT_LR": INIT_LR
                }
            }, model_filename)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument("--is-distributed", action="store_true", help="Device to train the model on")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size used during training")
    parser.add_argument("--init-lr", type=float, default=INIT_LR, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train the model")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="Sequence length of the input")
    parser.add_argument("--d-model", type=int, default=D_MODEL, help="Dimensionality of the model")
    parser.add_argument("--n-blocks", type=int, default=N_BLOCKS, help="Number of decoder blocks")
    parser.add_argument("--heads", type=int, default=HEADS, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="Dropout probability")
    parser.add_argument("--dff", type=int, default=DFF, help="Dimensionality of the feed forward layer")
    parser.add_argument("--dist-backend", type=str, default="nccl", help="Distributed backend")
    parser.add_argument("--preload-weights", type=str, default="", help="File path to load saved weights")
    parser.add_argument("--validation-samples", type=int, default=VALIDATION_SAMPLES, help="Number of samples to use for a single validation run")

    args = parser.parse_args()

    LOGGER.name = "GPU" if torch.cuda.is_available() else "CPU"
    if args.is_distributed:
        assert torch.cuda.device_count() > 1, "Must have more than one CUDA supporting GPUs to initiate distributed training"
        assert args.dist_backend in ["nccl", "gloo" "mpi", "ucc"], "Distributed backend must be one of the following: nccl, gloo, mpi or ucc"

        GLOBAL_RANK = int(os.environ["RANK"])
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
        torch.cuda.set_device(DEVICE)

        LOGGER.name = f"GPU {GLOBAL_RANK}" if torch.cuda.is_available() else f"CPU {GLOBAL_RANK}"

        dist.init_process_group(backend=args.dist_backend)

    assert args.d_model % args.heads == 0, "d_model must be divisible by heads"

    PRELOAD_WEIGHTS_FILEPATH = args.preload_weights
    VALIDATION_SAMPLES = args.validation_samples
    BATCH_SIZE = args.batch_size
    N_BLOCKS = args.n_blocks
    SEQ_LEN = args.seq_len
    INIT_LR = args.init_lr
    D_MODEL = args.d_model
    DROPOUT = args.dropout
    EPOCHS = args.epochs
    HEADS = args.heads
    DFF = args.dff

    os.makedirs(WEIGHTS_DIRECTORY, exist_ok=True)
    train_dataset, val_dataset, test_dataset = get_dataset()

    state, weights = {}, {}
    if PRELOAD_WEIGHTS_FILEPATH:
        if not os.path.exists(PRELOAD_WEIGHTS_FILEPATH):
            raise FileNotFoundError(f"File {PRELOAD_WEIGHTS_FILEPATH} does not exist")
        
        LOGGER.info(f"Preloading model weights {PRELOAD_WEIGHTS_FILEPATH}...")
        state: dict = torch.load(PRELOAD_WEIGHTS_FILEPATH, map_location=DEVICE)
        weights = state["model_state_dict"]
        state.pop("model_state_dict")
    
    model = GPTmodel.build(weights).to(DEVICE)

    train(model, train_dataset, val_dataset, args.is_distributed, state)

    test(model, test_dataset, args.is_distributed)

    if args.is_distributed:
        dist.destroy_process_group()