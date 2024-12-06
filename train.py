import os
import torch
import argparse
from config import *
import torch.nn as nn
from tqdm import tqdm
from model import GPTmodel
from dataset import TextDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import sentencepiece as spm
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


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

    loss_func = nn.CrossEntropyLoss(ignore_index=test_dataset.tokenizer.pad_id(), label_smoothing=0.1).to(LOCAL_RANK)

    sampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(), rank=LOCAL_RANK) if is_distributed else None
    batch_iterator = tqdm(test_dataset.batch_iterator(BATCH_SIZE,sampler=sampler), desc=f"Evaluating model on test dataset")

    evaluation_loss = 0
    for index, batch in enumerate(batch_iterator):
        # (N_BATCHES, SEQ_LEN)
        decoder_input = batch["decoder_input"].to(LOCAL_RANK)

        # (1, SEQ_LEN, SEQ_LEN)
        decoder_mask = batch["decoder_mask"].to(LOCAL_RANK)

        # (N_BATCHES, SEQ_LEN)
        label: torch.Tensor = batch['label'].to(LOCAL_RANK)

        # (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
        logits: torch.Tensor = model.forward(decoder_input, decoder_mask)

        # Compute the training loss
        test_loss: torch.Tensor = loss_func(
            # (N_BATCHES, SEQ_LEN, VOCAB_SIZE) --> (N_BATCHES * SEQ_LEN, VOCAB_SIZE)
            logits.view(-1, VOCAB_SIZE),

            # (N_BATCHES, SEQ_LEN) --> (N_BATCHES * SEQ_LEN, )
            label.view(-1)
        )

        if LOCAL_RANK == MASTER_LOCALRANK:
            batch_iterator.set_postfix({"avg test_loss": f"{test_loss.item() / (index + 1):6.3f}"})

        evaluation_loss += test_loss.item()
    
    
@torch.no_grad()
def validate(model: GPTmodel, val_batch_iterator: DataLoader, loss_func: nn.CrossEntropyLoss):
    model.eval()

    val_loss = 0
    for batch in val_batch_iterator:
        # (N_BATCHES, SEQ_LEN)
        decoder_input = batch["decoder_input"].to(LOCAL_RANK)

        # (1, SEQ_LEN, SEQ_LEN)
        decoder_mask = batch["decoder_mask"].to(LOCAL_RANK)

        # (N_BATCHES, SEQ_LEN)
        label: torch.Tensor = batch['label'].to(LOCAL_RANK)

        # (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
        logits: torch.Tensor = model.forward(decoder_input, decoder_mask)

        # Compute the cross-entropy loss
        loss: torch.Tensor = loss_func(
            # (N_BATCHES, SEQ_LEN, VOCAB_SIZE) --> (N_BATCHES * SEQ_LEN, VOCAB_SIZE)
            logits.view(-1, VOCAB_SIZE),

            # (N_BATCHES, SEQ_LEN) --> (N_BATCHES * SEQ_LEN, )
            label.view(-1)
        )
        val_loss += loss.item()

    return val_loss / len(val_batch_iterator)


def train(model: GPTmodel, train_dataset: TextDataset, val_dataset: TextDataset, is_distributed: bool = False) -> None:   
    writer = SummaryWriter(TB_LOG_DIR)
    IS_MASTER = (RANK == 0 and LOCAL_RANK == MASTER_LOCALRANK)
    IS_VALIDATOR = (RANK == 0 and LOCAL_RANK == VALIDATOR_LOCALRANK)

    if is_distributed:
        model = DDP(model, device_ids=[LOCAL_RANK])

    optimizer = torch.optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=1e-2)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    initial_epoch = 0
    global_step = 0
    training_loss = 0
    validation_loss = 0
    if PRELOAD_MODEL_FILENAME:
        model_filename = f"{MODELS_FOLDER}/{PRELOAD_MODEL_FILENAME}.pt"
        LOGGER.info(f"Preloading model {model_filename}...")

        state = torch.load(model_filename, map_location=f"cuda:{LOCAL_RANK}")
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        training_loss = state["training_loss"]
        validation_loss = state["validation_loss"]

        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])

    loss_func = nn.CrossEntropyLoss(ignore_index=train_dataset.tokenizer.pad_id(), label_smoothing=0.1).to(LOCAL_RANK)

    sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()) if is_distributed else None
    batch_iterator = train_dataset.batch_iterator(BATCH_SIZE, sampler=sampler)

    val_sampler = DistributedSampler(val_dataset, num_replicas=1, rank=VALIDATOR_LOCALRANK) if is_distributed else None
    val_batch_iterator = val_dataset.batch_iterator(BATCH_SIZE, sampler=val_sampler)

    for epoch in range(initial_epoch, EPOCHS):
        if is_distributed:
            sampler.set_epoch(epoch)

        torch.cuda.empty_cache()
        batch_iterator = tqdm(batch_iterator, desc=f"{LOGGER.name}: Processing epoch {epoch: 02d}")
        
        for batch in batch_iterator:
            model.train() 
                 
            # (N_BATCHES, SEQ_LEN)
            decoder_input = batch["decoder_input"].to(LOCAL_RANK)

            # (1, SEQ_LEN, SEQ_LEN)
            decoder_mask = batch["decoder_mask"].to(LOCAL_RANK)
            
            # (N_BATCHES, SEQ_LEN)
            label: torch.Tensor = batch['label'].to(LOCAL_RANK)

            # (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
            logits: torch.Tensor = model.forward(decoder_input, decoder_mask)
                        
            # Compute the cross-entropy loss
            batch_loss = loss_func.forward(
                # (N_BATCHES, SEQ_LEN, VOCAB_SIZE) --> (N_BATCHES * SEQ_LEN, VOCAB_SIZE)
                logits.view(-1, VOCAB_SIZE),

                # (N_BATCHES, SEQ_LEN) --> (N_BATCHES * SEQ_LEN, )
                label.view(-1)
            )

            if global_step and (not is_distributed or (is_distributed and (IS_MASTER or IS_VALIDATOR))):
                if (not is_distributed or IS_VALIDATOR) and global_step % 200 == 0:
                    validation_loss += validate(model, val_batch_iterator, loss_func)

                    writer.add_scalars(
                        "Loss", 
                        { 
                            "Training": training_loss / global_step, 
                            "Validation": validation_loss / (global_step // 200 + 1)
                        },
                        global_step
                    )
                else:
                    writer.add_scalars(
                        "Loss",
                        {
                            "Training": training_loss / global_step
                        }, 
                        global_step
                    )
                    
                writer.flush()

            # Perform the backward pass on the computation graph built during the forward pass, 
            # in order to calculate the grad for each of the intermediate and leaf tensors on the computation graph
            batch_loss.backward()
            
            # Update the model parameters
            optimizer.step()
            
            # Zero the gradients of the model parameters to prevent gradient accumulation 
            optimizer.zero_grad()

            global_step += 1

            if is_distributed and IS_MASTER:
                total_loss = torch.tensor(batch_loss.item(), device=LOCAL_RANK)
                dist.reduce(total_loss, dst=MASTER_LOCALRANK, op=dist.ReduceOp.SUM)

                training_loss += total_loss.item() / dist.get_world_size()
            else:
                training_loss += batch_loss.item()

            batch_iterator.set_postfix({"train_loss": f"{training_loss / global_step:6.3f}", "val_loss": f"{validation_loss / (global_step // 200 + 1):6.3f}"})

        
        model_filename = f"{MODELS_FOLDER}/amharic-gpt-base-model.pt"
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
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
    parser.add_argument("--is-distributed", type=bool, default=False, help="Device to train the model on")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size used during training")
    parser.add_argument("--init-lr", type=float, default=INIT_LR, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train the model")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="Sequence length of the input")
    parser.add_argument("--d-model", type=int, default=D_MODEL, help="Dimensionality of the model")
    parser.add_argument("--n-blocks", type=int, default=N_BLOCKS, help="Number of decoder blocks")
    parser.add_argument("--heads", type=int, default=HEADS, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="Dropout probability")
    parser.add_argument("--dff", type=int, default=DFF, help="Dimensionality of the feed forward layer")
    parser.add_argument("--master-localrank", type=int, default=MASTER_LOCALRANK, help="Local rank of the master process")
    parser.add_argument("--validator-localrank", type=int, default=VALIDATOR_LOCALRANK, help="Local rank of the validator process")
    parser.add_argument("--dist-backend", type=str, default="nccl", help="Distributed backend")

    args = parser.parse_args()

    LOGGER.name = "GPU" if torch.cuda.is_available() else "CPU"
    if args.is_distributed:
        assert args.dist_backend in ["nccl", "gloo"], "Distributed backend must be either nccl or gloo"

        dist.init_process_group(backend=args.dist_backend)

        RANK = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        LOGGER.name = "GPU {RANK}:{LOCAL_RANK}" if torch.cuda.is_available() else "CPU {RANK}:{LOCAL_RANK}"

        assert args.master_localrank != args.validator_localrank, "master local rank and validator local rank must be different"
        assert args.master_localrank < dist.get_world_size(), "master local rank must be less than world size"
        assert args.validator_localrank < dist.get_world_size(), "validator local rank must be less than world size"

    assert args.d_model % args.heads == 0, "d_model must be divisible by heads"

    MASTER_LOCALRANK = args.master_localrank
    VALIDATOR_LOCALRANK = args.validator_localrank
    BATCH_SIZE = args.batch_size
    N_BLOCKS = args.n_blocks
    SEQ_LEN = args.seq_len
    INIT_LR = args.init_lr
    D_MODEL = args.d_model
    DROPOUT = args.dropout
    EPOCHS = args.epochs
    HEADS = args.heads
    DFF = args.dff

    train_dataset, val_dataset, test_dataset = get_dataset()
    model = GPTmodel.build().to(LOCAL_RANK) 

    train(model, train_dataset, val_dataset, args.is_distributed)

    test(model, test_dataset, args.is_distributed)

    if args.is_distributed:
        dist.destroy_process_group()