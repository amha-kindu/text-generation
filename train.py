import os
import torch
import torch.distributed as dist
from config import *
import torch.nn as nn
from tqdm import tqdm
from model import GPTmodel
from dataset import TextDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import sentencepiece as spm
from torch.utils.data import DistributedSampler
from torch.utils.data.sampler import RandomSampler
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
def validate(model: GPTmodel, val_batch_iterator: DataLoader, loss_func):
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


def train(model: GPTmodel, train_dataset: TextDataset, val_dataset: TextDataset, rank: int, world_size: int, state=None) -> None:
    print("Initiazing DDP...")
    model = DDP(model, device_ids=[rank], output_device=rank)
    print("DDP initialized.")

    writer = SummaryWriter(TB_LOG_DIR)

    optimizer = torch.optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=1e-2)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    initial_epoch = 0
    global_step = 0
    training_loss = 0
    validation_loss = 0
    if state is not None:
        initial_epoch = state["epoch"] + 1    
        global_step = state["global_step"]
        training_loss = state["training_loss"]
        validation_loss = state["validation_loss"]
        optimizer.load_state_dict(state["optimizer_state_dict"])

    loss_func = nn.CrossEntropyLoss(ignore_index=train_dataset.tokenizer.pad_id(), label_smoothing=0.1).to(DEVICE)

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    batch_iterator = train_dataset.batch_iterator(BATCH_SIZE, sampler)

    val_sampler = RandomSampler(val_dataset)
    val_batch_iterator = val_dataset.batch_iterator(BATCH_SIZE, val_sampler)

    for epoch in range(initial_epoch, EPOCHS):
        torch.cuda.empty_cache()
        sampler.set_epoch(epoch)

        batch_iterator = tqdm(batch_iterator, desc=f"GPU {rank} Processing epoch {epoch: 02d}")
        
        for batch in batch_iterator:
            model.train() 
                 
            # (N_BATCHES, SEQ_LEN)
            decoder_input = batch["decoder_input"].to(DEVICE)

            # (1, SEQ_LEN, SEQ_LEN)
            decoder_mask = batch["decoder_mask"].to(DEVICE)
            
            # (N_BATCHES, SEQ_LEN)
            label: torch.Tensor = batch['label'].to(DEVICE)

            # (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
            logits: torch.Tensor = model.forward(decoder_input, decoder_mask)
                        
            # Compute the cross-entropy loss
            batch_loss = loss_func.forward(
                # (N_BATCHES, SEQ_LEN, VOCAB_SIZE) --> (N_BATCHES * SEQ_LEN, VOCAB_SIZE)
                logits.view(-1, VOCAB_SIZE),

                # (N_BATCHES, SEQ_LEN) --> (N_BATCHES * SEQ_LEN, )
                label.view(-1)
            )

            total_loss_tensor: torch.Tensor = torch.tensor(batch_loss).to(rank)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)

            training_loss += total_loss_tensor.item() / dist.get_world_size()

            if rank == 1 and global_step % 200 == 0:
                validation_loss += validate(model, val_batch_iterator, loss_func)

                writer.add_scalars(
                    "Loss", 
                    { 
                        "Validation": validation_loss / ((global_step + 1) // 200 + 1) 
                    },
                    global_step
                )

                writer.flush()

            if rank == 0:
                writer.add_scalars(
                    "Loss",
                    {
                        "Training": training_loss / (global_step + 1) 
                    }, 
                    global_step
                )
                
                writer.flush()
            
                batch_iterator.set_postfix({"train_loss": f"{training_loss / (global_step + 1):6.3f}", "val_loss": f"{validation_loss / ((global_step + 1) // 200 + 1):6.3f}"})

            # Perform the backward pass on the computation graph built during the forward pass, 
            # in order to calculate the grad for each of the intermediate and leaf tensors on the computation graph
            batch_loss.backward()
            
            # Update the model parameters
            optimizer.step()
            
            # Zero the gradients of the model parameters to prevent gradient accumulation 
            optimizer.zero_grad()

            global_step += 1
        
        if rank == 0:
            # Save the model at the end of every epoch
            model_filename = f"{MODELS_FOLDER}/amharic-gpt-model.pt"
            
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
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    world_size = torch.cuda.device_count()
    if rank == 0:
        print(f"Identified {world_size} GPUs...")
   
    DEVICE = torch.device(f"cuda:{rank}")

    train_dataset, val_dataset, _ = get_dataset()
    model = GPTmodel.build().to(DEVICE)

    state = None
    if PRELOAD_MODEL_FILEPATH:
        model_filename = f"{MODELS_FOLDER}/{PRELOAD_MODEL_FILEPATH}.pt"
        if rank == 0:
            print(f"Preloading model {model_filename}...")

        state: dict = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        state = {key: value for key, value in state.items() if key != "model_state_dict"}

    print("Initializing process group...")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print("Process group initialized...")

    train(model, train_dataset, val_dataset, rank, world_size, state)
    dist.destroy_process_group()