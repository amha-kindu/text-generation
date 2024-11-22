import json
import torch
from config import *
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from tokenizers import Tokenizer
from model import GPTmodel
from dataset import TextDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader


def get_tokenizer() -> Tokenizer:
    tokenizer: Tokenizer = Tokenizer.from_file(TOKENIZER_FILEPATH)
    tokenizer.enable_truncation(max_length=SEQ_LEN)
    
    return tokenizer

def get_dataset() -> tuple[TextDataset, TextDataset, TextDataset]:
    with open(DATASET_PATH, 'r', encoding='utf-8') as file:
        texts = file.readlines()
    
    train_size = int(0.8 * len(texts))
    test_size = int(0.15 * len(texts))
    val_size = len(texts) - train_size - test_size
    
    train_test_raw, val_raw = random_split(texts, (train_size+test_size, val_size))
    train_raw, test_raw = random_split(train_test_raw, (train_size, test_size))
    
    tokenizer = get_tokenizer()

    train_dataset = TextDataset(train_raw, tokenizer)
    val_dataset = TextDataset(val_raw, tokenizer)
    test_dataset = TextDataset(test_raw, tokenizer)
    
    return train_dataset, val_dataset, test_dataset
    
    
@torch.no_grad()
def validate(model: GPTmodel, val_batch_iterator: DataLoader, loss_func: nn.CrossEntropyLoss):
    #Set the transformer module(the model) to evaluation mode
    model.eval()

    val_loss = 0
    # Evaluate model with `num_examples` number of random examples
    for batch in val_batch_iterator:
        # Retrieve the data points from the current batch
        # (batches, seq_len)
        decoder_input = batch["decoder_input"].to(DEVICE)

        # (batches, 1, seq_len, seq_len)
        decoder_mask = batch["decoder_mask"].to(DEVICE)

        # (batches, seq_len, d_model)
        label: torch.Tensor = batch['label'].to(DEVICE)


        # (batches, seq_len, d_model)
        decoder_output = model.decode(decoder_input, decoder_mask)

        # (batches, seq_len, vocab_size)
        proj_output: torch.Tensor = model.project(decoder_output)

        # Compute the cross-entropy loss
        loss: torch.Tensor = loss_func(
            # (batches, seq_len, vocab_size) --> (batches*seq_len, vocab_size)
            proj_output.view(-1, val_dataset.tokenizer.get_vocab_size()),

            # (batches, seq_len) --> (batches * seq_len, )
            label.view(-1)
        )
        val_loss += loss.item()

        break

    return val_loss

    
def train(model: GPTmodel, train_dataset: TextDataset, val_dataset: TextDataset) -> None:   
    # Configure Tensorboard
    writer = SummaryWriter(TB_LOG_DIR)
    
    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR, eps=1e-09)
    
    initial_epoch = 0
    global_step = 0
    training_loss = 0
    validation_loss = 0
    if PRELOAD_MODEL_FILEPATH:
        model_filename = f"{MODELS_FOLDER}/{PRELOAD_MODEL_FILEPATH}.pt"
        print(f"Preloading model {model_filename}")

        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        training_loss = state["training_loss"]
        validation_loss = state["validation_loss"]

        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])

    loss_func = nn.CrossEntropyLoss(ignore_index=train_dataset.tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(DEVICE)

    batch_iterator = train_dataset.batch_iterator(BATCH_SIZE)
    val_batch_iterator = val_dataset.batch_iterator(BATCH_SIZE)

    for epoch in range(initial_epoch, EPOCHS):
        batch_iterator = tqdm(batch_iterator, desc=f"Processing epoch {epoch: 02d}", colour="BLUE")
        
        for batch in batch_iterator:
            model.train() 
                 
            # (batch, seq_len)
            decoder_input = batch["decoder_input"].to(DEVICE)

            # (batch, 1, seq_len, seq_len)
            decoder_mask = batch["decoder_mask"].to(DEVICE)
            
            # (batch, seq_len)
            label: torch.Tensor = batch['label'].to(DEVICE)
            
            # (batch, seq_len, d_model)
            decoder_output = model.decode(decoder_input, decoder_mask)

            # (batch, seq_len, vocab_size)
            logits: torch.Tensor = model.project(decoder_output)

                        
            # Compute the cross-entropy loss
            batch_loss = loss_func.forward(
                # (batch, seq_len, vocab_size) --> (batch*seq_len, vocab_size)
                logits.view(-1, train_dataset.tokenizer.get_vocab_size()),

                # (batch, seq_len) --> (batch * seq_len, )
                label.view(-1)
            )
            training_loss += batch_loss.item()

            if global_step % 200 == 0:
                # Evaluate the model on the validation dataset(aka unseen data)
                validation_loss += validate(model, val_batch_iterator, loss_func)
                
                # Log the training and validation loss on tensorboard
                writer.add_scalars("Cross-Entropy-Loss", { "Training": training_loss / (global_step + 1), "Validation": validation_loss / ((global_step + 1) // 200 + 1) }, global_step)
            else:
                writer.add_scalars("Cross-Entropy-Loss", { "Training": training_loss / (global_step + 1) }, global_step)
                
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
        
        # Save the model at the end of every epoch
        model_filename = f"{MODELS_FOLDER}/tmodel_avgTrainLoss-{training_loss / global_step:6.3f}_avgValLoss-{validation_loss / (global_step // 200 + 1):6.3f}.pt"
        
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
    print(f"Training started on `{DEVICE}` device")
    train_dataset, val_dataset, test_dataset = get_dataset()
    
    model = GPTmodel.build(train_dataset.tokenizer.get_vocab_size()).to(DEVICE) 
    
    train(model, train_dataset, val_dataset)