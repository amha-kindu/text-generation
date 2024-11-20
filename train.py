import json
import torch
from config import *
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from tokenizers import Tokenizer
from model import MtTransformerModel
from dataset import TextDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader



def get_tokenizer() -> Tokenizer:
    tokenizer_path = Path(TOKENIZER_FILEPATH)
    tokenizer: Tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=SEQ_LEN)
    
    return tokenizer

def get_dataset() -> tuple[TextDataset, TextDataset, TextDataset]:
    with open(DATASET_PATH, 'r', encoding='utf-8') as data:
        dataset = json.load(data)
    
    train_size = int(0.8 * len(dataset))
    test_size = int(0.15 * len(dataset))
    val_size = len(dataset) - train_size - test_size
    
    train_test_raw, val_raw = random_split(dataset, (train_size+test_size, val_size))
    train_raw, test_raw = random_split(train_test_raw, (train_size, test_size))
    
    tokenizer = get_tokenizer()

    train_dataset = TextDataset(train_raw, tokenizer)
    val_dataset = TextDataset(val_raw, tokenizer)
    test_dataset = TextDataset(test_raw, tokenizer)
    
    return train_dataset, val_dataset, test_dataset
    
    
@torch.no_grad()
def validate(model: MtTransformerModel, val_batch_iterator: DataLoader, loss_func: nn.CrossEntropyLoss):
    #Set the transformer module(the model) to evaluation mode
    model.eval()

    val_loss = 0
    # Evaluate model with `num_examples` number of random examples
    for batch in val_batch_iterator:
        # Retrieve the data points from the current batch
        encoder_input = batch["encoder_input"].to(DEVICE)       # (batches, seq_len)
        decoder_input = batch["decoder_input"].to(DEVICE)       # (batches, seq_len)
        encoder_mask = batch["encoder_mask"].to(DEVICE)         # (batches, 1, 1, seq_len)
        decoder_mask = batch["decoder_mask"].to(DEVICE)         # (batches, 1, seq_len, seq_len)
        label: torch.Tensor = batch['label'].to(DEVICE)         # (batches, seq_len)

        # Perform the forward pass according to the operations defined in
        # the transformer model in order to build the computation graph of the model
        encoder_output = model.encode(encoder_input, encoder_mask)                                  # (batches, seq_len, d_model)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)    # (batches, seq_len, d_model)
        proj_output: torch.Tensor = model.project(decoder_output)                                   # (batches, seq_len, vocab_size)

        # Compute the cross entropy loss
        loss: torch.Tensor = loss_func(
            proj_output.view(-1, val_dataset.tokenizer.get_vocab_size()),     # (batches, seq_len, vocab_size) --> (batches*seq_len, vocab_size)
            label.view(-1)                                                          # (batches, seq_len) --> (batches * seq_len, )
        )

        val_loss += loss.item()

    return val_loss / len(val_batch_iterator)

    
def train(model: MtTransformerModel, train_dataset: TextDataset, val_dataset: TextDataset) -> None:   
    # Configure Tensorboard
    writer = SummaryWriter(TB_LOG_DIR)
    
    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR, eps=1e-09)
    
    initial_epoch = 0
    global_step = 0
    if PRELOAD_MODEL_SUFFIX:
        model_filename = f"{MODEL_FOLDER}/{PRELOAD_MODEL_SUFFIX}.pt"
        print(f"Preloading model {model_filename}")
        
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        
    loss_func = nn.CrossEntropyLoss(ignore_index=train_dataset.tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(DEVICE)
    
    batch_iterator = train_dataset.batch_iterator(BATCH_SIZE)
    val_batch_iterator = val_dataset.batch_iterator(BATCH_SIZE)
    
    training_loss = 0
    validation_loss = 0
    for epoch in range(initial_epoch, EPOCHS):
        # Wrap train_dataloader with tqdm to show a progress bar to show
        # how much of the batches have been processed on the current epoch
        batch_iterator = tqdm(batch_iterator, desc=f"Processing epoch {epoch: 02d}", colour="BLUE")
        
        n = len(batch_iterator)
        # Iterate through the batches
        for idx, batch in enumerate(batch_iterator):    
            """
                Set the transformer module(the model) to back to training mode
            """
            model.train() 
                 
            # Retrieve the data points from the current batch
            encoder_input = batch["encoder_input"].to(DEVICE)       # (batch, seq_len) 
            decoder_input = batch["decoder_input"].to(DEVICE)       # (batch, seq_len) 
            encoder_mask = batch["encoder_mask"].to(DEVICE)         # (batch, 1, 1, seq_len) 
            decoder_mask = batch["decoder_mask"].to(DEVICE)         # (batch, 1, seq_len, seq_len) 
            label: torch.Tensor = batch['label'].to(DEVICE)         # (batch, seq_len)
            
            # Perform the forward pass according to the operations defined in 
            # the transformer model in order to build the computation graph of the model
            encoder_output = model.encode(encoder_input, encoder_mask)                                  # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)    # (batch, seq_len, d_model)
            proj_output: torch.Tensor = model.project(decoder_output)                                   # (batch, seq_len, vocab_size)
                        
            # Compute the training loss
            batch_loss = loss_func.forward(
                proj_output.view(-1, train_dataset.tokenizer.get_vocab_size()),     # (batch, seq_len, vocab_size) --> (batch*seq_len, vocab_size)
                label.view(-1)                                                          # (batch, seq_len) --> (batch * seq_len, )
            )
            training_loss += batch_loss.item()

            passed_batches = epoch * n + (idx + 1)
            if global_step % 200 == 0:
                # Evaluate the model on the validation dataset(aka unseen data)
                validation_loss += validate(model, val_batch_iterator, loss_func)
                
                # Log the training and validation loss on tensorboard
                writer.add_scalars("Cross-Entropy-Loss", { "Training": training_loss / passed_batches, "Validation": validation_loss / (passed_batches // 200 + 1) }, global_step)
            else:
                writer.add_scalars("Cross-Entropy-Loss", { "Training": training_loss / passed_batches }, global_step)
                
            writer.flush()
            
            # Add the calculated training loss and validation loss as a postfix to the progress bar shown by tqdm
            batch_iterator.set_postfix({"train_loss": f"{training_loss / passed_batches:6.3f}", "val_loss": f"{validation_loss / (passed_batches // 200 + 1):6.3f}"})

            # Perform the backward pass on the computation graph built during the forward pass, 
            # in order to calculate the grad for each of the intermediate and leaf tensors on the computation graph
            batch_loss.backward()
            
            # Update the model parameters
            optimizer.step()
            
            # Zero the gradients of the model parameters to prevent gradient accumulation 
            optimizer.zero_grad()
                        
            global_step += 1
        
        # Save the model at the end of every epoch
        model_filename = f"{MODEL_FOLDER}/tmodel_epoch-{epoch:02d}_avgTrainLoss-{training_loss / passed_batches:6.3f}_avgValLoss-{validation_loss / (passed_batches // 200 + 1):6.3f}_batch-{BATCH_SIZE}_init_lr-{INIT_LR:.0e}.pt"
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_filename)


if __name__ == "__main__":
    print(f"Training started on `{DEVICE}` device")
    train_dataset, val_dataset, test_dataset = get_dataset()
    
    model = MtTransformerModel.build(train_dataset.tokenizer.get_vocab_size()).to(DEVICE) 
    
    train(model, train_dataset, val_dataset)