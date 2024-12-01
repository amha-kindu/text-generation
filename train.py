import torch
from config import *
import torch.nn as nn
from tqdm import tqdm
from model import GPTmodel
from dataset import TextDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import sentencepiece as spm


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


def train(model: GPTmodel, train_dataset: TextDataset, val_dataset: TextDataset) -> None:   
    writer = SummaryWriter(TB_LOG_DIR)
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR, eps=1e-08)
    
    initial_epoch = 0
    global_step = 0
    training_loss = 0
    validation_loss = 0
    if PRELOAD_MODEL_FILEPATH:
        model_filename = f"{MODELS_FOLDER}/{PRELOAD_MODEL_FILEPATH}.pt"
        print(f"Preloading model {model_filename}...")

        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        training_loss = state["training_loss"]
        validation_loss = state["validation_loss"]

        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])

    loss_func = nn.CrossEntropyLoss(ignore_index=train_dataset.tokenizer.pad_id(), label_smoothing=0.1).to(DEVICE)

    batch_iterator = train_dataset.batch_iterator(BATCH_SIZE)

    num_of_samples = 10
    val_batch_iterator = val_dataset.random_samples(BATCH_SIZE, num_of_samples)

    for epoch in range(initial_epoch, EPOCHS):
        torch.cuda.empty_cache()
        batch_iterator = tqdm(batch_iterator, desc=f"Processing epoch {epoch: 02d}", colour="BLUE")
        
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
                logits.view(-1, train_dataset.tokenizer.vocab_size()),

                # (N_BATCHES, SEQ_LEN) --> (N_BATCHES * SEQ_LEN, )
                label.view(-1)
            )
            training_loss += batch_loss.item()

            if global_step % 200 == 0:
                validation_loss += validate(model, val_batch_iterator, loss_func)

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
        model_filename = f"{MODELS_FOLDER}/gpt_model-epoch_{epoch}-avg_train_loss_{training_loss / global_step:6.3f}-avg_val_loss_{validation_loss / (global_step // 200 + 1):6.3f}.pt"
        
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

    train_dataset, val_dataset, _ = get_dataset()
    model = GPTmodel.build().to(DEVICE) 

    train(model, train_dataset, val_dataset)