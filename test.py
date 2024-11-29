import torch
import torch.nn as nn
from tqdm import tqdm
from config import BATCH_SIZE, DEVICE, MODELS_FOLDER, PRELOAD_MODEL_FILEPATH, VOCAB_SIZE
from dataset import TextDataset
from model import GPTmodel
from train import get_dataset


@torch.no_grad()
def test(model: GPTmodel, test_dataset: TextDataset):
    model.eval()

    loss_func = nn.CrossEntropyLoss(ignore_index=test_dataset.tokenizer.pad_id(), label_smoothing=0.1).to(DEVICE)

    batch_iterator = tqdm(test_dataset.batch_iterator(BATCH_SIZE), desc=f"Evaluating model on test dataset", colour="GREEN")

    evaluation_loss = 0
    for batch in batch_iterator:
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

        batch_iterator.set_postfix({"test_loss": f"{test_loss.item():6.3f}"})

        evaluation_loss += test_loss.item()

    avg_loss = evaluation_loss / len(batch_iterator)
    print(f"\nTesting finished with an average cross-entropy loss of {avg_loss:.3f}")



if __name__ == "__main__":
    print(f"Testing started on `{DEVICE}` device")
    state = torch.load(f"{MODELS_FOLDER}/{PRELOAD_MODEL_FILEPATH}.pt", map_location=DEVICE)
    model = GPTmodel.build(state["model_state_dict"]).to(DEVICE)

    _, _, test_dataset = get_dataset()
    
    test(model, test_dataset)