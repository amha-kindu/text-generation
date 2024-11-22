import sys
import torch
from config import *
from PyQt5.QtGui import QFont
from tokenizers import Tokenizer
from model import GPTmodel
from dataset import TextDataset
from train import get_tokenizer
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QGridLayout


class MtInferenceEngine:
    
    def __init__(self, model: GPTmodel, tokenizer: Tokenizer, top_k: int= 5, nucleus_threshold=10) -> None:
        self.model = model
        self.top_k = top_k
        self.tokenizer = tokenizer
        self.nucleus_threshold = nucleus_threshold

        self.sos_id = self.tokenizer.token_to_id("[SOS]")
        self.pad_id = self.tokenizer.token_to_id("[PAD]")

        self.model.eval()

    def size(self, tensor: torch.Tensor) -> int:
        return (tensor == self.pad_id).nonzero()[0][1].item() - 1
       
    @torch.no_grad() 
    def translate(self, text: str, max_len: int) -> str:
        dataset = TextDataset(
            dataset=[text],
            tokenizer=self.tokenizer
        )
        batch_iterator = iter(dataset.batch_iterator(1))
        batch = next(batch_iterator)
        
        # (1, 1, seq_len, seq_len) 
        decoder_mask: torch.Tensor = batch["decoder_mask"].to(DEVICE)

        # (1, seq_len)
        decoder_input: torch.Tensor = batch["decoder_input"].to(DEVICE)

        tokens = self.size(decoder_input)

        # Initialize the decoder input with the continue token
        predicted_token = None
        while tokens < max_len:
            # (1, seq_len, d_model)
            decoder_out = model.decode(decoder_input, decoder_mask)

            # (1, d_model)
            temp = decoder_out[:, tokens + 1]
            
            # (1, d_model) --> (1, vocab_size)
            logits = model.project(temp)
            
            # Evaluate the probability distribution across the vocab_size 
            # dimension using softmax
            # (1, vocab_size)
            probab_distribution = torch.softmax(logits, dim=1)
            
            # Greedily pick the token with the highest probability
            _, predicted_token = torch.max(probab_distribution, dim=1)
            
            # Add the predicted token to the decoder input for the subsequent iterations
            decoder_input[0, tokens + 1] = predicted_token.item()

            tokens += 1

        # Remove the batch dimension
        # (1, seq_len) ---> (seq_len,)
        decoder_input = decoder_input.squeeze(0)

        return self.tokenizer.decode(decoder_input.detach().cpu().tolist())


class TranslationApp(QWidget):
    def __init__(self, inference_engine: MtInferenceEngine):
        super().__init__()
        self.init_ui()
        self.inference_engine = inference_engine

    def init_ui(self):
        self.setWindowTitle('Translation App')
        self.setGeometry(100, 100, 600, 400)

        self.input_label = QLabel('Input(English):')
        self.input_label.setFont(QFont('Arial', 12))  
        self.input_textbox = QLineEdit(self)
        self.input_textbox.setFont(QFont('Nyala', 14))  
        self.input_textbox
        self.input_textbox.returnPressed.connect(self.on_translate_button_clicked)

        self.output_label = QLabel('Output(Amharic):')
        self.output_label.setFont(QFont('Arial', 12))  
        self.output_textbox = QLineEdit(self)
        self.output_textbox.setReadOnly(True)
        self.output_textbox.setFont(QFont('Nyala', 14)) 

        self.translate_button = QPushButton('Translate', self)
        self.translate_button.setFont(QFont('Arial', 12))
        self.translate_button.setFixedWidth(self.width() // 2)
        self.translate_button.clicked.connect(self.on_translate_button_clicked)

        layout = QGridLayout()
        layout.addWidget(self.input_label, 0, 0)
        layout.addWidget(self.input_textbox, 0, 1)
        layout.addWidget(self.output_label, 1, 0)
        layout.addWidget(self.output_textbox, 1, 1)
        layout.addWidget(self.translate_button, 2, 1, 1, 2)

        self.setLayout(layout)

    def on_translate_button_clicked(self):
        input_text = self.input_textbox.text()
        prediction: str = self.inference_engine.translate(input_text, 10)
        self.output_textbox.setText(prediction)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    state = torch.load("./models/tmodel-en-am-v1-20k.pt", map_location=DEVICE)
    model = GPTmodel.build(VOCAB_SIZE, state).to(DEVICE)

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {DEVICE}")
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Model Size(MB): {total_params * 4 / (1024 ** 2):.2f}MB")
    
    tokenizer: Tokenizer = get_tokenizer()
    inference_engine = MtInferenceEngine(model, tokenizer)

    translation_app = TranslationApp(inference_engine)
    translation_app.show()
    sys.exit(app.exec_())