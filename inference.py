import sys
import torch
from config import *
from PyQt5.QtGui import QFont
from model import GPTmodel
import sentencepiece as spm
from preprocessor import AmharicPreprocessor
from train import get_tokenizer
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QGridLayout


class GptInferenceEngine:

    def __init__(self, model: GPTmodel, tokenizer: spm.SentencePieceProcessor, top_k: int= 5, nucleus_threshold=10) -> None:
        self.model = model
        self.top_k = top_k
        self.tokenizer = tokenizer
        self.nucleus_threshold = nucleus_threshold
        self.pad_id = self.tokenizer.pad_id()
        self.eos_id = self.tokenizer.eos_id()
        self.preprocessor = AmharicPreprocessor(tokenizer)

        self.model.eval()

    @torch.no_grad()
    def complete(self, text: str, max_len: int) -> str:
        token_ids = self.preprocessor.preprocess(text)
        padding = SEQ_LEN - len(token_ids)

        # (1, SEQ_LEN)
        decoder_input = torch.concat([
            torch.tensor(token_ids, dtype=torch.int64),
            torch.tensor([self.pad_id] * padding, dtype=torch.int64)
        ]).unsqueeze(0).to(LOCAL_RANK)

        predicted_token = None
        non_pad_tokens = len(token_ids)
        while non_pad_tokens > 0 and non_pad_tokens < max_len and predicted_token != self.eos_id:
            # (1, SEQ_LEN, VOCAB_SIZE)
            logits = self.model(decoder_input)

            # (1, VOCAB_SIZE)
            next_token_logits = logits[:, non_pad_tokens - 1]

            # Evaluate the probability distribution across the VOCAB_SIZE
            # dimension using softmax - (1, VOCAB_SIZE)
            probab_distribution = torch.softmax(next_token_logits, dim=1)

            # Greedily pick the token with the highest probability
            _, predicted_token = torch.max(probab_distribution, dim=1)

            # Add the predicted token to the decoder input for the subsequent iterations
            decoder_input[:, non_pad_tokens] = predicted_token.item()

            non_pad_tokens += 1

        # Remove the batch dimension
        # (1, SEQ_LEN) ---> (SEQ_LEN,)
        decoder_input = decoder_input.squeeze(0)

        return self.tokenizer.Decode(decoder_input.detach().cpu().tolist())


class TranslationApp(QWidget):
    def __init__(self, inference_engine: GptInferenceEngine):
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

    state = torch.load("./models/tmodel-en-am-v1-20k.pt", map_location=LOCAL_RANK)
    model = GPTmodel.build(VOCAB_SIZE, state).to(LOCAL_RANK)

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f"Device: {LOCAL_RANK}")
    LOGGER.info(f"Total Parameters: {total_params}")
    LOGGER.info(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    LOGGER.info(f"Model Size(MB): {total_params * 4 / (1024 ** 2):.2f}MB")
    
    tokenizer: spm.SentencePieceProcessor = get_tokenizer()
    inference_engine = GptInferenceEngine(model, tokenizer)

    translation_app = TranslationApp(inference_engine)
    translation_app.show()
    sys.exit(app.exec_())