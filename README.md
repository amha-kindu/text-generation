# English to Amharic Machine Translation with Transformers

## Overview

This repository contains the implementation of a Transformer model for machine translation from English to Amharic. The model is based on the groundbreaking paper "Attention is All You Need" by Vaswani et al. (2017). The training process is implemented from scratch using PyTorch along with other essential libraries like tokenizers, tqdm, nltk, tensorboard, and torchmetrics.

## Requirements

Make sure you have the following packages installed:

- PyTorch
- Tokenizers
- tqdm
- NLTK
- Tensorboard
- TorchMetrics

You can install them using the following command:

```bash
pip install torch tokenizers tqdm nltk tensorboard torchmetrics
```

## Data

The dataset used for training the model is found in data folder in JSON file containing English to Amharic translation pairs. Each entry in the dataset has the following format:

```json
[
    { "en": "English text here", "am": "Amharic text here" },
    { "en": "Another English text", "am": "Another Amharic text" },
    ...
]
```


## Tokenizer Training

Train the tokenizer using Byte-Pair Encoding (BPE). This will create tokenizers for both the source (English) and target (Amharic) languages. Save the tokenizers for later use during model training.

```bash
python train_tokenizer.py
```

## Transformer Model Training

Train the Transformer model using the train.py script. Adjust the hyperparameters defined in config.py according to requirements. Monitor the training process using Tensorboard.

```bash
python train.py
```

## Inference with PyQt5 Interface

The inference script is found on inference.py. Here it loads trained model and tokenizers and uses the model to translate English sentences into Amharic with a simple PyQt5 interface.

```bash
python inference.py
```

Running the inference script will launch a PyQt5 interface where you can input English sentences and get the corresponding Amharic translations.