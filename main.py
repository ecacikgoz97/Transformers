import torch
import torch.nn as nn
import torch.optim as optim
import config
from transformer import Transformer
from train import training
from utils import bleu,load_checkpoint
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import time

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

# Data Processing
spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")

german  = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")

train_data, validation_data, test_data = Multi30k.splits(exts=(".de", ".en"), fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


SRC_VOCAB_SIZE = len(german.vocab)
TRGT_VOCAB_SIZE = len(english.vocab)
SRC_PAD_SIZE = english.vocab.stoi["<pad>"]

train_loader, validation_loader, test_loader = BucketIterator.splits((train_data, validation_data, test_data), batch_size=config.BATCH_SIZE, sort_within_batch=True, sort_key=lambda x: len(x.src), device=config.DEVICE)

model = Transformer(
    embedding_size    =config.EMBEDDING_SIZE,
    src_vocab_size    =SRC_VOCAB_SIZE,
    trg_vocab_size    =TRGT_VOCAB_SIZE,
    src_pad_idx       =SRC_PAD_SIZE,
    num_heads         =config.NUM_HEADS,
    num_encoder_layers=config.NUM_ENCODER_LAYERS,
    num_decoder_layers=config.NUM_DECODER_LAYERS,
    forward_expansion =config.FORWARD_EXPANSION,
    dropout           =config.DROPOUT_RATE,
    max_len           =config.MAX_LEN,
    device            =config.DEVICE,
).to(config.DEVICE)

optimizer = optim.Adam(model.parameters(), lr=config.LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if config.LOAD_MODEL:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

sentence = "ein pferd geht unter einer brücke neben einem boot."

start_time = time.time()
training(epochs=config.EPOCHS, model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler, train_loader=train_loader, sentence=sentence, german=german, english=english, save_model=config.SAVE_MODEL, device=config.DEVICE, writer=config.WRITER)
end_time = time.time()
print(f"Training time: {end_time-start_time}")

# running on entire test data takes a while
score = bleu(test_data[1:100], model, german, english, config.DEVICE)
print(f"Bleu score {score * 100:.2f}")
