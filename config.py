import torch
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOAD_MODEL = False
SAVE_MODEL = True

# Training hyperparameters
EPOCHS = 10000
LR = 3e-4
BATCH_SIZE = 32

# Model hyperparameters
EMBEDDING_SIZE = 512
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROPOUT_RATE = 0.10
MAX_LEN = 100
FORWARD_EXPANSION = 4

# Tensorboard to get nice loss plot
WRITER = SummaryWriter("runs/loss_plot")
