import os
from uuid import uuid4 as uuid

TRAINING_ID = str(uuid())

DATA_PATH = 'data'
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
TEST_PATH = os.path.join(DATA_PATH, 'test')
# paths 

IMAGE_DIM = 128
PROCESS_CHANNELS = 32
# model configuration 

LOG_FILE = os.path.join('logs', TRAINING_ID+".log")
CHECKPOINT_DIR = os.path.join('checkpoints', TRAINING_ID)
CHECKPOINT_EVERY = 10
# checkpoint every 10 epochs at the checkpoint directory

DEVICE = 'cuda'
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 16
EPOCHS = 200
LR = 1e-3