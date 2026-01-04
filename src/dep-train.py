""" Deployment Training for SHELDON-T on DT = 4 and DT = 6
    Author: Andrew Chung
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import torch
from tqdm import tqdm
from torch import nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset, DataLoader
from transformers import PreTrainedTokenizerFast
from model import SheldonTransformer
from dataset import SitcomDataset

""" HARD CODED HYPERPARAMETERS FOR TRAINING VALIDITY
    IN CONCORDANCE WITH VALIDATION LOSS PERFORMANCE
--- Batch Size (-b) = 32
--- Epochs (-e) = 12
--- Embedding Dims (-d) = 128
--- Attention Head Count (-H) = 2
--- Max Token Length (--maxt) = 512
--- Dropout Prob (--dout) = 0.1
--- Learning Rate (--lr) = 1e-4
--- Weight Decay (--wd) = 0.01
--- Gradient Accumulation Steps (--gacc) = 4
"""

DEFAULT_DIR = './Dataset/Dataset/'
BATCH_SIZE = 32
EPOCHS = 12
D_MODEL = 128
HEAD_COUNT = 2
MAX_TOKEN_LENGTH = 512
DROPOUT = 0.1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRADIENT_ACC = 4

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', choices = [4, 6], required = True, type = int, help = 'Dialogue Turns (DT)')
    args = parser.parse_args()

    assert os.path.exists(DEFAULT_DIR), \
        f'Error: WD does not exist: {DEFAULT_DIR}'
    PWD = f'{DEFAULT_DIR}/DT_{args.t}'
    print(f'Importing Tokenizer...')
    tokenizer = PreTrainedTokenizerFast(tokenizer_file = os.path.join(PWD, 'tokenizer.json'),
        cls_token = '[CLS]', sep_token = '[SEP]', pad_token = '[PAD]', unk_token = '[UNK]', mask_token = '[MASK]')

    print(f'Importing and Consolidating Train/Validation Sets...')
    training = SitcomDataset(os.path.join(PWD, 'train.jsonl'), tokenizer = tokenizer, maxt = MAX_TOKEN_LENGTH)
    validation = SitcomDataset(os.path.join(PWD, 'val.jsonl'), tokenizer = tokenizer, maxt = MAX_TOKEN_LENGTH)
    LOADER = DataLoader(ConcatDataset([training, validation]), 
                        batch_size = BATCH_SIZE, shuffle = True, num_workers = 0, pin_memory = True)

    print(f'Loading SHELDON-T Transformer Model...')
    device = 'cpu'
    model = SheldonTransformer(
        maxt = MAX_TOKEN_LENGTH, d_model = D_MODEL, n_heads = HEAD_COUNT, 
        dropout = DROPOUT, tokenizer = tokenizer)
    model.to(device = device)

    optimizer = AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    writer = SummaryWriter(log_dir = f'runs/dep_DT{args.t}_BS{BATCH_SIZE}')

    print(f'Successfully Loaded SHELDON-T. Training...')
    os.makedirs('./dep-models', exist_ok = True)

    for epoch in range(EPOCHS):
        train_loss = 0.0
        loop = tqdm(LOADER, desc = f'Epoch {epoch+1}/{EPOCHS}', leave = False)
        model.train()
        for b, batch in enumerate(loop):
            batch = {k: v.to(device) for k, v in batch.items()}
            if b % GRADIENT_ACC == 0: optimizer.zero_grad()
            y = batch['label']
            output = model(batch)
            loss = criterion(output.squeeze(1), y) / GRADIENT_ACC
            loss.backward()
            if b % GRADIENT_ACC == GRADIENT_ACC - 1 or b == len(loop) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_loss += loss.item() * GRADIENT_ACC
        train_loss /= len(LOADER)

        print(f'Epoch {epoch+1}/{EPOCHS}: Train Loss {train_loss:.4f}')
        writer.add_scalar('TrainLoss', train_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()},
            f'dep-models/SHELDONT-DT{args.t}.pt')
    
    print(f'Training Complete for DT = {args.t}.')

if __name__ == '__main__':
    main()
