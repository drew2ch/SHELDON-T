""" SHELDON-T Model Training Regimen
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from transformers import PreTrainedTokenizerFast
from model import SheldonTransformer
from dataset import SitcomDataset

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', required = True, type = int, help = 'Dialogue Turns')
    parser.add_argument('-b', default = 16, type = int, help = 'Training Batch Size')
    parser.add_argument('-e', default = 50, type = int, help = 'Training Epoch Size')
    parser.add_argument('-d', default = 128, type = int, help = 'Model Embedding Dimensions')
    parser.add_argument('-H', default = 2, type = int, help = 'Number of Attention Heads')
    parser.add_argument('--dir', default = './Dataset/Dataset/', type = str, help = 'Working Directory for Data and Tokenizer')
    parser.add_argument('--maxt', default = 512, type = int, help = 'Max Sequence (Token) Length')
    parser.add_argument('--dout', default = 0.1, type = float, help = 'Dropout Probability')
    parser.add_argument('--lr', default = 1e-4, type = float, help = 'Optimizer Learning Rate')
    parser.add_argument('--wd', default = 0.01, type = float, help = 'Optimizer Weight Decay')
    parser.add_argument('--gacc', default = 4, type = int, help = 'Gradient Accumulation (n*b)')
    args = parser.parse_args()

    assert os.path.exists(args.dir), \
        f'Error: WD does not exist: {args.dir}'
    PWD = f'{args.dir}/DT_{args.t}'
    print(f'Importing Tokenizer...')
    tokenizer = PreTrainedTokenizerFast(tokenizer_file = os.path.join(PWD, 'tokenizer.json'),
        cls_token = '[CLS]', sep_token = '[SEP]', pad_token = '[PAD]', unk_token = '[UNK]', mask_token = '[MASK]')

    print(f'Importing Train and Validation Sets...')
    training = SitcomDataset(os.path.join(PWD, 'train.jsonl'), tokenizer = tokenizer, maxt = args.maxt)
    validation = SitcomDataset(os.path.join(PWD, 'val.jsonl'), tokenizer = tokenizer, maxt = args.maxt)
    T_LOADER = DataLoader(training, batch_size = args.b, shuffle = True, num_workers = 0, pin_memory = True)
    V_LOADER = DataLoader(validation, batch_size = args.b * 2, shuffle = False, num_workers = 0, pin_memory = True)

    print(f'Loading SHELDON-T Transformer Model...')
    device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SheldonTransformer(
        maxt = args.maxt, d_model = args.d, n_heads = args.H, 
        dropout = args.dout, tokenizer = tokenizer)
    model.to(device = device)

    optimizer = AdamW(model.parameters(), lr = args.lr, weight_decay = args.wd)
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', patience = 3, factor = 0.5)
    criterion = nn.BCEWithLogitsLoss() # logit-to-binary loss comparison
    writer = SummaryWriter(log_dir = f'runs/exp_DT{args.t}_BS{args.b}')
    scaler = GradScaler()

    best_val_loss = float('inf')
    patience = 4
    counter = 0
    print(f'Successfully Loaded SHELDON-T. Training...')
    os.makedirs('./models', exist_ok = True)

    for epoch in range(args.e):
        train_loss = 0.0
        loop = tqdm(T_LOADER, desc = f'Epoch {epoch+1}/{args.e}', leave = False)
        model.train()
        for b, batch in enumerate(loop):
            batch = {k: v.to(device) for k, v in batch.items()}
            if b % args.gacc == 0: optimizer.zero_grad()
            y = batch['label']
            with autocast('cuda'):
                output = model(batch)
                loss = criterion(output.squeeze(1), y)
            scaler.scale(loss / args.gacc).backward()
            if b % args.gacc == args.gacc - 1 or b == len(loop) - 1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            train_loss += loss.item()
        train_loss /= len(T_LOADER)
        
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in V_LOADER:
                batch = {k: v.to(device) for k, v in batch.items()}
                y = batch['label']
                with autocast('cuda'):
                    output = model(batch)
                    loss = criterion(output.squeeze(), y)
                val_loss += loss.item()
        val_loss /= len(V_LOADER)

        print(f'Epoch {epoch+1}/{args.e}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}')
        writer.add_scalar('TrainLoss', train_loss, epoch)
        writer.add_scalar('ValLoss', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss}, 
                f'models/best_model_dt{args.t}_b{args.b}.pt')
        else: counter += 1
        if counter >= patience:
            print(f'Early Stopping: Epoch {epoch + 1}')
            break
    
    print(f'Training Complete for DT = {args.t}.')

if __name__ == '__main__':
    main()
