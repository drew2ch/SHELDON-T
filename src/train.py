""" SHELDON-T Model Training Regimen

    Author: Andrew Chung
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import argparse
import torch
from tqdm import tqdm
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from transformers import PreTrainedTokenizerFast
from model import SheldonTransformer

class SitcomDataset(Dataset):
    """ Wrapper Class for BBT dialogue data (.jsonl)
        Important for generating segment embeddings for scene/prior/target
    """

    def __init__(self, data_path, tokenizer, maxt):
        self.data = []
        self.tokenizer = tokenizer
        self.maxt = maxt

        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

        with open(data_path, 'r', encoding = 'utf-8') as f:
            for line in f:
                item = json.loads(line)
                sequence = item['text']

                # compile tokenized IDs for scene, context, and target
                segments = sequence.strip().split(' [SEP] ')
                scene_ids = self.tokenizer.encode(segments.pop(0).strip(), add_special_tokens = False)
                target_ids = self.tokenizer.encode(segments.pop().strip(), add_special_tokens = False) # last element should include terminal [SEP]
                context_ids = []
                for turn in segments: # all remaining segments belong to prior dialogue turns
                    turn_ids = self.tokenizer.encode(turn.strip(), add_special_tokens = False)
                    context_ids += turn_ids + [self.sep_id]

                # safety net: discard context tokens if maximum token count is exceeded
                static_len = len(scene_ids) + len(target_ids) + 1
                max_context_len = self.maxt - static_len

                # Case 1: scene + target exceeds maxt on their own
                if max_context_len < 0:
                    # scene_ids = scene_ids[:self.maxt - len(target_ids)]
                    scene_ids = scene_ids[:self.maxt - len(target_ids) - 1]
                    context_ids = []
                # Case 2: sequence exceeds max capacity with full context corpus
                elif len(context_ids) > max_context_len:
                    context_ids = context_ids[-max_context_len:]
                    if context_ids[0] == self.sep_id: # dangling [SEP]
                        context_ids = context_ids[1:]

                # re-format input IDs by token sequence
                input_ids = (scene_ids + [self.sep_id] + context_ids + target_ids)

                # construct segment IDs: scene 0, context 1, target 2
                segment_ids = (
                    [0] * (len(scene_ids) + 1) +
                    [1] * len(context_ids) +
                    [2] * len(target_ids)
                )

                # pad and generate attention mask
                attention_mask = [1] * len(input_ids)
                padding_len = self.maxt - len(input_ids)
                if padding_len > 0:
                    input_ids = input_ids + ([self.pad_id] * padding_len)
                    segment_ids = segment_ids + ([0] * padding_len)
                    attention_mask = attention_mask + ([0] * padding_len)

                self.data.append({
                    'input_ids': torch.tensor(input_ids, dtype = torch.long),
                    'segment_ids': torch.tensor(segment_ids, dtype = torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype = torch.long),
                    'label': torch.tensor(item['label'], dtype = torch.float32)})

    def __len__(self): return len(self.data)
    def __getitem__(self, index): return self.data[index]

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
                os.path.join(PWD, f'best_model_dt{args.t}.pt'))
        else: counter += 1
        if counter >= patience:
            print(f'Early Stopping: Epoch {epoch + 1}')
            break
    
    print(f'Training Complete for DT = {args.t}.')

if __name__ == '__main__':
    main()
