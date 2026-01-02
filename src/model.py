""" Transformer Model Implementation for SHELDON-T: Contextual Dialogue Humor Detection in Big Bang Theory S1-S5.

    Author: Andrew Chung
"""

import math
import json
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

class SitcomDataset(Dataset):
    """ Wrapper Class for BBT dialogue data (.jsonl)
        Important for generating segment embeddings for scene/prior/target
    """

    def __init__(self, data_path, tokenizer, max_len):
        self.data = []
        with open(data_path, 'r', encoding = 'utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_len = max_len

        # define special token IDs
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
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
        max_context_len = self.max_len - static_len

        # Case 1: scene + target exceeds max_len on their own
        if max_context_len < 0:
            # scene_ids = scene_ids[:self.max_len - len(target_ids)]
            scene_ids = scene_ids[:self.max_len - len(target_ids) - 1]
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
        padding_len = self.max_len - len(input_ids)
        if padding_len > 0:
            input_ids = input_ids + ([self.pad_id] * padding_len)
            segment_ids = segment_ids + ([0] * padding_len)
            attention_mask = attention_mask + ([0] * padding_len)

        return {
            'input_ids': input_ids,
            'segment_ids': segment_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(item['label'], dtype = torch.float32)
        }

class DialogueEmbedding(nn.Module):
    """ Custom Embedding Layer for Dialogue Corpus
    """

    def __init__(self, max_len, d_model, tokenizer, dropout = 0.1):

        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.tokenizer = tokenizer # pre-trained custom tokenizer

        # embedding layers (QUQ, lookup tables)
        self.word_embedding_layer = nn.Embedding(num_embeddings = len(self.tokenizer), embedding_dim = self.d_model)
        self.segment_embedding_layer = nn.Embedding(num_embeddings = 3, embedding_dim = self.d_model)

        # Layernorm and Dropout
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Define Positional embeddings (static)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        emb_index = torch.arange(0, d_model, 2).float()
        div = torch.pow(10000, -emb_index / d_model)

        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, batch):
        """ Params
        --- batch: a DataLoader-configured batch of dialogue line data points (from SitcomDataset).
        """

        input_ids = batch['input_ids']
        segment_ids = batch['segment_ids']
        attention_mask = batch['attention_mask']

        # get max sequence token length for batch-wise mask trimming
        max_batch_length = attention_mask.sum(dim = 1).max().item()
        if max_batch_length > self.max_len: # safety net if sequence length exceeds max_len (unlikely)
            print(f'Error: Max Batch Length ({max_batch_length}) > Max Token Count ({self.max_len})')
            input_ids = input_ids[:, :self.max_len]
            segment_ids = segment_ids[:, :self.max_len]
            attention_mask = attention_mask[:, :self.max_len]
        else:
            input_ids = input_ids[:, :max_batch_length]
            segment_ids = segment_ids[:, :max_batch_length]
            attention_mask = attention_mask[:, :max_batch_length]

        # generate word embedding tensor
        word_embeddings = self.word_embedding_layer(input_ids)

        # generate segment embedding
        segment_embeddings = self.segment_embedding_layer(segment_ids)

        # sum embeddings then apply LayerNorm and Dropout
        embeddings = word_embeddings + \
            self.pe[:, :word_embeddings.size(1), :] + \
            segment_embeddings
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings, attention_mask
    
class DialogueAttentionHead(nn.Module):
    """ Attention Head within Encoder Block
    """

    def __init__(self, d_model, n_heads, dropout):

        super().__init__()
        assert d_model % n_heads == 0, \
            f'Error: d_model ({d_model}) not divisible by n_heads ({n_heads})'
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads
        self.dropout = dropout

        # vectorized weight matrices (d_head * n_heads = d_model
        self.w_q = nn.Linear(self.d_model, self.d_model, bias = False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias = False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias = False)
        self.scale = 1 / math.sqrt(self.d_head)
    
    def forward(self, embedding, attention_mask):
        assert embedding.shape[-1] == self.d_model, \
            f'Error: embedding dimension ({embedding.shape[-1]}) does not match d_model = {self.d_model}'
        
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        MASK = (1 - attention_mask.float()) * -1e9

        batch_size, seq_len = embedding.shape[:1]

        # vectorized Q, K, V matrices
        Q = self.w_q(embedding).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.w_k(embedding).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.w_v(embedding).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Return self-attention tensor (B * |S| * d_head)
        attention_weights = F.softmax((Q @ K.transpose(-2, -1)) * self.scale + MASK, dim = -1)
        attention_weights = F.dropout(attention_weights, p = self.dropout, training = self.training)
        attention = attention_weights @ V
        return attention.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

class DialogueEncoder(nn.Module):
    """ Transformer Encoder Block for Dialogue Embeddings
    """

    def __init__(self, d_model, n_heads, dropout = 0.1):

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.head = DialogueAttentionHead(self.d_model, self.n_heads, dropout = self.dropout)

        # output projection layer before residual
        self.output_projection = nn.Linear(d_model, d_model)

        # FFN after initial residual addition
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), 
            nn.GELU(), nn.Dropout(dropout), 
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout))
        
        # define dropout and layernorm modules
        self.dropout_ = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(self.d_model)
        self.layernorm2 = nn.LayerNorm(self.d_model)

    def forward(self, embedding, attention_mask):
        """ Params
        --- Embeddings: a 3D tensor of dimensions B * |S| * d,
            where B = batch size, |S| = max seq length, d = embedding dimension.
        """

        # concatenate self attention tensors, then apply linear activation + residual
        sa_linear = self.head(embedding, attention_mask)
        sa_output_projection = self.output_projection(sa_linear)
        sa_residual = self.dropout_(sa_output_projection) + embedding
        sa_residual = self.layernorm1(sa_residual)

        # FFN and secondary residual addition
        sa_ffn = self.ffn(sa_residual) + sa_residual
        output = self.layernorm2(sa_ffn)

        return output

class SheldonTransformer(nn.Module):

    def __init__(self, max_len = 512, d_model = 256, n_heads = 4,  dropout = 0.1, tokenizer = None):

        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.tokenizer = tokenizer

        self.embedder = DialogueEmbedding(self.max_len, self.d_model, self.tokenizer, self.dropout)

        # status quo architecture employs 2 stacked encoders
        # in the future, I can customize the number of encoder layers using nn.ModuleList
        self.encoder1 = DialogueEncoder(self.d_model, self.n_heads, dropout = self.dropout)
        self.encoder2 = DialogueEncoder(self.d_model, self.n_heads, dropout = self.dropout)

        self.prediction_head = nn.Linear(d_model, 1)

    def forward(self, batch):

        # Transformer Encoder Block
        embedding, attention_mask = self.embedder(batch)
        output1 = self.encoder1(embedding, attention_mask)
        output2 = self.encoder2(output1, attention_mask)

        # prediction head
        output = self.prediction_head(output2[:, 0, :]) # skim [CLS]
        return output
