""" Transformer Model Implementation for SHELDON-T: Contextual Dialogue Humor Detection in Big Bang Theory S1-S5.

    Author: Andrew Chung
"""

import math
import torch
import torch.nn.functional as F
from torch import nn

class DialogueEmbedding(nn.Module):
    """ Custom Embedding Layer for Dialogue Corpus
    """

    def __init__(self, maxt, d_model, tokenizer, dropout = 0.1):

        super().__init__()
        self.maxt = maxt
        self.d_model = d_model
        self.tokenizer = tokenizer # pre-trained custom tokenizer

        # embedding layers (QUQ, lookup tables)
        self.word_embedding_layer = nn.Embedding(num_embeddings = len(self.tokenizer), embedding_dim = self.d_model)
        self.segment_embedding_layer = nn.Embedding(num_embeddings = 3, embedding_dim = self.d_model)

        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(maxt, d_model)
        position = torch.arange(0, maxt).unsqueeze(1).float()
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
        if max_batch_length > self.maxt: # safety net if sequence length exceeds maxt (unlikely)
            print(f'Error: Max Batch Length ({max_batch_length}) > Max Token Count ({self.maxt})')
            input_ids = input_ids[:, :self.maxt]
            segment_ids = segment_ids[:, :self.maxt]
            attention_mask = attention_mask[:, :self.maxt]
        else:
            input_ids = input_ids[:, :max_batch_length]
            segment_ids = segment_ids[:, :max_batch_length]
            attention_mask = attention_mask[:, :max_batch_length]

        word_embeddings = self.word_embedding_layer(input_ids)
        segment_embeddings = self.segment_embedding_layer(segment_ids)

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

        batch_size, seq_len = embedding.shape[:2]

        # vectorized Q, K, V matrices
        Q = self.w_q(embedding).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.w_k(embedding).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.w_v(embedding).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

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

        self.output_projection = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), 
            nn.GELU(), nn.Dropout(dropout), 
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout))
        
        self.dropout_ = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(self.d_model)
        self.layernorm2 = nn.LayerNorm(self.d_model)

    def forward(self, embedding, attention_mask):
        """ Params
        --- Embeddings: a 3D tensor of dimensions B * |S| * d,
            where B = batch size, |S| = max seq length, d = embedding dimension.
        """

        norm_embedding = self.layernorm1(embedding)
        sa_linear = self.head(norm_embedding, attention_mask)
        sa_output_projection = self.output_projection(sa_linear)
        sa_residual = self.dropout_(sa_output_projection) + embedding

        norm_residual = self.layernorm2(sa_residual)
        output = self.ffn(norm_residual) + sa_residual

        return output

class SheldonTransformer(nn.Module):

    def __init__(self, maxt = 512, d_model = 128, n_heads = 2,  dropout = 0.1, tokenizer = None):

        super().__init__()
        self.maxt = maxt
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.tokenizer = tokenizer

        self.embedder = DialogueEmbedding(self.maxt, self.d_model, self.tokenizer, self.dropout)

        # status quo (my) architecture employs a single encoder
        # in the future, I can customize the number of encoder layers using nn.ModuleList
        self.encoder = DialogueEncoder(self.d_model, self.n_heads, dropout = self.dropout)
        # self.encoder1 = DialogueEncoder(self.d_model, self.n_heads, dropout = self.dropout)
        # self.encoder2 = DialogueEncoder(self.d_model, self.n_heads, dropout = self.dropout)

        self.prediction_head = nn.Linear(d_model, 1)

    def forward(self, batch):

        embedding, attention_mask = self.embedder(batch)
        # output1 = self.encoder1(embedding, attention_mask)
        # output2 = self.encoder2(output1, attention_mask)
        output_ = self.encoder(embedding, attention_mask)

        output = self.prediction_head(output_[:, 0, :]) # skim [CLS]
        return output
