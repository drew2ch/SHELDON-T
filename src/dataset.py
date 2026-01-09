""" Custom Sitcom Data Set Wrapper Script.
    Author: Andrew Chung
"""

import json
import torch
from torch.utils.data import Dataset

class SitcomDataset(Dataset):
    """ Wrapper Class for BBT dialogue data (.jsonl)
        Important for generating segment embeddings for scene/prior/target
    """

    def __init__(self, data, tokenizer, maxt):
        self.data = []
        self.tokenizer = tokenizer
        self.maxt = maxt

        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

        if isinstance(data, str): # passed .jsonl path
            with open(data, 'r', encoding = 'utf-8') as f:
                items = (json.loads(line) for line in f)
        else: # directly passed dict iterable
            items = data

        for item in items:
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
