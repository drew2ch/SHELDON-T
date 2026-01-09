import os
import torch
from typing import List, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, model_validator
from transformers import PreTrainedTokenizerFast
from model import SheldonTransformer

MODEL_DIR = './models'
MAX_TOKEN_LENGTH = 512
D_MODEL = 128
HEAD_COUNT = 2
DROPOUT = 0.1

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f'Loading Models. This may take a second.')
    TOK_DT4 = PreTrainedTokenizerFast(
    tokenizer_file = os.path.join(MODEL_DIR, 'tokenizer_dt4.json'),
    cls_token = '[CLS]', sep_token = '[SEP]', pad_token = '[PAD]', unk_token = '[UNK]', mask_token = '[MASK]')
    TOK_DT6 = PreTrainedTokenizerFast(
        tokenizer_file = os.path.join(MODEL_DIR, 'tokenizer_dt6.json'),
        cls_token = '[CLS]', sep_token = '[SEP]', pad_token = '[PAD]', unk_token = '[UNK]', mask_token = '[MASK]')
    DT4_MODEL = SheldonTransformer(
            maxt = MAX_TOKEN_LENGTH, d_model = D_MODEL, n_heads = HEAD_COUNT, 
            dropout = DROPOUT, tokenizer = TOK_DT4)
    DT6_MODEL = SheldonTransformer(
            maxt = MAX_TOKEN_LENGTH, d_model = D_MODEL, n_heads = HEAD_COUNT, 
            dropout = DROPOUT, tokenizer = TOK_DT6)

    DT4_STATE_DICT = torch.load(os.path.join(MODEL_DIR, 'SHELDONT-DT4.pt'), map_location = 'cpu')
    DT6_STATE_DICT = torch.load(os.path.join(MODEL_DIR, 'SHELDONT-DT6.pt'), map_location = 'cpu')
    DT4_MODEL.load_state_dict(DT4_STATE_DICT['model_state'])
    DT6_MODEL.load_state_dict(DT6_STATE_DICT['model_state'])

    app.state.tok4 = TOK_DT4
    app.state.tok6 = TOK_DT6
    app.state.dt4 = DT4_MODEL
    app.state.dt6 = DT6_MODEL

    print(f'Models successfully loaded.')
    yield
    print(f'Application Terminated. Cleaning up...')

APP = FastAPI(lifespan = lifespan)

class DialogueLine(BaseModel):
    text: str
    speaker: str
    recipients: List[str]

class SitcomAnalysisRequest(BaseModel):
    dt: int
    scene: str
    context: List[DialogueLine]
    target: DialogueLine

    @model_validator(mode = 'after')
    def validate(self):
        if self.dt not in (4, 6):
            raise ValueError(f'Illegal DT: {self.dt}. Only DT = 4 and DT = 6 are permitted.')
        n_lines = len(self.context) + 1
        if self.dt != n_lines:
            raise ValueError(f'Error: DT ({self.dt}) does not match dialogue line count: {n_lines}')
        return self
    
def format_line(line: DialogueLine) -> str:
    recipients = ', '.join(line.recipients) if line.recipients else 'Self'
    return f'({line.speaker} to {recipients}) {line.text}'
    
def transform_input(payload: SitcomAnalysisRequest, 
                    tokenizer: PreTrainedTokenizerFast, maxt: int) -> Dict[str, torch.Tensor]:
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    scene_ids = tokenizer.encode('[CLS] ' + payload.scene.strip(), add_special_tokens = False)
    target_ids = tokenizer.encode(format_line(payload.target).strip() + ' [SEP]', add_special_tokens = False)
    context_ids = []
    for line in payload.context:
        context_ids.extend(tokenizer.encode(format_line(line).strip(), add_special_tokens = False) + [sep_id])
    
    # safety net: discard context tokens if maximum token count is exceeded
    static_len = len(scene_ids) + len(target_ids) + 1
    max_context_len = maxt - static_len

    # Case 1: scene + target exceeds maxt on their own
    if max_context_len < 0:
        # scene_ids = scene_ids[:self.maxt - len(target_ids)]
        scene_ids = scene_ids[:maxt - len(target_ids) - 1]
        context_ids = []
    # Case 2: sequence exceeds max capacity with full context corpus
    elif len(context_ids) > max_context_len:
        context_ids = context_ids[-max_context_len:]
        if context_ids[0] == sep_id: # dangling [SEP]
            context_ids = context_ids[1:]

    # re-format input IDs by token sequence
    input_ids = (scene_ids + [sep_id] + context_ids + target_ids)

    # construct segment IDs: scene 0, context 1, target 2
    segment_ids = (
        [0] * (len(scene_ids) + 1) +
        [1] * len(context_ids) +
        [2] * len(target_ids)
    )

    # pad and generate attention mask
    attention_mask = [1] * len(input_ids)
    padding_len = maxt - len(input_ids)
    if padding_len > 0:
        input_ids = input_ids + ([pad_id] * padding_len)
        segment_ids = segment_ids + ([0] * padding_len)
        attention_mask = attention_mask + ([0] * padding_len)

    return {
        'input_ids': torch.tensor(input_ids, dtype = torch.long).unsqueeze(0),
        'segment_ids': torch.tensor(segment_ids, dtype = torch.long).unsqueeze(0),
        'attention_mask': torch.tensor(attention_mask, dtype = torch.long).unsqueeze(0)}

@APP.post('/Predict')
async def predict(payload: SitcomAnalysisRequest, request: Request):
    
    if payload.dt == 4: 
        MODEL = request.app.state.dt4
        TOK = request.app.state.tok4
    elif payload.dt == 6: 
        MODEL = request.app.state.dt6
        TOK = request.app.state.tok6
    else: raise HTTPException(status_code = 400, detail = 'Invalid DT')

    try:
        DATA = transform_input(payload = payload, tokenizer = TOK, maxt = MAX_TOKEN_LENGTH)
    except Exception as e:
        raise HTTPException(status_code = 400, detail = f'Preprocessing failed: {e}')
    
    with torch.no_grad():
        output = MODEL(DATA)
        pred = (output > 0).long()
    
    return {
        'status': 'success',
        'model': f'DT{payload.dt}',
        'output': pred.tolist()
    }
