""" Custom BERT-based tokenizer for Big Bang Theory Dialogue Corpus.
    Additionally, formats dialogue lines into comprehensive transformer-readable sequences.
=== Inputs ===
    --- dt: Dialogue Turn value to process (default: 2)
    --- file: JSON data file containing BBT dialogues
    Author: Andrew Chung
"""

import os
import re
import json
import argparse
from tokenizers import BertWordPieceTokenizer

DIALOGUE_REGEX = re.compile(r'Dialog Turns \d+')

def safe_strip(val):
    """ Helper function to safely strip string values. """
    if val is None: return ''
    return str(val).strip()

def format_attribute_dialogue(val):
    """ Helper function to format dialogue line with speaker and recipients.
        Format: (Speaker to Recipient1, Recipient2, ...) Dialogue Line
    """
    speaker = safe_strip(val.get('Speaker', 'Unknown'))
    recipients = val.get('Recipients', [])
    line = safe_strip(val.get('Dialog', ''))
    return f"({speaker} to {', '.join(recipients) if recipients else 'Self'}) {line}"

def get_training_corpus(dialogues):
    """ Generator function that yields unique dialogue lines from the dialogues dict.
    --- Global scene de-duplication
    --- Episode-Local dialogue de-duplication
    """
    seen_scenes, seen_dialogues = set(), set()
    current_av_id = None
    for _, dialogue in dialogues.items():

        av_id = dialogue.get('AV_ID', None)
        if av_id != current_av_id:
            seen_dialogues.clear()
            current_av_id = av_id

        # yield scene if unique
        scene = dialogue.get('Scene', '').strip()
        if scene:
            clean_scene = scene.lower()
            if clean_scene not in seen_scenes:
                seen_scenes.add(clean_scene)
                yield scene

        # yield unique dialogue lines
        turns = [k for k in dialogue.keys() if DIALOGUE_REGEX.match(k)]
        for turn in turns:
            line = format_attribute_dialogue(dialogue[turn])
            if line:
                clean_line = line.lower()
                if clean_line not in seen_dialogues:
                    seen_dialogues.add(clean_line)
                    yield line

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', type = int, default = 2, 
                        help = 'Dialogue Turn value (default: 2 -- 1 contextual, 1 line)')
    parser.add_argument('--file', type = str, nargs = '+', required = True, 
                        help = 'JSON data file containing BBT dialogues')
    args = parser.parse_args()

    # load all json data files (i.e. pertaining to a DT value)
    # simultaneously attribute speaker-recipient structures
    print(f'Loading dialogues for DT = {args.dt}...')
    files = {}
    try:
        for fi in args.file:
            assert fi.endswith('.json'), f'Error: File {fi} is not a .json file.'
            with open(fi, 'r', encoding = 'utf-8') as f:
                data = json.load(f)
                files[fi] = data
    except FileNotFoundError:
        print(f"Error: File {fi} not found.")
        return
    
    # generate transformer-readable sequences
    for name, file in files.items():
        DATA = []
        print(f'Generating sequences from {name}...')
        for _, dialogue in file.items():
            scene = safe_strip(dialogue.get('Scene', ''))
            turns = [k for k in dialogue.keys() if DIALOGUE_REGEX.match(k)]
            turns.sort(key = lambda x: int(x.split()[-1]))
            dialogue_lines = []
            for turn in turns:
                line = format_attribute_dialogue(dialogue[turn])
                dialogue_lines.append(line)
            
            # construct sequence
            sequence = f'[CLS] {scene} [SEP] ' + ' [SEP] '.join(dialogue_lines) + ' [SEP]'
            DATA.append({'text': sequence, 'label': int(dialogue.get('GT', 0))})
        
        with open(f'{name}l', 'w', encoding = 'utf-8') as f:
            for row in DATA:
                f.write(json.dumps(row) + '\n')
        print(f'Sequences generated and saved to {name}l.')

    # merge all dialogues into a single dict
    DIALOGUES = {}
    for file in files.values():
        DIALOGUES.update(file)
    print(f'Loaded and processed all .json files.')
    
    # initialize BERT WordPiece tokenizer
    print(f'Training BERT WordPiece tokenizer...')
    tokenizer = BertWordPieceTokenizer(clean_text = True, lowercase = True, strip_accents = True)
    SPECIAL_TOKENS = ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']

    # train tokenizer on constructed sequences
    corpus = get_training_corpus(DIALOGUES)
    tokenizer.train_from_iterator(
        corpus, vocab_size = 30000, 
        min_frequency = 2, show_progress = True,
        special_tokens = SPECIAL_TOKENS
    )

    # save trained tokenizer model
    print(f'BERT training complete. Saving trained tokenizer...')
    OUTPUT_DIR = f'./Dataset/Dataset/DT_{args.dt}/'
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR) # unlikely
    tokenizer.save(os.path.join(OUTPUT_DIR, 'tokenizer.json'))
    print(f'Tokenizer saved to {os.path.join(OUTPUT_DIR, "tokenizer.json")}.')

if __name__ == "__main__":
    main()
