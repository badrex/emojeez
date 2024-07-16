#!/usr/bin/env python3
import numpy as np
from collections import defaultdict, Counter
import pickle
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import datasets

# this python module is need to get official emoji metadata
import emoji as em


# read emoji dictionary with LLM generated metadata from desk
with open('emoji_llm.pkl', 'rb') as file:
    emoji_dict: Dict[str, Dict[str, str]] = pickle.load(file)

# clean up the dictionary
for emoji in tqdm(emoji_dict):
    try:
        short_desc = ' '.join(em.demojize(emoji).split('_'))[1:-1].upper()
        emoji_dict[emoji]['character'] = emoji
        emoji_dict[emoji]['unicode'] = 'U+' + ' '.join(
            [str(hex(ord(char)))[2:].upper() for char in emoji]
        )

        emoji_dict[emoji]['short description'] = short_desc
        emoji_dict[emoji]['semantic tags'] = [
            str(t) for t in emoji_dict[emoji]['Semantic_Tags']
        ]
        emoji_dict[emoji]['LLM description'] = emoji_dict[emoji]['Description']


        del emoji_dict[emoji]['Emoji']
        del emoji_dict[emoji]['Description']
        del emoji_dict[emoji]['Semantic_Tags']


    # exception handling was added to avoid errors when reading dicts
    # TODO: add more specific exception handling    
    except Exception as e: 
        print(f"Error occurred for emoji: {emoji}. Error message: {str(e)}")
        
print(emoji_dict['😶‍🌫️'])

# this part is ugly, refactor later if possible 
emoji_chars = list(emoji_dict.keys())
emoji_unicodes = [emoji_dict[emoji]['unicode'] for emoji in emoji_chars]
emoji_tags = [emoji_dict[emoji]['semantic tags'] for emoji in emoji_chars]

emoji_short_desc = [
    emoji_dict[emoji]['short description'] for emoji in emoji_chars
]

emoji_llm_desc = [
    emoji_dict[emoji]['LLM description'] for emoji in emoji_chars
]

# create Hugging Face dataset
dataset = datasets.Dataset.from_dict(
    {
        'character': emoji_chars,
        'unicode': emoji_unicodes,
        'short description': emoji_short_desc,
        'tags': emoji_tags,
        'LLM description': emoji_llm_desc
    }
)

# push dataset to Hugging Face hub
dataset.push_to_hub('badrex/llm-emoji-dataset')


# save dataset to disk
dataset.save_to_disk('llm_emoji_dataset')

