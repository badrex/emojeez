#!/usr/bin/env python3
import numpy as np
from collections import defaultdict, Counter
import pickle
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

# for debugging
#from icecream import ic

# read emoji dictionary from desk
with open('emoji_llm.pkl', 'rb') as file:
    emoji_dict: Dict[str, Dict[str, str]] = pickle.load(file)

# initialize sentence encoder
embedging_model = 'paraphrase-multilingual-MiniLM-L12-v2'
sentence_encoder = SentenceTransformer(embedging_model)

# make a full sentence description for each emoji
for emoji in tqdm(emoji_dict):
    try:
        emoji_dict[emoji]['LLM_description'] = emoji_dict[emoji]["Description"] + \
            ' This emoji is usually used in the contexts of: ' + \
            ', '.join(str(s) for s in emoji_dict[emoji]["Semantic_Tags"][:-1]) + \
            ', and ' + \
            str(emoji_dict[emoji]["Semantic_Tags"][-1]) + '.'
        
    except Exception as e: 
        print(e)
        

#print(emoji_dict['🧬'])

# generate sentence embedding for each emoji 
vector_dict:Dict[str, np.array] = {}

for emoji in tqdm(emoji_dict):
    vector_dict[emoji] = sentence_encoder.encode(
        emoji_dict[emoji]['LLM_description']
    )

#print(vector_dict['🧬'])

emoji_embeddings_dict: Dict[str, Dict[str, str]] = {
    emoji: {
        **emoji_dict[emoji],
        "embedding": vector_dict[emoji]
    }
    for emoji in emoji_dict
}

#print(emoji_embeddings_dict['🧬'])

with open('emoji_embeddings_dict.pkl', 'wb') as file:
    pickle.dump(emoji_embeddings_dict, file)

