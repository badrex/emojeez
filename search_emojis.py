#!/usr/bin/env python3
import numpy as np
from collections import defaultdict, Counter
import pickle
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

import emoji as em 
from sentence_transformers import SentenceTransformer
from qdrant_client import models, QdrantClient

# supress warnings coming from Hugging Face library
import warnings
warnings.filterwarnings('ignore')

# read emoji dictionary from desk
with open('emoji_embeddings_dict.pkl', 'rb') as file:
    emoji_dict = pickle.load(file)

# initialize sentence encoder
embedding_model = 'paraphrase-multilingual-MiniLM-L12-v2'
sentence_encoder = SentenceTransformer(embedding_model)

# make a new dict for embeddings, delete embedding key from old dict
# deletion is importatn becauase Qdrant cannot handle np.array payloods 
embedding_dict:Dict[str, np.array] = {}

for emoji in emoji_dict:
    embedding_dict[emoji] = np.array(emoji_dict[emoji]['embedding'])
    del emoji_dict[emoji]['embedding']

embedding_dim = embedding_dict[emoji].shape[0]

# initialize vector database client
vector_DB_client = QdrantClient(":memory:")


vector_DB_client.create_collection(
    collection_name="EMOJIS",
    vectors_config=models.VectorParams(
        size=embedding_dim,
        distance=models.Distance.COSINE,
    ),
)

# populate the collection with emojis and embeddings
vector_DB_client.upload_points(
    collection_name="EMOJIS",
    points=[
        models.PointStruct(
            id=idx, 
            vector=embedding_dict[emoji].tolist(),
            payload=emoji_dict[emoji]
        )
        for idx, emoji in enumerate(emoji_dict)
    ],
)


def return_simialr_emojis(query: str) -> None:
    """
    Return similar emojis to a given query
    
    Args:
        query (str): The query string to search for similar emojis.
        
    Returns:
        None
        
    """
    hits = vector_DB_client.search(
        collection_name="EMOJIS",
        query_vector=sentence_encoder.encode(query).tolist(),
        limit=40,
    )

    hit_emojis = set()

    for i, hit in enumerate(hits, start=1):
        emoji_char = hit.payload['Emoji']
        score = hit.score

        # to handle emojies with multiple byte characters
        _ord = ' '.join(str(ord(c)) for c in emoji_char)

        s = len(emoji_char) + 3
        emoji_desc = ' '.join(em.demojize(emoji_char).split('_'))[1:-1].upper()

        if emoji_char not in hit_emojis: 
            print(f"{emoji_char:<{s}} ", end='')
            #print(f"{i:<1} {emoji_char:<{s}}", end='')
            #print(f"{score:<7.3f}", end= '')
            #print(f"{emoji_desc:<55}")
            
        hit_emojis.add(emoji_char)


# return_simialr_emojis(
#     "innovation"
# ) # animal you can find in Australiaa

query = input("\nEnter a query: ")

while query:
    return_simialr_emojis(query)
    query = input("\nEnter a query: ")