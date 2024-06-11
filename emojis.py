# %%
#!/usr/bin/env python3
import sys
import unicodedata

# %%
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm

# %%
from typing import List, Dict, Any, Tuple, Union

# %%
# emoji_ranges = [
#     (0x1F600, 0x1F64F),  # Emoticons 
#     (0x1F300, 0x1F5FF),  # Miscellaneous Symbols and Pictographs
#     #(0x1F680, 0x1F6FF),  # Transport and Map Symbols
#     #(0x1F700, 0x1F77F),  # Alchemical Symbols
#     #(0x2600, 0x26FF),    # Miscellaneous Symbols
#     #(0x2700, 0x27BF),    # Dingbats
#     #(0x2B50, 0x2BFF),    # Additional symbols
#     (0x1F1E8, 0x1F1FC),   # Flags (iOS)
#     (0x1F1E6, 0x1F1FF),  # Regional Indicator Symbols for country flags
#     (0x1F680, 0x1F6FF),  # Transport and Map Symbols
#     (0x1F3F4, 0x1F3F4),  # Black flag, pirate flag, etc., in Miscellaneous Symbols and Pictographs
#     (0x1F900, 0x1F9FF)   # Supplemental Symbols and Pictographs for additional flag-related symbols
# ]

# def is_emoji(character):
#     return any(start <= ord(character) <= end for start, end in emoji_ranges)

# # %%

# START, END = ord(' '), sys.maxunicode + 1

# emojis = []
    
# for code in range(START, END):
#     char = chr(code)

#     name = unicodedata.name(char, None)

#     #print(f'U+{code:04X}\t{char}\t{name}')
#     if is_emoji(char) and name:
#         emojis.append(
#             {
#                 'code': ord(char),
#                 'char': char,
#                 'name': 'This is an emoji that represents a ' + name.lower() + '.'
#             }
#         )

# import unicodedata

def return_flag(country_code):
    assert len(country_code) == 2 and country_code.isalpha(), "Country code must be two alphabetical characters"
    base = 0x1F1E6
    flag = chr(base + ord(country_code[0]) - ord('A')) + chr(base + ord(country_code[1]) - ord('A'))
    
    print(flag)

def get_country_flags():
    flags = []
    base = 0x1F1E6  # Start of regional indicator symbols

    for code1 in range(base, base + 26):  # Loop over A to Z
        for code2 in range(base, base + 26):  # Loop over A to Z

            flag_char = chr(code1) + chr(code2)
            first, second = chr(code1 - base + 65), chr(code2 - base + 65)
            flag_name = f"Flag of {first}{second}"
            flag_emoji = return_flag(first.upper() + second.upper())

            

            #if flag_name == "Flag of us":
            print(flag_char, flag_emoji, flag_name)

            flags.append(
                {
                    'code': ord(flag_char[0]) * 0x10000 + ord(flag_char[1]), 
                    'char': flag_emoji, 
                    'name': 'This is an emoji that represents a ' + flag_name.lower() + '.'
                }
            )

    return flags



# # Example usage:
# print_flag('US')  # Outputs: 🇺🇸
# print_flag('DE')  # Outputs: 🇩🇪


def get_standard_emojis():
    emojis = []
    # Ranges of emojis (excluding country flags)
    emoji_ranges = [
        (0x1F600, 0x1F64F),  # Emoticons
        (0x1F300, 0x1F5FF),  # Miscellaneous Symbols and Pictographs
        (0x1F680, 0x1F6FF),  # Transport and Map Symbols
        #(0x1F700, 0x1F77F),  # Alchemical Symbols
        (0x2600, 0x26FF),    # Miscellaneous Symbols
        (0x2700, 0x27BF),    # Dingbats
        #(0x2B50, 0x2BFF),    # Additional symbols
    ]

    for start, end in emoji_ranges:
        for code in range(start, end + 1):
            try:
                char = chr(code)
                name = unicodedata.name(char)
                emojis.append(
                    {
                        'code': ord(char),
                        'char': char,
                        'name': 'This is an emoji that represents a ' + name.lower() + '.'
                    }
                )
            except ValueError:
                continue
    return emojis

#Combine all emojis and flags into one list
emojis = get_standard_emojis() + get_country_flags()

#print(emojis)

# Print the first few for demonstration
print(emojis[-10:])  # Print only the first 10 for brevity

# %%
from fastembed import TextEmbedding

# %%
supported_models = (
    pd.DataFrame(TextEmbedding.list_supported_models())
    .sort_values("size_in_GB")
    .drop(columns="sources")
    .reset_index(drop=True)
)
supported_models

	

# %%
supported_models.loc[9]['model']

# %%
# documents: List[str] = [
#     "passage: Hello, World!",
#     "query: Hello, World!",
#     "passage: This is an example passage.",
#     "fastembed is supported by and maintained by Qdrant."
# ]
# embedding_model = TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
# embeddings: List[np.ndarray] = list(embedding_model.embed(documents))


# %%
# np.array(embeddings).shape 

# %%
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer


# %%
encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# %%
client = QdrantClient(":memory:")


# %%
client.recreate_collection(
    collection_name="my_books",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=models.Distance.COSINE,
    ),
)


# %%



while True:
    query = str(input('Enter search query: '))
    query = 'This is an emoji that represents a ' + query.lower() + '.'
    str_len = len('This is an emoji that represents a ')

    if query:
        hits = client.search(
            collection_name="my_books",
            query_vector=encoder.encode(query).tolist(),
            limit=10,
        )

        for hit in hits:
            print(f"{hit.payload['char']:<10} {hex(hit.payload['code'])} {hit.payload['name'][str_len:-1]:>50} {hit.score:>5.2f}") #, "score:", hit.score) [str_len:-1][str_len:]
    else:
        break

# %%



# %%



