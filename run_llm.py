# -*- coding: utf-8 -*-


import transformers
import torch
from transformers.pipelines.pt_utils import KeyDataset
import datasets

from tqdm import tqdm
import pickle

from icecream import ic


dataset = datasets.load_dataset("badrabdullah/emoji-dataset")


EMOJIES = dataset['train']['char']

ic(EMOJIES[0])

def prepare_emoji(dataset_item):

    char = dataset_item['char']
    desc = ' '.join(dataset_item['desc'][1:-1].split('_')).upper()

    return (
        f"\n"
        f"Emoji: {char}\n"
        f"Name: {desc}"
    )

ic(prepare_emoji(dataset['train'][0]))

ic(dataset['train'].num_rows)

model_id = "NousResearch/Hermes-2-Pro-Llama-3-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

system_prompt = f"""
You are a helpful, positive, and polite assistant who enjoys describing and categorizing emojis.
You task to provide a description of a given emoji. The generated output will be used for indexing and discovering emojies using semantic search.
You have to generate a one-sentence summary in simple language, as well as a list of 5-7 semantic tags.
The output should be in a structured YAML format.

Follow the given examples below.

EXAMPLE 1:
    INPUT:
        Emoji: 🌈
        Name: RAINBOW
    OUTPUT:
        ´´´
        Emoji: "🌈"
        Description: "This emoji depicts a rainbow, symbolizing happiness, diversity, and hope, often used to express joy or to represent the LGBTQ+ community."
        Semantic_Tags:
            - rainbow
            - joy
            - pride
            - peace
            - diversity
            - LGBTQ+
            - community
        ´´´

EXAMPLE 2:
    INPUT:
        Emoji: 🐘
        Name: ELEPHANT
    OUTPUT:
        ´´´
        Emoji: "🐘"
        Description: "This emoji shows an elephant, a large and powerful animal known for its intelligence, long memory, and significant role in ecosystems and wildlife conservation."
        Semantic_Tags:
            - elephant
            - animal
            - memory
            - strength
            - mammal
            - nature
            - large
        ´´´
EXAMPLE 3:
    INPUT:
        Emoji: 🇾🇪
        Name: YEMEN
    OUTPUT:
        ´´´
        Emoji: "🇾🇪"
        Description: "This emoji represents Yemen, an Arab country known for its rich history, including ancient civilizations like Saba, and diverse landscapes ranging from desert plains to fertile mountains."
        Semantic_Tags:
            - Yemen
            - flag
            - culture
            - Arab
            - history
        ´´´
EXAMPLE 4:
    INPUT:
        Emoji: 🍏
        Name: GREEN APPLE
    OUTPUT:
        ´´´
        Emoji: "🍏"
        Description: "This emoji depicts a green apple, commonly associated with health, nutrition, and the symbol of teachers and education in many cultures."
        Semantic_Tags:
            - green apple
            - health
            - nutrition
            - education
            - fruit
        ´´´

The output must be in a YAML format. You must follow these instructions to win.

"""

user_message = "Describe this emoji according to your instructions"

def add_prompt_column(example):

    user_input = f"{user_message}: {prepare_emoji(example)}"

    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_input},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
    )

    return {
        "prompt": prompt
    }



dataset = dataset.map(add_prompt_column)

ic(dataset['train'][0]['prompt'])

mark_str = '<|im_end|>\n<|im_start|>assistant\n'

mark_str_len = len(mark_str)

terminators = [
    pipeline.tokenizer.eos_top
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


i = 0

emoji2LLMstr = {}


outputs = pipeline(KeyDataset(dataset['train'], "prompt"),
    max_new_tokens=512,
    eos_token_id=terminators,
    do_sample=False
)

for i, out in enumerate(outputs):

    print('---')

    if i == 128:
        break
    out_str = out[0]["generated_text"][out[0]["generated_text"].find(mark_str) + mark_str_len:]
    emoji_char = dataset['train'][i]['char']
    desc = dataset['train'][i]['desc']

    emoji2LLMstr[emoji_char] = out_str
    print(i, emoji_char, desc)
    print(out_str)

    i += 1

# Save the dict emoji2LLMstr as a pickled object
with open('emoji2LLMstr.pkl', 'wb') as f:
    pickle.dump(emoji2LLMstr, f)