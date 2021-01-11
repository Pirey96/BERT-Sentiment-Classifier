import numpy as np
import transformers
import pandas as pd
import torch

########################PREPROCESSING AND PREPERATION OF DATA####################################################
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
def dataset():
    df = pd.readcsv()

def process_text(text):
    tokens = tokenizer.tokenize(text)
    print ((tokens))
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print (token_ids)
    #special tokens [SEP] 102 [CLS] 101 [PAD] 0
    #encoding behaves like
    encoding = tokenizer.encode_plus(
        text,
        max_length=512,
        add_special_tokens=True,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
        return_token_type_ids=False,
        return_tensors='pt'
    )
    print(encoding['input_ids'])
    print(len(encoding['input_ids'][0]))


def input_():
    while (1):
        text = input('Input text: ')
        process_text(text)

#input_()
df = pd.read_csv("Dataset.csv")
print(df.Intensity)
