import numpy as np
import transformers
import pandas as pd
import DatasetTraining

sentiment = []
########################PREPROCESSING AND PREPERATION OF DATA####################################################
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')


def dataset():
    df = pd.read_csv('Dataset.csv')
    return df
def process_text(text):
    tokens = tokenizer.tokenize(text)
    print ((tokens))
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print (token_ids)
    #special tokens [SEP] 102 [CLS] 101 [PAD] 0
    #encoding behaves like
    encoding = tokenizer.encode_plus(
        text,                     #input text
        max_length=512,           #number of input words
        add_special_tokens=True,  #CES,CLS,SEP tokens
        padding='max_length',     #Padding to 512
        return_attention_mask=True, #attention mask for weights
        truncation=True,            #truncate to 512
        return_token_type_ids=False,  #sequence identification
        return_tensors='pt'
    )
    #print(encoding['input_ids'])
    #print(len(encoding['input_ids'][0]))
    #print(encoding['attention_mask'])

def input_():
    while (1):
        text = input('Input text: ')
        process_text(text)


DatasetTraining.train(dataset())
#input_()


