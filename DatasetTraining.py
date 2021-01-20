import torch
import transformers
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig, data
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
BATCH_SIZE = 16
MAX_LEN = 128
EPOCHS = 8
class DatasetTraining(torch.utils.data.Dataset):
    #target needs to be modified
    def __init__(self, tweet, sentiment, max_len):
        self.tweet = tweet
        self.sentiment = sentiment
        self.max_len = max_len

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        tweet = str(self.tweet[item])
        encoding = tokenizer.encode_plus(
            tweet,
            max_length=self.max_len,
            add_special_tokens=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'sentiments': torch.tensor(self.sentiment[item], dtype=torch.long)
        }





def data_loader (dataset, max_len, batch_size):
    data_set = DatasetTraining (
        tweet=dataset.Tweet.to_numpy,
        sentiment=np.array(dataset[["Emotion", "Intensity"]]),
        max_len=max_len
    )
    return DataLoader(
        data_set,
        batch_size=batch_size,
        num_workers=4
    )

def train (Dataset):
    dataset_train, dataset_test = train_test_split(Dataset, test_size=0.2)
    train = data_loader(dataset_train,MAX_LEN,BATCH_SIZE)
    testing = data_loader(dataset_test,MAX_LEN,BATCH_SIZE)
    data = (iter(train))
    data.next()
    print (data.keys())