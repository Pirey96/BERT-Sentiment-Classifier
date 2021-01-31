from collections import defaultdict

import torch
from torch import nn
from torch.cuda import device
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertModel
from SentimentDataset import SentimentDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from Input import Input
from Classifier import Classifier
from Training import Training


BATCH_SIZE = 16
MAX_LEN = 150
EPOCHS = 8
#the possible classification
labels = [0, 1, 2, 3]


def create_df(dataset):
    df = pd.read_csv(dataset +".csv")
    return df


def create_SentimentDataset(df, max_len):
    instance = SentimentDataset (df, max_len)
    return instance

def create_loader( df, batch_size):       ##loading the dataset (transforming them at the same time)
    return DataLoader(
        create_SentimentDataset(df, MAX_LEN),
        batch_size=batch_size,
        num_workers=4
    )

def create_split_dataset(df):
    #print(len(df))
    dataset_train, dataset_test = train_test_split(df, test_size=0.2)
    return [dataset_train, dataset_test]

train_anger = create_loader(create_split_dataset(create_df("anger"))[0], BATCH_SIZE)
train_joy = create_loader(create_split_dataset(create_df("joy"))[0], BATCH_SIZE)
train_sadness = create_loader(create_split_dataset(create_df("sadness"))[0], BATCH_SIZE)
train_fear = create_loader(create_split_dataset(create_df("fear"))[0], BATCH_SIZE)


def training(dataset_type):
    model = Classifier(len(labels))
    ##model = model.to(device)    ##!!!!!NOT WORKING
    ########################################training the anger model
    optimizer = AdamW(model.parameters(), lr=2e-5,
                      correct_bias=False)  # optimizer as per the bert paper (may be more calibrated)
    total_training_steps = len(create_df(dataset_type)) * EPOCHS  # length of total training data loader
    loss_funct = nn.CrossEntropyLoss()
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_training_steps
    )
    trained_models = defaultdict(list)
    accuracy = 0
    training = Training(model, create_loader(create_df(dataset_type), BATCH_SIZE), loss_funct, optimizer, scheduler, len(create_df(dataset_type)))
    for epochs in range(EPOCHS):
        print(f'Epoch {epochs+1}/{EPOCHS}')
        print ('-'*10)
        #train_accuracy, train_loss = training.training_model()
        #print (f'Train loss {train_loss} accuracy {train_accuracy}')
        val_acc, val_loss = training.evaluate()
        print (f'val loss {val_loss} accuracy {val_acc}')


def Start ():
##for some reason pytorch and windows causes an error with the sentiment dataset
    if __name__ == '__main__':


        while(1):
            input_text = Input()
            encoding = input_text.input_text()
            print(encoding)

def debug():
    if __name__ == '__main__':
        #training("anger")
        dataloadertrain = create_loader(create_df("anger"),BATCH_SIZE)
        data = next(iter(dataloadertrain))
        #lata.pop(1)
        model_class = Classifier(len(labels))


        input_ids=data['input_ids']
        attention_mask=data['attention_mask']
        print(input_ids.shape)
        print(attention_mask.shape)
        print(type(input_ids))
        print(model_class(input_ids, attention_mask))
        training ("anger")
        while(1):
            input_text = Input()
            encoding = input_text.input_text()
            print(encoding)


#Start()
debug()