from collections import defaultdict
from torch import nn
from torch.cuda import device
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from SentimentDataset import SentimentDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from Input import Input
from Classifier import Classifier
from Training import Training


BATCH_SIZE = 32
MAX_LEN = 170
EPOCHS = 1
#the possible classification
labels = [0, 1, 2, 3]

def format_df(df):
    tweet = df['Tweet'].tolist()
    formatted_tweet = []
    for line in tweet:
        tweet_tokens = line.split()
        temp = []
        for word in tweet_tokens:

            if word[0] != '@':
                temp.append(word)
        string =' '.join(temp)
        formatted_tweet.append(string)

    df = df.assign(Tweet=formatted_tweet)

    return df

def create_df(dataset):
    df = pd.read_csv(dataset +".csv")
    df = format_df(df)
    return df


def create_SentimentDataset(df, max_len):
    instance = SentimentDataset (df, max_len)
    return instance

def create_loader( df, batch_size):       ##loading the dataset (transforming them at the same time)
    #print (f'Tweet num: {df["Unnamed: 0"]} Intensity {df["Intensity"]} ')
    return DataLoader(
        create_SentimentDataset(df, MAX_LEN),
        batch_size=batch_size,
        num_workers=4
    )

def create_split_dataset(df):
    #print(len(df))
    dataset_train, dataset_test = train_test_split(df, test_size=0.2)
    return dataset_train, dataset_test



def training(dataset_type):
    model = Classifier(len(labels))
    ##model = model.to(device)    ##!!!!!NOT WORKING
    ########################################training the anger model
    optimizer = AdamW(model.parameters(),
                      lr=1e-5)  # optimizer as per the bert paper (may be more calibrated)
    df = create_df(dataset_type)
    df_train_set, df_test_set = create_split_dataset(df)

    total_training_steps = len(df_test_set) * EPOCHS  # length of total training data loader
    loss_funct = nn.CrossEntropyLoss()
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_training_steps
    )
    trained_models = defaultdict(list)
    accuracy = 0
    training = Training(model,
                        create_loader(df_train_set, BATCH_SIZE),
                        create_loader(df_test_set, BATCH_SIZE),
                        loss_funct,
                        optimizer,
                        scheduler,
                        len(df_train_set)
                        )

    for epochs in range(EPOCHS):
        print(f'Epoch {epochs+1}/{EPOCHS}')
        print ('-'*100)
        train_accuracy, train_loss = training.training_model()
        print (f'Train loss {train_loss} accuracy {train_accuracy}')

        val_acc, val_loss = training.evaluate()
        print (f'val loss {val_loss} accuracy {val_acc}')


def Start ():
##for some reason pytorch and windows causes an error with the sentiment dataset
    if __name__ == '__main__':
        training("anger")

        while(1):
            input_text = Input()
            encoding = input_text.input_text()
            print(encoding)


Start()











##RESERVED FOR DEBUG PURPOSES
def debug():
    if __name__ == '__main__':
        df = create_df("anger")
        print(len(create_loader(df, BATCH_SIZE)))


#debug()