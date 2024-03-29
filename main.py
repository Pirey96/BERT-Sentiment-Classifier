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
import joblib


# # # GPU SETUP # # #
import torch.cuda
#import tensorflow as tf
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.__version__)
#tf.config.experimental.list_physical_devices('GPU')
device = torch.device("cuda")
#torch.cuda.empty_cache()

# # # GPU SETUP END # # #

BATCH_SIZE = 4
MAX_LEN = 170
EPOCHS = 8
# the possible classification
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

def create_loader( df, batch_size):
    # # loading the dataset (transforming them at the same time)
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
    model = model.to(device)    # #!!!!!NOT WORKING
    # #######################################training the anger model
    optimizer = AdamW(model.parameters(),
                      lr=1e-5)  # optimizer as per the bert paper (may be more calibrated)
    df = create_df(dataset_type)
    df_train_set, df_test_set = create_split_dataset(df)

    total_training_steps = len(df_test_set) * EPOCHS  # length of total training data loader
    loss_funct = nn.CrossEntropyLoss().to(device)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_training_steps
    )
    trained_models = defaultdict(list)
    f1_score = 0.0
    for epochs in range(EPOCHS):
        #df_train_set = shuffle(df_train_set)
        training = Training(model,
                            create_loader(df_train_set, BATCH_SIZE),
                            create_loader(df_test_set, BATCH_SIZE),
                            loss_funct,
                            optimizer,
                            scheduler,
                            len(df_train_set),
                            len(df_test_set),
                            )
        print(f'Epoch {epochs+1}/{EPOCHS}')
        print('-'*100)
        train_f1, train_loss = training.training_model(device)
        print(f'Train loss {train_loss} f1-Score {train_f1}')

        test_f1, test_loss = training.testing_model(device)
        print(f'test loss {test_loss} f1-Score {test_f1}')
        trained_models['f1-score'].append(test_f1)



         #Saving the best Models
        if test_f1 > f1_score:
            f1_score = test_f1  #new best
            SAVEFILE = dataset_type +".bin"
            torch.save(model.state_dict(), SAVEFILE)    # saving the model in a .bin file
            print(f'NEW SAVE AT EPOCH: {epochs+1} at FILE: {SAVEFILE}')


        # Exporting the best model to a binary file





def Start ():
# #for some reason pytorch and windows causes an error with the sentiment dataset
    if __name__ == '__main__':
        #training('joy')       # --DONE TRAINING
        #training('anger')     # --DONE TRAINING
        #training('fear')      # --DONE TRAINING
        #training('sadness')   # --DONE TRAINING



        while(1):
            input_text = Input()
            input_text.input_text()



Start()











# # RESERVED FOR DEBUG PURPOSES
def debug():
    if __name__ == '__main__':
        df = create_df("anger")

        print(len(create_loader(df, BATCH_SIZE)))


#debug()