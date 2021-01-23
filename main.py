from torch.utils.data import DataLoader
from SentimentDataset import SentimentDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from Input import Input
BATCH_SIZE = 1
MAX_LEN = 128
EPOCHS = 8

def create_df():
    df = pd.read_csv('Dataset.csv')
    return df
def create_SentimentDataset(df, max_len):
    instance = SentimentDataset (df, max_len)
    return instance
def create_loader( df, batch_size):
    return DataLoader(
        create_SentimentDataset(df, MAX_LEN),
        batch_size=batch_size,
        num_workers=4
    )

def create_split_dataset(df):
        dataset_train, dataset_test = train_test_split(df, test_size=0.2)
        return [dataset_train, dataset_test]


train = create_loader(create_split_dataset(create_df())[0],BATCH_SIZE)

##for some reason pytorch and windows causes an error with the sentiment dataset
if __name__ == '__main__':
    data = next(iter(train))
    print (data['input_ids'])

    

    while(1):
        input_text = Input()
        encoding = input_text.input_text()
    #print(encoding)