from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification,AdamW,BertConfig


class DatasetTraining:
    print ("LOL")
    def __init__(self):
        print("Train with dataset")

    def train_model(self):
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_label=16,
            output_attentions=False,
            output_loading_info=False,
        )