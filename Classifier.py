from torch import nn
from transformers import BertForSequenceClassification, AdamW, BertConfig, data, BertModel


class Classifier(nn.Module):

    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.drop = nn.Dropout(0.03)
        self.out = nn.Linear(self.bert.config.hiddensize, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
           input_ids,
           attention_mask
        )
        output = self.drop(pooled_output)
        output = self.out(output)
        return self.softmax(output)
