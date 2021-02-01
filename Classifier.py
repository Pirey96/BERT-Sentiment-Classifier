from torch import nn
from transformers import BertModel

class Classifier (nn.Module):
    def __init__(self, emotion_intensity):
        super(Classifier, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-cased')  #import the bert model of cased datasets
        self.drop = nn.Dropout(p=0.6)    #dropout probabilties
        self.out = nn.Linear(self.bert_model.config.hidden_size, emotion_intensity)  #linear
        self.softmax = nn.Softmax(dim=1)    #normalize the tensors

    def forward(self, input_ids, attention_mask):
        _,pooled_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.softmax(self.out(output))
