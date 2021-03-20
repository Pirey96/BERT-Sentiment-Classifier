import transformers
from Predict import Predict
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

class Input():
    def input_text (self):
        text = input("Input text: ")
        return self.tokenize_input(text)

    def tokenize_input(self, text):
            # special tokens [SEP] 102 [CLS] 101 [PAD] 0
            # encoding behaves like
            encoding = tokenizer.encode_plus(
                text,  # input text
                max_length=512,  # number of input words
                add_special_tokens=True,  # CES,CLS,SEP tokens
                padding='max_length',  # Padding to 512
                return_attention_mask=True,  # attention mask for weights
                truncation=True,  # truncate to 512
                return_token_type_ids=True,  # sequence identification
                return_tensors='pt'
            )
            self.classifier(encoding)
            return encoding



    def classifier(self, encoding):
        Predict(encoding, "fear", "joy", "sadness", "anger")
