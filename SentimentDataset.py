import torch
import transformers

#define the tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')

class SentimentDataset():
    def __init__(self, df, max_len):
        self.tweet = df.Tweet.to_numpy()
        self.emotion = df.Emotion.to_numpy()
        self.intensity = df.Intensity.to_numpy()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        tweet_item = str(self.tweet[item])
        encoding = tokenizer.encode_plus(
            tweet_item,
            max_length=self.max_len,
            add_special_tokens=True,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'emotion': torch.tensor(self.string_to_num(self.emotion[item]), dtype=torch.long),
            'intensity': torch.tensor(self.intensity[item], dtype=torch.long)
        }

    def string_to_num(self, emotions):
        if emotions == 'anger':
            return 1
        elif emotions == 'sadness':
            return 2
        elif emotions == 'fear':
            return 3
        else:
            return 4
