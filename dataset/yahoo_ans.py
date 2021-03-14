import torch
import os
import pandas as pd


class YahooAnsDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, tokenizer, mode='train'):
        assert mode in ('train', 'test')

        self.data = pd.read_csv(os.path.join(data_dir, mode+'.csv'), header=None)
        self.data = self.data[~self.data.iloc[:, 3].isnull()]
        self.label_desc = [l for l in open(os.path.join(data_dir, 'classes.txt')).readlines()]
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        question = self.data.iloc[idx][1]
        answer = self.data.iloc[idx][3]
        label = self.data.iloc[idx][0] - 1
        label_desc = self.label_desc[label]

        question = self.tokenizer.encode(question)
        answer = self.tokenizer.encode(answer)
        label_desc = self.tokenizer.encode(label_desc)

        return {'question': question, 'answer': answer, 'label desc': label_desc, 'label': label}

    def __len__(self):
        return len(self.data)
