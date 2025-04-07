from typing import List
import torch
import pandas as pd
import numpy as np
import datetime
import pickle
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from collections import defaultdict

def sen2vec(sentence, embeddings, dim=50):
    #words = sentence.lower().split()
    vectors = [embeddings[word] for word in sentence if word in embeddings]
    if not vectors:
        return torch.zeros(dim)
    return torch.from_numpy(np.mean(vectors, axis=0))

def lstm_collate_fn(batch):
    """
    Collate function for dataloader
    """
    inputs = {'contexts': [], 'seq_lens': []}
    outputs = []

    inputs['seq_lens'] = [len(entry[0]) for entry in batch]
    max_len = max(inputs['seq_lens'])

    for entry in batch:
      while len(entry[0]) < max_len:
          entry[0].append(torch.zeros(50))
      inputs['contexts'].append(torch.stack(entry[0]))
      outputs.append(torch.tensor(torch.tensor(entry[1])))

    inputs['contexts'] = torch.stack(inputs['contexts'])
    outputs = torch.tensor(outputs)

    return inputs, outputs

class NewsDataset(Dataset):
    """Dataset for StockGAP"""

    # stock_closes: pandas dataframe with stock,category,date,open,close columns
    # news_articles: pandas dataframe with article date and title columns
    # date_range: number of days of data leading up to the prediction day to include
    def __init__(self, stock_closes: pd.DataFrame, news_articles: pd.DataFrame, date_range: int, glove: dict, sen2vec):
        super().__init__()

        # Convert DataFrame dates to datetime.date
        stock_closes['date'] = pd.to_datetime(stock_closes['date']).dt.date
        news_articles['date'] = pd.to_datetime(news_articles['date']).dt.date

        self.data = []
        self.date_range = date_range
        self.glove = glove
        self.sen2vec = sen2vec

        # Use date as key; get list of article titles for given day
        self.articles = defaultdict(list)

        # Use date as key; get dicts with category as key to get list of stock closing prices
        self.stocks = defaultdict(lambda: defaultdict(list))

        for row in news_articles.itertuples(index=False):
            self.articles[row.date].append(json.loads(row.title.replace('\'', '\"')))

        for row in stock_closes.itertuples(index=False):
            self.stocks[row.date][row.category].append(row.close - row.open)

        target = datetime.date(2020, 1, 1) + datetime.timedelta(days=date_range)
        final_date = datetime.date(2020, 4, 1)
        categories = ['industry', 'tech', 'health', 'finance', 'energy']
        while target <= final_date:
            # Skip date if we don't have data on it
            if target not in self.stocks.keys():
                target = target + datetime.timedelta(days=1)
                continue

            context = []

            # Get news articles for each date
            for i in range(1, date_range):
                day = target - datetime.timedelta(days=i)
                for title in self.articles[day]:
                    context.append(sen2vec(title, glove))
            
            for category in categories:
                # We should have all categories for dates that do exist, but check to be sure
                if category not in self.stocks[target].keys():
                    continue
                avg_change = sum(self.stocks[target][category]) / len(self.stocks[target][category])
                category_context = [torch.tensor(self.glove[category])] + context
                self.data.append((category_context, avg_change))
            
            target = target + datetime.timedelta(days=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]