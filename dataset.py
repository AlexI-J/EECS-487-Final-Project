from typing import List
import torch
import pandas as pd
import numpy as np
import datetime
import pickle
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from collections import defaultdict

def sen2vec(sentence, embeddings, dim=50):
    words = sentence.lower().split()
    vectors = [embeddings[word] for word in words if word in embeddings]
    if not vectors:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)

class NewsDataset(Dataset):
    """Dataset for StockGAP"""

    # stock_closes: pandas dataframe with stock,category,date,open,close columns
    # news_articles: pandas dataframe with article date and title columns
    # date_range: number of days of data leading up to the prediction day to include
    def __init__(self, stock_closes: pd.DataFrame, news_articles: pd.DataFrame, date_range: int, glove: dict, sen2vec):
        super().__init__()

        self.data = []
        self.date_range = date_range
        self.glove = glove
        self.sen2vec = sen2vec

        # Use date as key; get list of article titles for given day
        self.articles = defaultdict(list)

        # Use date as key; get dicts with category as key to get list of stock closing prices
        self.stocks = defaultdict(lambda: defaultdict(list))

        for row in news_articles.itertuples(index=False):
            self.articles[row.date].append(row.title)

        for row in stock_closes.itertuples(index=False):
            self.stocks[row.date][row.category].append(row.close - row.open)

        target = datetime.datetime.strptime('2016-01-01', '%Y-%m-%d') + datetime.timedelta(days=date_range)
        final_date = datetime.datetime.strptime('2020-04-01', '%Y-%m-%d')
        categories = ['industry', 'tech', 'health', 'finance', 'energy']
        while target <= final_date:
            # Skip date if we don't have data on it
            if target not in self.stocks.keys():
                target = target + datetime.timedelta(days=1)
                continue

            context = [torch.tensor(self.glove[category])]

            # Get news articles for each date
            for i in range(1, date_range):
                day = target - datetime.timedelta(days=i)
                for title in self.articles[day]:
                    context.append(sen2vec(title))

            for category in categories:
                # We should have all categories for dates that do exist, but check to be sure
                if category not in self.stocks[target].keys():
                    continue
                avg_change = sum(self.stocks[day][category]) / len(self.stocks[day][category])
                self.data.append((context, avg_change))
            
            target = target + datetime.timedelta(days=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        # Model input:
        # 1) Word vector representation of stock category
        # 2) List of vector representations of news article titles


        # Output is a tuple:
        # (news_text, stock_price_change)
        # news_text: List of "text" for each article from a certain time range (1 week)
        # Number of weeks in time range 1-1-2016 to 4-2-2020: Roughly 222 weeks
        # (dataset is 222 items long)
        # stock_price_change: A dictionary:
        #   Keys: Stock ticker names, e.g. AAPL.
        #   Values: Relative change in adj close for that stock.