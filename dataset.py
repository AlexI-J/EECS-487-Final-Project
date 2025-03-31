from typing import List
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict

# load GloVe embeddings
def load_glove_embeddings(glove_path, embedding_dim=100):
    print(f"Loading GloVe embeddings")
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f"Found {len(embeddings_index)} word vectors.")
    return embeddings_index

# get padded sequences of tokens for each article
def preprocess_text_data(articles, max_len=100):
    tokenized_articles = [article.split() for article in articles]

    # crib from homework 3 for this part
    tokenizer = Tokenizer(oov_token='<UNK>')
    
    # process the rest of the articles
    sequences = tokenizer.texts_to_sequences(tokenized_articles)

    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    return padded_sequences, tokenizer

# creates an embedding matrix. also crib from homework 3 for this part
def create_embedding_matrix(vocabulary, glove_embeddings, embedding_dim=100):
    embedding_matrix = np.zeros((len(vocabulary) + 1, embedding_dim))
    for word, i in vocabulary.items():
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


class NewsDataset(Dataset):
    """Dataset for """

    # stock_closes: pandas dataframe with stock,category,date,open,high,low,close,adj close,volume columns
    # news_articles: pandas dataframe with article date and title columns
    # date_range: number of days of data leading up to the prediction day to include
    def __init__(self, stock_closes: pd.DataFrame, news_articles: pd.DataFrame, date_range: int):
        super().__init__()

        self.date_range = date_range

        # use date as key get list of article titles for given day
        self.articles = defaultdict(list)

        # use date as key to get dicts with category as key to get list of stock closing prices
        self.stocks = defaultdict(defaultdict(list))

        for date, title in news_articles.iterrows():
            self.articles[date].append(title)

        for stock,category,date,open,high,low,close,adj_close,volume in stock_closes.iterrows():
            self.stocks[date][category].append(adj_close)

    
    def __len__(self):
        pass    


    def __getitem__(self, idx):
        # Model input:
        # 1) Word vector representation of stock category
        # 2) Recent [time frame] of stock prices
        # 3) List of vector representations of news article titles


        # Output is a tuple:
        # (news_text, stock_price_change)
        # news_text: List of "text" for each article from a certain time range (1 week)
        # Number of weeks in time range 1-1-2016 to 4-2-2020: Roughly 222 weeks
        # (dataset is 222 items long)
        # stock_price_change: A dictionary:
        #   Keys: Stock ticker names, e.g. AAPL.
        #   Values: Relative change in adj close for that stock.

        # 1 2 3 4 5 6 7 8 9 10
        # [           ]
        #   [           ]
        #     [           ]
        pass
