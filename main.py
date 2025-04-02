import pandas as pd
import torch
import os
from preprocessing import read_stocks, vis_stocks, analyze_stocks, read_articles, load_word_vectors, get_unk_vec
from dataset import NewsDataset, sen2vec, collate_fn
from model import run_model

def main():
    # TODO: preprocess the dataset
    # Read it in line by line
    # Remove stopwords and punctuation
    # Write some helper functions for each step of training and testing process

    if not os.path.exists("data"):
        os.mkdir("data")

    print("Reading stocks...")
    stock_closes = read_stocks()
    print(stock_closes.head())

    print(next(stock_closes.iterrows())[0])

    print("Visualizing stocks...")
    #vis_stocks()

    # Analysis of integrity of stock dataset
    analyze_stocks()

    print("Loading and tokenizing article text...")
    news_articles = read_articles()

    print("Loading word vectors...")
    glove = load_word_vectors()
    unk = get_unk_vec(glove)
    
    print("Loading dataset...")
    news_data = NewsDataset(stock_closes=stock_closes, news_articles=news_articles, date_range=2, glove=glove, sen2vec=sen2vec)
    print(f"Len of news_data = {len(news_data)}")
    print(news_data[0])

    (train, val, test) = torch.utils.data.random_split(news_data, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, collate_fn=collate_fn, shuffle=True)


if __name__=="__main__":
    main()