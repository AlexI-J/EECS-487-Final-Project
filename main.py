import pandas as pd
import torch
import os
from preprocessing import read_stocks, vis_stocks, analyze_stocks, read_articles, get_news_windows
from model import load_data, get_stats, StockGAP

def main():
    if not os.path.exists("data"):
        os.mkdir("data")

    print("Reading stocks...")
    stock_closes = read_stocks()
    print(stock_closes.head())

    print(next(stock_closes.iterrows())[0])

    # Analysis of integrity of stock dataset
    #analyze_stocks()

    print("Loading and tokenizing article text...")
    news_articles = read_articles()
    
    print("Loading windows...")
    get_news_windows(7, 30)

    print("Loading and splitting data...")
    train, test = load_data()
    get_stats(train)
    
    print("Creating model...")
    ffnn = StockGAP('tfidf')
    trn_tfidf = ffnn.fit_tfidf(train)
    print(trn_tfidf.shape)
    trn_tfidf.toarray()
    
    print("Finding hyperparameters...")
    hyperparam = ffnn.cross_validation(trn_tfidf, train.delta)
    print(hyperparam)
    ffnn.fit(trn_tfidf, train.delta, hyperparam)
    ffnn.clf
    print("Testing model...")
    print(f"(accuracy, macro F1 score) on the test set is {ffnn.test_performance(test)}")

    print("Finished tests.")


if __name__=="__main__":
    main()