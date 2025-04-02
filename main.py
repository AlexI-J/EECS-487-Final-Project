import pandas as pd
import os
from preprocessing import read_stocks, vis_stocks, analyze_stocks, read_articles, load_word_vectors
from dataset import NewsDataset, sen2vec
from model import run_model
from gensim.models import KeyedVectors

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
    vis_stocks()

    # Analysis of integrity of stock dataset
    analyze_stocks()

    print("Loading and tokenizing article text...")
    news_articles = read_articles()

    print("Loading word vectors...")
    load_word_vectors()
    word2vec_output_file = 'glove.6B.50d.word2vec.txt'
    glove = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    
    print("Loading dataset...")
    news_data = NewsDataset(stock_closes=stock_closes, news_articles=news_articles, date_range=1, glove=glove, sen2vec=sen2vec)

    print("Done!")

if __name__=="__main__":
    main()