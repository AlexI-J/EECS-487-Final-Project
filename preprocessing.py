import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
from concurrent.futures import ProcessPoolExecutor
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import torch
import csv
from datetime import timedelta
import ast
import random

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

executor = ProcessPoolExecutor()

def clean_tokens(text):
    tokens = word_tokenize(text.lower())
    tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens if re.sub(r'[^\w\s]', '', token)]
    return tokens

def clean_article_tokens(article):
    # Limit article text to its first 4 sentences
    # (typically includes headline and opening paragraph)
    return [clean_tokens(sentence) for sentence in sent_tokenize(article)[0:4]]

def clean_titles(titles):
    return list(executor.map(clean_tokens, titles))

def clean_articles(articles):
    return list(executor.map(clean_article_tokens, articles))

def read_stocks():
    if os.path.exists("data/stocks.csv"):
        return pd.read_csv("data/stocks.csv")
    
    # Define category mappings
    category_map = {
        "tech": ["AAPL", "NVDA", "MSFT", "FB", "GOOGL", "AMZN", "TSLA", "AMD", "ORCL", "ADBE"],
        "health": ["JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "UNH", "GILD", "ZBH", "CVS"],
        "finance": ["JPM", "BAC", "GS", "MS", "WFC", "C", "AXP", "SCHW", "USB", "BLK"],
        "energy": ["XOM", "CVX", "COP", "EOG", "ENPH", "FSLR", "NEE", "DUK", "SO", "VLO"],
        "industry": ["BA", "CAT", "GE", "DE", "UPS", "LMT", "HON", "RTN", "UNP", "MMM"]
    }

    # Invert the mapping to get stock -> category
    stock_to_category = {stock: category for category, stocks in category_map.items() for stock in stocks}

    folder = "stocks"
    df_list = []

    # Loop over all CSV files in the folder
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            # Remove .csv extension
            stock_name = filename[:-4]  

            if stock_name in stock_to_category:
                category = stock_to_category[stock_name]
                file_path = os.path.join(folder, filename)
                
                df = pd.read_csv(file_path, parse_dates=['Date'])

                df = df[['Date', 'Open', 'Close']].copy()
                df.rename(columns=str.lower, inplace=True)

                mask = (df['date'] >= '2016-01-01') & (df['date'] <= '2020-04-01')
                df = df.loc[mask]

                df.insert(0, 'stock', stock_name)
                df.insert(1, 'category', category)

                df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    # Drop entry from stock CAT, date 2016-01-18
    # (empty row from original dataset)
    merged_df = merged_df[merged_df['date'] != "2016-01-18"]
    merged_df.to_csv("data/stocks.csv", index=False)

    print("Stock data saved to data/stocks.csv.")
    return merged_df

# Visualization function
def vis_stocks():
    if not os.path.exists("images"):
        os.mkdir("images")

    df = pd.read_csv("data/stocks.csv", parse_dates=['date'])
    categories = df['category'].unique()

    for category in categories:
        plt.figure(figsize=(12, 6))
        category_df = df[df['category'] == category]

        for stock in category_df['stock'].unique():
            stock_df = category_df[category_df['stock'] == stock]
            plt.plot(stock_df['date'], stock_df['close'], label=stock)

        plt.title(f"Adjusted Close Prices - {category}")
        plt.xlabel("Date")
        plt.ylabel("Adjusted Close Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"images/{category}_stocks.png")
        plt.close()

def dates_without_stocks():
    df = pd.read_csv("data/stocks.csv")
    # Generate all calendar dates in the stocks dataset range
    all_dates = pd.date_range(start="2016-01-01", end="2020-04-01", freq='D')

    dates_in_data = pd.to_datetime(df['date'].unique())
    missing_dates = all_dates.difference(dates_in_data)

    print(f"From a total of {len(all_dates)} dates in the date range of the stocks dataset, {len(missing_dates.tolist())} have no data.")

def dates_with_stocks():
    df = pd.read_csv("data/stocks.csv")
    df.drop(['close'], axis=1, inplace=True)

    result = df.groupby(['date', 'category']).count().unstack(level=-1)

    # Find all rows where any value is less than 10
    filtered = result[(result < 10).any(axis=1)]
    if filtered.empty:
        print("Of the dates in the stocks dataset, all of them have values for all stocks.")
    else:
        print("WARNING: Of the dates in the stocks dataset, some of them have are missing stock values!")

def analyze_stocks():
    dates_without_stocks()
    dates_with_stocks()

def read_articles():
    if not os.path.exists("data/news.csv"):
        input_file = "all-the-news-2-1.csv"
        output_file = "data/news.csv"
        chunksize = 10_000
        cols_to_remove = ['year', 'month', 'day', 'author', 'url', 'section', 'publication']
        
        total_lines = 2_688_878
        total_chunks = total_lines // chunksize + 1

        with pd.read_csv(input_file, chunksize=chunksize) as reader:
            for i, chunk in enumerate(tqdm(reader, total=total_chunks)):
                chunk.drop(columns=cols_to_remove, errors='ignore', inplace=True)

                if 'title' in chunk:
                    chunk['title'] = chunk['title'].fillna("").astype(str)
                    chunk['title'] = clean_titles(chunk['title'].values)

                if 'article' in chunk:
                    chunk['article'] = chunk['article'].fillna("").astype(str)
                    chunk['article'] = clean_articles(chunk['article'].values)

                if 'date' in chunk:
                    chunk['date'] = [pd.to_datetime(datetime_str).date() for datetime_str in chunk['date']]

                mode = 'w' if i == 0 else 'a'
                header = i == 0
                
                chunk.to_csv(output_file, mode=mode, header=header, index=False)
    print("News data saved to data/news.csv. Reading data into Python...")
    return pd.read_csv("data/news.csv", encoding="utf-8")

def load_embeddings():
    embeddings = {}
    
    with open('glove.6B.100d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(v) for v in values[1:]], dtype=torch.float)
            embeddings[word] = vector
    
    return embeddings

def get_unk_vec(glove):
    print(glove)

def get_news_windows(window_size, num_windows):
    data = [["headlines", "delta"]]
    
    # Load the data
    news_df = pd.read_csv("data/news.csv", parse_dates=["date"])
    stocks_df = pd.read_csv("data/stocks.csv", parse_dates=["date"])

    # Sort for safety
    news_df = news_df.sort_values("date")
    stocks_df = stocks_df.sort_values("date")

    # Set window size
    window_days = window_size

    # Date range
    start_date = news_df["date"].min()
    end_date = news_df["date"].max()
    count = 0
    
    latest_start = end_date - timedelta(days=num_windows * window_days)
    start_date = start_date + timedelta(
        days=random.randint(0, max(0, (latest_start - start_date).days))
    )

    current_date = start_date
    while current_date + timedelta(days=window_days) <= end_date and count != num_windows:
        window_start = current_date
        window_end = current_date + timedelta(days=window_days)

        headlines_df = news_df[(news_df["date"] >= window_start) & (news_df["date"] < window_end)]
        words = headlines_df["title"].dropna().tolist()
        flattened = []
        for title in words:
            flattened = flattened + ast.literal_eval(title)

        window_stocks = stocks_df[(stocks_df["date"] >= window_start) & (stocks_df["date"] <= window_end)]
        changes = []
        for ticker in window_stocks["stock"].unique():
            ticker_data = window_stocks[window_stocks["stock"] == ticker].sort_values("date")
            if not ticker_data.empty:
                start_open = ticker_data.iloc[0]["open"]
                end_close = ticker_data.iloc[-1]["close"]
                changes.append(end_close - start_open)

        avg_change = sum(changes) / len(changes) if changes else 0
        delta = "FLAT"

        if avg_change < 0:
            delta = "LOSS"
        if avg_change > 0:
            delta = "GAIN"

        data.append([flattened, delta])

        current_date += timedelta(days=1)
        count += 1
        print(f"Windows created: {count}")

    
    with open("data/final.csv", mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    return start_date