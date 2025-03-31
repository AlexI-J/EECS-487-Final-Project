import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string
from concurrent.futures import ProcessPoolExecutor
from model import run_model

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')

# Define these globally
stop_words = set(stopwords.words('english'))
punctuation_set = set(string.punctuation)
executor = ProcessPoolExecutor()

def clean_tokens(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return ' '.join([word for word in tokens if word not in stop_words])

def clean_column(texts):
    return list(executor.map(clean_tokens, texts))

def read_stocks():
    if os.path.exists("stocks.csv"):
        return pd.read_csv("stocks.csv")
    else:
        # Define category mappings
        category_map = {
            "Technology": ["AAPL", "NVDA", "MSFT", "FB", "GOOGL", "AMZN", "TSLA", "AMD", "ORCL", "ADBE"],
            "Healthcare": ["JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "UNH", "GILD", "ZBH", "CVS"],
            "Financial": ["JPM", "BAC", "GS", "MS", "WFC", "C", "AXP", "SCHW", "USB", "BLK"],
            "Energy": ["XOM", "CVX", "COP", "EOG", "ENPH", "FSLR", "NEE", "DUK", "SO", "VLO"],
            "Industrial": ["BA", "CAT", "GE", "DE", "UPS", "LMT", "HON", "RTN", "UNP", "MMM"]
        }

        # Invert the mapping to get stock -> category
        stock_to_category = {stock: category for category, stocks in category_map.items() for stock in stocks}

        folder = "stocks"

        df_list = []

        # Loop over all CSV files in the folder
        for filename in os.listdir(folder):
            if filename.endswith(".csv"):
                stock_name = filename[:-4]  # Remove .csv extension

                # Check if stock is in our list
                if stock_name in stock_to_category:
                    category = stock_to_category[stock_name]
                    file_path = os.path.join(folder, filename)
                    
                    df = pd.read_csv(file_path, parse_dates=['Date'])

                    # Filter by date range
                    mask = (df['Date'] >= '2016-01-01') & (df['Date'] <= '2020-04-01')
                    df = df.loc[mask].copy()

                    # Add stock name and category columns
                    df.insert(0, 'Stock', stock_name)
                    df.insert(1, 'Category', category)

                    df_list.append(df)

        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.columns = merged_df.columns.str.lower()
        merged_df.to_csv("stocks.csv", index=False)

        return merged_df

# Visualization function
def vis_stocks():
    df = pd.read_csv("stocks.csv", parse_dates=['date'])
    categories = df['category'].unique()

    for category in categories:
        plt.figure(figsize=(12, 6))
        category_df = df[df['category'] == category]

        for stock in category_df['stock'].unique():
            stock_df = category_df[category_df['stock'] == stock]
            plt.plot(stock_df['date'], stock_df['adj close'], label=stock)

        plt.title(f"Adjusted Close Prices - {category}")
        plt.xlabel("Date")
        plt.ylabel("Adjusted Close Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{category}_stocks.png")
        plt.close()

def read_articles():
    input_file = "all-the-news-2-1.csv"
    output_file = "all-the-news-2-1-processed.csv"
    chunksize = 10_000
    cols_to_remove = ['year', 'month', 'day', 'author', 'url']

    total_lines = 2_688_878
    total_chunks = total_lines // chunksize + 1

    with pd.read_csv(input_file, chunksize=chunksize) as reader:
        for i, chunk in enumerate(tqdm(reader, total=total_chunks)):
            chunk.drop(columns=cols_to_remove, errors='ignore', inplace=True)

            if 'title' in chunk:
                chunk['title'] = chunk['title'].fillna("").astype(str)
                chunk['title'] = clean_column(chunk['title'].values)

            if 'article' in chunk:
                chunk['article'] = chunk['article'].fillna("").astype(str)
                chunk['article'] = clean_column(chunk['article'].values)

            mode = 'w' if i == 0 else 'a'
            header = i == 0
            chunk.to_csv(output_file, mode=mode, header=header, index=False)

def main():
    # TODO: preprocess the dataset
    # Read it in line by line
    # Remove stopwords and punctuation
    # Write some helper functions for each step of training and testing process
    
    print("Reading stocks...")
    stocks = read_stocks()
    print(stocks.head())
    print("Visualizing stocks...")
    vis_stocks()
    if not os.path.exists("all-the-news-2-1-processed.csv"):
        print('Cleaning articles')
        read_articles()
    run_model("all-the-news-2-1-processed.csv", "stocks.csv")


if __name__=="__main__":
    main()
