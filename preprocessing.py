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
    return pd.read_csv("data/news.csv")

def load_word_vectors():
    glove_input_file = 'glove.6B.50d.txt'
    word2vec_output_file = 'glove.6B.50d.word2vec.txt'
    glove2word2vec(glove_input_file, word2vec_output_file)