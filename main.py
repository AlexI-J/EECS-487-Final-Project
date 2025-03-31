import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
import string
from concurrent.futures import ProcessPoolExecutor

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
    else:
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

def read_articles():
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

def main():
    # TODO: preprocess the dataset
    # Read it in line by line
    # Remove stopwords and punctuation
    # Write some helper functions for each step of training and testing process

    if not os.path.exists("data"):
        os.mkdir("data")

    print("Reading stocks...")
    stocks = read_stocks()
    print(stocks.head())
    print("Visualizing stocks...")
    vis_stocks()

    # Analysis of integrity of stock dataset
    dates_without_stocks()
    dates_with_stocks()

    print("Reading and tokenizing article titles...")
    read_articles()


if __name__=="__main__":
    main()