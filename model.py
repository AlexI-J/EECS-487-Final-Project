import pandas as pd
import numpy as np
import tensorflow as tf

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
def preprocess_text_data(text_data, tokenizer=None, max_len=100):
    tokenized_articles = [article.split() for article in text_data]

    if tokenizer is None:
        tokenizer = Tokenizer(oov_token='<OOV>')
        tokenizer.fit_on_texts(tokenized_articles)  # Fit tokenizer to the tokenized text data

    sequences = tokenizer.texts_to_sequences(tokenized_articles)

    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    return padded_sequences, tokenizer

def prepare_embedding_matrix(vocabulary, glove_embeddings, embedding_dim=100):
    embedding_matrix = np.zeros((len(vocabulary) + 1, embedding_dim))
    for word, i in vocabulary.items():
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def calculate_stock_change(ticker, start, end):
    # calculates the change of a stock from start to end date
    pass

def create_model(training_news, training_stocks):
    # create and train model on news and stocks using LSTM and word embeddings
    # first focus on stock going up or down, then get more precise
    pass

def test_model(test_data):
    # run model on a set of test articles to generate predictions
    pass

def evaluate_model(predicted, correct):
    # compare predicted change to correct change and adjust hyperparameters
    # use logistic regression to measure loss
    pass

def run_model(news_data, stocks_data):
    build_model()
    train_model()
    test_model()
    evaluate_model()
    # visualize results
    pass