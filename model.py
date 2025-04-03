import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from gensim.models import KeyedVectors
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from datetime import timedelta
import os

# Loads embeddings. Same as the function in dataset.py
def load_word_vectors():
    word2vec_output_file = "glove.6B.50d.word2vec.txt"
    if not os.path.exists(word2vec_output_file):
        glove_input_file = "glove.6B.50d.txt"
        glove2word2vec(glove_input_file, word2vec_output_file)
    return KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# Prepares and returns stock and news data for training and testing
class StockNewsDataset(Dataset):
    def __init__(self, news_data, stock_data, word_vectors, start_date, date_range, embedding_dim):
        self.news_data = news_data
        self.stock_data = stock_data
        self.word_vectors = word_vectors
        self.start_date = pd.to_datetime(start_date)
        self.date_range = date_range
        self.embedding_dim = embedding_dim
        
        print("Preparing data...")
        self.data = self.prepare_data()
    
    def __len__(self):
        return len(self.data)
    
    # Extracts stock data, news articles, and net change for a given index
    def __getitem__(self, idx):
        stock_context, next_day_news, stock_change = self.data[idx]
        stock_context = torch.tensor(stock_context, dtype=torch.float32)
        next_day_news = torch.tensor(next_day_news, dtype=torch.float32)
        stock_change = torch.tensor(stock_change, dtype=torch.float32)
        
        return stock_context, next_day_news, stock_change
    
    # Combines stock data and news articles
    def prepare_data(self):
        data = []
        for i in range(self.date_range, len(self.stock_data) - 1):
            stock_context = self.stock_data.iloc[(i - self.date_range):i][['open', 'close']].values
            
            next_day_news = self.get_news_data(i + 1)
            next_day_stock_data = self.stock_data.iloc[i + 1]
            stock_change = 1 if next_day_stock_data['close'] > next_day_stock_data['open'] else 0
            
            data.append((stock_context, next_day_news, stock_change))
        
        return data
    
    # Retrieves and processes article data for a given index
    def get_news_data(self, idx):
        news_article = self.news_data.iloc[idx]['article']
        news_title = self.news_data.iloc[idx]['title']
        title_vec = self.get_sentence_embedding(news_title)
        article_vec = self.get_sentence_embedding(news_article)
        news_vec = np.concatenate([title_vec, article_vec], axis=0)
        
        return news_vec

    # Get the average word vector embedding for a sentence    
    def get_sentence_embedding(self, sentence):
        words = sentence.split()
        word_embeddings = []
        for word in words:
            if word in self.word_vectors:
                word_embeddings.append(self.word_vectors[word])
            else:
                word_embeddings.append(np.zeros(self.embedding_dim))
        return np.mean(word_embeddings, axis=0)

# LSTM model for stock prediction
class StockLSTM(nn.Module):
    def __init__(self, historical_input_dim, news_input_dim, embedding_dim, hidden_dim=128, num_layers=2, output_dim=1):
        super(StockLSTM, self).__init__()
        self.embed_dim = embedding_dim
        
        self.stock_lstm = nn.LSTM(input_size=historical_input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_stock = nn.Linear(hidden_dim, output_dim)
        self.fc_news = nn.Linear(news_input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    # Forward pass
    def forward(self, historical_data, news_data):
        lstm_out, (hn, cn) = self.stock_lstm(historical_data)
        last_hidden_state = lstm_out[:, -1, :]
        news_pred = self.fc_news(news_data)
        combined = last_hidden_state + news_pred
        output = self.fc_stock(combined)
        output = self.sigmoid(output)
        
        return output

    # Trains model using CE loss
    def train_model(self, train_loader, epochs):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            correct_preds = 0
            total_preds = 0

            for historical_data, news_data, stock_change in train_loader:
                optimizer.zero_grad()
                outputs = self(historical_data, news_data)
                loss = criterion(outputs, stock_change.unsqueeze(1))

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct_preds += (predicted == stock_change.unsqueeze(1)).sum().item()
                total_preds += stock_change.size(0)

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct_preds / total_preds
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    # Tests model by predicting stocks the next day after the context data
    def test_model(self, word_vectors, news_df, stocks_df, start_date, test_range, category):
        test_date = pd.to_datetime(start_date) + timedelta(days=test_range)
    
        test_news = news_df[news_df['date'] == test_date]
        if len(test_news) == 0:
            print("No news available for the test date.")
            return
        
        filtered_stocks = stocks_df[(stocks_df['category'] == category) & (stocks_df['date'] == test_date)]
        if len(filtered_stocks) == 0:
            print(f"No stock data available for the category '{category}' on {test_date}.")
            return
    
        test_dataset = StockNewsDataset(test_news, filtered_stocks, word_vectors, start_date, test_range, self.embed_dim)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        if len(test_loader.dataset) == 0:
            print("No data available in test loader.")
            return

        self.eval()
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for historical_data, news_data, stock_change in test_loader:
                outputs = self(historical_data, news_data)
                predicted = (outputs > 0.5).float()
                
                # Count correct predictions
                print(f"Predicted: {predicted} Correct: {stock_change.unsqueeze(1)}")
                correct_preds += (predicted == stock_change.unsqueeze(1)).sum().item()
                total_preds += stock_change.size(0)

        test_accuracy = (correct_preds / total_preds) * 100
        print(f"Test Accuracy: {round(test_accuracy)}%")

# Creates, trains, and tests the model
def run_model(category, news_df, stocks_df, embed_dims, start_date, date_range, test_range=1):
    print("Loading embeddings...")
    word_vectors = load_word_vectors()

    print("Filtering stocks...")
    filtered_stocks = filter_stocks_by_category_and_date(stocks_df, category, start_date, date_range)
    
    print("Merging data...")
    merged_data = pd.merge(news_df, filtered_stocks, on='date', how='inner')

    print("Creating dataset...")
    train_dataset = StockNewsDataset(merged_data, filtered_stocks, word_vectors, start_date, date_range, embed_dims)
    
    print("Loading dataset...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    stock_history_dim = 2
    news_input_dim = 100
    print("Creating model...")
    model = StockLSTM(historical_input_dim, news_input_dim, embed_dims, hidden_dim=128, num_layers=2)
    
    print("Training model...")
    model.train_model(train_loader, epochs=60)

    print("Testing model...")
    model.test_model(word_vectors, news_df, stocks_df, start_date, date_range + test_range, category)

# Filter stocks by category and date
def filter_stocks_by_category_and_date(stocks_df, category, start_date, date_range):
    end_date = pd.to_datetime(start_date) + timedelta(days=date_range)
    filtered_stocks = stocks_df[(stocks_df['category'] == category) &
                                 (stocks_df['date'] >= start_date) & 
                                 (stocks_df['date'] <= end_date)]
    return filtered_stocks