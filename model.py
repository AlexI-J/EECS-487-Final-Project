import pandas as pd
import numpy as np
import torch
import torch.nn as nn
# import other libraries as needed

class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=1):
       super().__init__()
       self.lstm = nn.LSTM(50, hidden_dim, num_layers, batch_first=True)
        

    def forward(self, input_ids, seq_lens):
        output = []
        
        return output

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

def visualize_results():
    # plot model accuracy and loss
    # may end up being called in one of the other functions
    pass

def run_model(news_data, stocks_data):
    create_model()
    test_model()
    evaluate_model()
    # visualize results maybe?
    pass