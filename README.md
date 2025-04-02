# StockGAP: Generalized Article Predictor for Stock Prices from Category-Based News
## Usage
1. Clone this repository to your system for local access.
2. Download the [All the News 2.0](https://components.one/datasets/all-the-news-2-news-articles-dataset) dataset and move `all-the-news-2-1.csv` to the same directory as the repository.
3. Download the [Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset) by Oleh Onyshchak and move the `stocks` folder to the same directory as the repository.
4. Download ``glove.6B.zip`` from the [GloVe official website](https://nlp.stanford.edu/projects/glove/). Extract the file ``glove.6B.50d.txt`` and move it to the same directory as the repository.
5. In your Python virtual environment, ensure the following libraries are installed:
  - `pandas`
  - `tqdm`
  - `matplotlib`
  - `nltk`
  - `gensim`
6. Run the following: ``python main.py``
