from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import pandas as pd
import os
from preprocessing import read_stocks, read_articles, get_news_windows
from model import load_data, get_stats, StockGAP

app = FastAPI()
ffnn = None  # global model instance

@app.get("/")
def root():
    return {"message": "Welcome to StockGAP Dashboard API"}

@app.get("/train")
def train_model(window_size: int = Query(7), num_windows: int = Query(30)):
    global ffnn

    if not os.path.exists("data"):
        os.mkdir("data")

    read_stocks()
    read_articles()
    get_news_windows(window_size, num_windows)

    train, _ = load_data()
    get_stats(train)

    ffnn = StockGAP('tfidf')
    trn_tfidf = ffnn.fit_tfidf(train)
    ffnn.save_vectorizer(f"vectorizer_{window_size}_{num_windows}.pkl")
    hyperparam = ffnn.cross_validation(trn_tfidf, train.delta)
    ffnn.fit(trn_tfidf, train.delta, hyperparam)

    model_filename = f"model_{window_size}_{num_windows}.pkl"
    ffnn.save_model(model_filename)

    return {
        "message": "Training complete",
        "model_file": model_filename,
        "hyperparameters": {
            "hidden_layer": hyperparam["hidden_layer"],
            "learning_rate": hyperparam["learning_rate"],
            "alpha": hyperparam["alpha"],
            "mean_f1_macro": hyperparam["mean_f1_macro"],
            "f1_macro": hyperparam["f1_macro"].tolist()
        }
    }

@app.get("/test")
def test_model(window_size: int = Query(7), num_windows: int = Query(30)):
    global ffnn
    model_filename = f"model_{window_size}_{num_windows}.pkl"
    vectorizer_filename = f"vectorizer_{window_size}_{num_windows}.pkl"

    # Check files
    if not os.path.exists(model_filename) or not os.path.exists(vectorizer_filename):
        return {"error": f"Model or vectorizer not found. Please train model first."}

    # Load test set
    _, test = load_data()  # You can change this to load from a saved file

    if test.empty:
        return {"error": "Test set is empty or invalid."}

    # Load model and vectorizer (no refitting)
    ffnn = StockGAP("tfidf")
    ffnn.load_model(model_filename)
    ffnn.load_vectorizer(vectorizer_filename)

    # Run test
    accuracy, f1 = ffnn.test_performance(test)

    return {
        "message": f"Test results for {model_filename}",
        "test_set_size": len(test),
        "label_distribution": test["delta"].value_counts().to_dict(),
        "accuracy": accuracy,
        "macro_f1": f1
    }

class HeadlineInput(BaseModel):
    headlines: List[str]

@app.post("/predict")
def predict(input_data: HeadlineInput):
    global ffnn
    if ffnn is None or not hasattr(ffnn, "clf"):
        return {"error": "Model not trained or loaded. Use /train or /test first."}

    processed = [word.lower() for headline in input_data.headlines for word in headline.split()]
    df = pd.DataFrame([["dummy", "FLAT"]], columns=["headlines", "delta"])
    df.at[0, "headlines"] = str(processed)

    X = ffnn.vectorizer.transform(df["headlines"])
    pred = ffnn.clf.predict(X)

    return {"prediction": pred.tolist()[0]}