from fastapi import FastAPI, Query
import feedparser
from pydantic import BaseModel
from typing import List
import pandas as pd
import os
from preprocessing import read_stocks, read_articles, get_news_windows
from model import load_data, StockGAP

app = FastAPI()
ffnn = None  # global model instance

@app.get("/")
def root():
    return {"message": "Welcome to StockGAP Dashboard API"}

@app.get("/train")
def train_model(window_size: int = Query(7, description="Number of days in each time window (e.g., 7 for a weekly window)"), 
                num_windows: int = Query(30, description="Number of non-overlapping windows to extract from the dataset")):
    global ffnn

    if not os.path.exists("data"):
        os.mkdir("data")

    read_stocks()
    read_articles()
    start_date = get_news_windows(window_size, num_windows)

    train, _ = load_data()

    ffnn = StockGAP('tfidf')
    trn_tfidf = ffnn.fit_tfidf(train)

    start_str = start_date.strftime("%Y%m%d")

    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    vectorizer_filename = os.path.join(model_dir, f"vectorizer_{window_size}_{num_windows}_{start_str}.pkl")
    model_filename = os.path.join(model_dir, f"model_{window_size}_{num_windows}_{start_str}.pkl")

    ffnn.save_vectorizer(vectorizer_filename)

    hyperparam = ffnn.cross_validation(trn_tfidf, train.delta)
    ffnn.fit(trn_tfidf, train.delta, hyperparam)

    ffnn.save_model(model_filename)

    with open(os.path.join("models", f"start_date_{window_size}_{num_windows}_{start_str}.txt"), "w") as f:
        f.write(str(start_date))

    return {
        "message": "Training complete",
        "model_file": model_filename,
        "vectorizer_file": vectorizer_filename,
        "start_date": str(start_date),
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

    model_dir = "models"
    model_filename = None
    vectorizer_filename = None

    # Search for the correct files including start_date
    for fname in os.listdir(model_dir):
        if fname.startswith(f"model_{window_size}_{num_windows}_") and fname.endswith(".pkl"):
            model_filename = os.path.join(model_dir, fname)
            vectorizer_filename = os.path.join(
                model_dir, fname.replace("model_", "vectorizer_")
            )
            break

    if model_filename is None or not os.path.exists(model_filename) or not os.path.exists(vectorizer_filename):
        return {"error": f"Model or vectorizer not found. Please train model first."}

    # Load test set
    _, test = load_data()
    if test.empty:
        return {"error": "Test set is empty or invalid."}

    # Load model and vectorizer
    ffnn = StockGAP("tfidf")
    ffnn.load_model(model_filename)
    ffnn.load_vectorizer(vectorizer_filename)

    # Run test
    accuracy, f1 = ffnn.test_performance(test)

    return {
        "message": f"Test results for {os.path.basename(model_filename)}",
        "test_set_size": len(test),
        "label_distribution": test["delta"].value_counts().to_dict(),
        "accuracy": accuracy,
        "macro_f1": f1
    }

class HeadlineInput(BaseModel):
    headlines: List[str]

@app.post("/predict")
def predict(input_data: HeadlineInput = None, use_scraped: bool = Query(False), model_name: str = Query(None)):
    global ffnn
    if model_name is None:
        return {"error": "Model filename must be provided."}

    model_path = os.path.join("models", model_name)
    vectorizer_path = model_path.replace("model_", "vectorizer_")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return {"error": f"Model or vectorizer file not found: {model_path}, {vectorizer_path}"}

    ffnn = StockGAP("tfidf")
    ffnn.load_model(model_path)
    ffnn.load_vectorizer(vectorizer_path)

    if use_scraped:
        feed = feedparser.parse("https://news.google.com/news/rss")
        headlines = [entry.title for entry in feed.entries[:10]]
    else:
        if input_data is None or not input_data.headlines:
            return {"error": "No headlines provided."}
        headlines = input_data.headlines

    processed = [word.lower() for headline in headlines for word in headline.split()]
    df = pd.DataFrame([["dummy", "FLAT"]], columns=["headlines", "delta"])
    df.at[0, "headlines"] = str(processed)

    X = ffnn.vectorizer.transform(df["headlines"])
    pred = ffnn.clf.predict(X)

    return {
        "prediction": pred.tolist()[0],
        "headlines_used": headlines,
        "model_used": model_name
    }

@app.get("/models")
def list_models():
    model_dir = "models"
    model_files = [f for f in os.listdir(model_dir) if f.startswith("model_") and f.endswith(".pkl")]

    def parse_model_name(filename):
        parts = filename.replace(".pkl", "").split("_")
        if len(parts) == 4:
            _, ws, nw, date = parts
            return {
                "filename": filename,
                "readable_name": f"Window size: {ws}, Number of windows: {nw}, Start date: {date[:4]}-{date[4:6]}-{date[6:]}"
            }
        return None

    models = [parse_model_name(f) for f in model_files]
    models = [m for m in models if m is not None]
    return models
