from fastapi import FastAPI
from inference import TSPredictor
app = FastAPI(title="MLOps Basics App")
predictor = TSPredictor("C:/Users/HI/Desktop/.dev/python/Deep learning/Projects/TweetsSentimentAnalysis/models/1/TSModel.hdf5","C:/Users/HI/Desktop/.dev/python/Deep learning/Projects/TweetsSentimentAnalysis/models/1/vectorize_layer")
@app.get("/")
async def home():
    return "<h2>This is a sample NLP Project</h2>"
@app.get("/predict")
async def get_prediction(text: str):
    predictor.preprocess(text)
    return predictor.infer(text)