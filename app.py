from fastapi import FastAPI
from inference import TSPredictor
app = FastAPI(title="MLOps Basics App-S2")
predictor = TSPredictor("./models/1/TSModel.hdf5","./models/1/vectorize_layer")
@app.get("/")
async def home():
    return "<h2>This is a sample NLP Project (sentiment analysis)</h2>"

@app.get("/predict")
async def get_prediction(text: str):
    processed_text = predictor.preprocess(raw_text = text)
    return predictor.infer(processed_text)