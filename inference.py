import tensorflow as tf
import numpy as np


class TSPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = tf.keras.models.load_model(self.saved_path)

    def preprocess(self, raw_text):
        text_vectorization = tf.keras.layers.TextVectorization(max_tokens=2000, output_mode="int", output_sequence_length=100,)
        
        return text_vectorization(raw_text)[np.newaxis, :]

    def infer(self, text=None):
        pred = self.model.predict(text)
        return {'output':pred}


if __name__ == "__main__":
    text = "image path"
    predictor = TSPredictor("models/1/emotionModel.hdf5")
    predictor.preprocess(text)
    print(predictor.infer(text))
