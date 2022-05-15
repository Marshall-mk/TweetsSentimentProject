import tensorflow as tf
import numpy as np


class TSPredictor:
    def __init__(self, model_path,vectorize_layer_path):
        self.model_path = model_path
        self.model = tf.keras.models.load_model(self.model_path)
        self.vectorize_layer_path = vectorize_layer_path
        self.loaded_vectorize_layer_model = tf.keras.models.load_model(self.vectorize_layer_path)
        
    def preprocess(self, raw_text):
        # Uses the trained vectorization layer to preprocess the text
        loaded_vectorize_layer = self.loaded_vectorize_layer_model.layers[-1]
        return loaded_vectorize_layer(raw_text)[np.newaxis, :] # Creates a new axis for batch size

    def infer(self, text):
        pred = self.model.predict(text)
        return {'output':pred.tolist()}


if __name__ == "__main__":
    text = "text"
    file_path = "./models/1/vectorize_layer"
    model_path = "./models/1/TSModel.hdf5"
    predictor = TSPredictor(model_path,file_path)
    print(predictor.infer(predictor.preprocess(text)))
