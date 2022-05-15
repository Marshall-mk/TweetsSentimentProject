import numpy as np
import onnxruntime as rt
import tensorflow as tf



class TSONNXPredictor:
    def __init__(self, model_path, vectorize_layer_path):
        self.ort_session = rt.InferenceSession(model_path)
        self.vectorize_layer_path = vectorize_layer_path
        self.loaded_vectorize_layer_model = tf.keras.models.load_model(self.vectorize_layer_path)
    
    def preprocess(self, raw_text):
        # Uses the trained vectorization layer to preprocess the text
        loaded_vectorize_layer = self.loaded_vectorize_layer_model.layers[-1]
        return loaded_vectorize_layer(raw_text)

    def predict(self, processed_text):
        input_name = self.ort_session.get_inputs()[0].name
        # label_name = self.ort_session.get_outputs()[0].name
        return self.ort_session.run(None, {input_name: [np.array(processed_text, dtype=np.float32)]})[0]
        


if __name__ == "__main__":
    sentence = "The boy hates sitting on a bench"
    file_path = "./models/1/vectorize_layer"
    model_path = "./models/1/TSModel.onnx"
    predictor = TSONNXPredictor(model_path, file_path)
    processed_sent = predictor.preprocess(sentence)
    print(predictor.predict(processed_sent))
