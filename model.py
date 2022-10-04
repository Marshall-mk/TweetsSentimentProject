import tensorflow as tf
from tensorflow.keras import Model


class TSModel(Model):
    def __init__(self, model_name="TweetsSentimentModel", max_tokens=2000):
        super(TSModel, self).__init__()
        self.model_name = model_name
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.embed_layer = tf.keras.layers.Embedding(input_dim=max_tokens, output_dim=64,mask_zero=True)
        self.lstm = tf.keras.layers.LSTM(32)
        self.bidirectional = tf.keras.layers.Bidirectional(self.lstm)
        
        self.dense = tf.keras.layers.Dense(128, activation="relu")
        self.out  = tf.keras.layers.Dense(1, activation="sigmoid")


    def call(self, input):
        """ Builds the Keras model based """
        x = input[:]
        x = self.embed_layer(x)
        x = self.bidirectional(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.dropout(x)
        return self.out(x)

if __name__ == '__main__':
    model = TSModel()