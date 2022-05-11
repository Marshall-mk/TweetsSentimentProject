import tensorflow as tf
from tensorflow.keras import layers, Model


class TSModel:
    def __init__(self, model_name="TweetsSentimentModel"):
        super(TSModel, self).__init__()
        self.model_name = model_name
        self.build()

    def build(self):
        """ Builds the Keras model based """
        max_tokens = 2000
        inputs = tf.keras.Input(shape=(None,), dtype="int64")
        # Next, we add a layer to map those vocab indices into a space of dimensionality
        # 'embedding_dim'.
        embedded = layers.Embedding(input_dim=max_tokens, output_dim=64,mask_zero=True)(inputs)
        x = layers.Bidirectional(layers.LSTM(32))(embedded)

        # Dropout 
        x = layers.Dropout(0.5)(x)
        # Conv1D + global max pooling

        # x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
        # x = layers.GlobalMaxPooling1D()(x)

        # We add a vanilla hidden layer:
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

        outputs = layers.Dense(1, activation="sigmoid")(x)
        self.model = Model(inputs=inputs, outputs=outputs)
        #print(self.model.summary())

        
    def compile(self, optimizer, loss, metrics):
        """ Compiles the model """
        return self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics,
                           weighted_metrics=None,
                           run_eagerly=None,
                           steps_per_execution=None,
                           jit_compile=None)
    
    def train(self, train_dataset, batch_size=None, epochs=1,  validation_data=None, callbacks=None):
        """ Trains the model """
        return self.model.fit(train_dataset, batch_size=batch_size, epochs=epochs,  validation_data=validation_data, callbacks=callbacks)
    
    def evaluate(self, test_dataset):
        """Evaluates the model"""
        return self.model.evaluate(test_dataset)
    def save(self, file_path):
        """Saves the model"""
        self.model.save(file_path)
if __name__ == '__main__':
    model = TSModel()