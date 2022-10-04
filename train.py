from data import DataLoader # the data
from model import TSModel # the model
import hydra  # for configurations
import tf2onnx # model conversion
import tensorflow as tf
from omegaconf.omegaconf import OmegaConf # configs
import matplotlib.pyplot as plt # plots
import mlflow # for tracking


#EXPERIMENT_NAME = "TweetsSentiment"
#EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
MLFLOW_TRACKING_URI="https://dagshub.com/Marshall-mk/TweetsSentimentProject.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.tensorflow.autolog()


@hydra.main(config_path="./configs", config_name="configs")
def main(cfg):
    OmegaConf.to_yaml(cfg, resolve=True)
    """defines the data and the model"""
    ts_data = DataLoader()
    ts_model = TSModel()
    max_length = 100
    max_tokens = 2000
    text_vectorization = tf.keras.layers.TextVectorization(max_tokens=max_tokens, output_mode="int", output_sequence_length=max_length,)
    text_only_train_ds = ts_data.load_train_data(cfg.model.Train_path).map(lambda x, y: x)
    text_vectorization.adapt(text_only_train_ds)

    train_ds = ts_data.load_train_data(cfg.model.Train_path).map(lambda x, y: (text_vectorization(x), y),num_parallel_calls=4)

    val_ds = ts_data.load_val_data(cfg.model.Val_path).map(lambda x, y: (text_vectorization(x), y),num_parallel_calls=4)

    test_ds = ts_data.load_test_data(cfg.model.Test_path).map(lambda x, y: (text_vectorization(x), y),num_parallel_calls=4)
    
    # we need to save the text vectorization layer in order to use it in the inference
    vectorize_layer_model = tf.keras.models.Sequential()
    vectorize_layer_model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    vectorize_layer_model.add(text_vectorization)
    filepath = cfg.model.vectorize_layer_path
    vectorize_layer_model.save(filepath, save_format="tf")

    """Compiles and trains the model"""
    ts_model.compile(optimizer= cfg.train.optimizer, loss = cfg.train.loss,  metrics= cfg.train.metrics)

    """Model callbacks"""
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=cfg.model.ckpt_path, save_weights_only=True, save_best_only=True)
    
    """Trains the model"""
    with mlflow.start_run():
    # ... Define a model
        model_info = ts_model.fit(
            train_ds,
            batch_size=cfg.train.batch_size,
            epochs=cfg.train.epochs,
            validation_data=val_ds,
            callbacks= [earlystopping, checkpointer]) 

        """Evaluates the model on the test set"""
        print(f'Model evaluation metrics: {ts_model.evaluate(test_ds)}')
        mlflow.end_run()
    
    """Saving the model"""

    ts_model.save(cfg.model.save_path)

    # """converting the model to onnx"""
    spec = (tf.TensorSpec((None,None,), tf.float32, name="input"),)
    output_path = cfg.model.onnx_path
    model_proto, _ = tf2onnx.convert.from_keras(ts_model, input_signature=spec, opset=13, output_path=output_path)
    
    """Model training history """
    _model_history(model_info=model_info, cfg=cfg)






def _model_history(model_info, cfg):
    accuracy = model_info.history["accuracy"]
    val_accuracy = model_info.history["val_accuracy"]
    loss = model_info.history["loss"]
    val_loss = model_info.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)
    plt.figure(figsize=(20,10))
    plt.plot(epochs, accuracy, "g-", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.grid()
    plt.savefig(f'{cfg.model.history_path}accuracy.png', dpi=300, bbox_inches='tight')

    plt.legend()

    plt.figure(figsize=(20,10))
    plt.plot(epochs, loss, "g-", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid()
    plt.savefig(f'{cfg.model.history_path}loss.png', bbox_inches='tight', dpi=300)
if __name__ == "__main__":
    main()
