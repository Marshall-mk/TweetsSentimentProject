# -*- coding: utf-8 -*-
"""Data Loader"""
import pandas as pd
import tensorflow as tf

class DataLoader():
    """Data Loader class"""
    def __init__(self):
        super().__init__()
    
    def load_train_data(self,path):
        """Loads dataset from path"""
        return self.preprocess_data(pd.read_csv(path))

    def load_val_data(self,path):
        """Loads dataset from path"""
        return self.preprocess_data(pd.read_csv(path))
    
    
    def load_test_data(self,path):
        """Loads dataset from path"""
        
        return self.preprocess_data(pd.read_csv(path))
    
    def preprocess_data(self,dataframe, shuffle=True, batch_size=256):
        """Preprocesses data"""
        df = dataframe.copy()
        labels = df.pop('target')
        features = df.pop('tweet')
        ds = tf.data.Dataset.from_tensor_slices((features.values, labels.values))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
        return ds


if __name__ == "__main__":
    data_model = DataLoader()

    
