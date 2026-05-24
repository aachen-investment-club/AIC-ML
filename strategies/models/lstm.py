from ..strategy import Strategy

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout



class LSTMStrategy(Strategy): 


    strategy_name = "LSTM_strategy" 
    strategy_file_name = strategy_name + ".json"
    explanation = "some LSTM strategy" 




    @classmethod
    def trade(cls): 
        """

        - when called, this should fetch the data, extract features and perform trades
        - new trades should added to the tradelog 
        
        """
        pass


    @classmethod
    def get_training_data(cls):
        """retrieve the data for training"""
        cls.data = ... #TODO: connect with portoflio/ use some csv file , etc...

    @classmethod
    def extract_features(cls):
        """implement feature extraction for the model; should be used for inference (=trading) and training"""
        cls.scaler = MinMaxScaler(feature_range = (0,1))

        scaled_data = cls.scaler.fit_tranform(cls.data)
        sequence_length = 60

        X, y = [], []

        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])

        X = np.array(X)
        y = np.array(y)

        X = np.reshape(X, (X.shape[0], X.shape[1], 1))


        train_size = int(len(X) * 0.8)

        cls.X_train = X[:train_size]
        cls.X_test = X[train_size:]

        cls.y_train = y[:train_size]
        cls.y_test = y[train_size:]


        


    @classmethod
    def train(cls): 
        """
        
        train the model. this method is supposed to be implemented for production training; not development

        get the data, extract features, train the model 
        
        
        """

        history = cls.model.fit(
            cls.X_train,
            cls.y_train,
            epochs=10,
            batch_size=32,
            validation_data=(cls.X_test, cls.y_test),
            verbose=1
        )




    @classmethod 
    def initialize_architecture(cls, input_dim): 

        cls.model = Sequential()

        cls.model.add(LSTM(units=64, return_sequences=True,
                    input_shape=(input_dim, 1)))

        cls.model.add(Dropout(0.2))

        cls.model.add(LSTM(units=64))
        cls.model.add(Dropout(0.2))

        cls.model.add(Dense(1))

        cls.model.compile(optimizer="adam", loss="mean_squared_error")