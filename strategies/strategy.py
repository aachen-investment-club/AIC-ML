from abc import ABC, abstractmethod
import boto3

class Strategy(ABC): 

    @classmethod
    def update_tradelog(cls): 
        """after performing trades, update the json tradelog"""

        return 

    @classmethod
    def get_tradelog(cls): 
        """retrieve the json tradelog"""
        return 

    @classmethod
    @abstractmethod
    def get_training_data(cls):
        """retrieve the data for training"""
        pass

    @classmethod
    @abstractmethod
    def extract_features(cls):
        """implement feature extraction for the model"""
        pass


    @classmethod
    @abstractmethod
    def train(cls): 
        """train the model. this method is supposed to be implemented for production training; not development"""
        pass


    @classmethod
    @abstractmethod
    def train_production(cls): 
        """train the model. this method is supposed to be implemented for production training; not development"""
        pass



    @classmethod
    @abstractmethod
    def trade(cls): 
        """given some features, perform the trades"""
        pass




