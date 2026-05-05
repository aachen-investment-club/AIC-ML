from abc import ABC, abstractmethod
import boto3
import json

from strategies.interfaces import TradeLog



class Strategy(ABC): 


    strategy_name = "" 
    strategy_file_name = strategy_name + ".json"
    tradelog = None | TradeLog

    @staticmethod
    def __init_s3(): 
        client = boto3.client("s3")
        return client 



    @classmethod
    def get_tradelog(cls)-> None: 
        """retrieve the json tradelog; stores a dictionary"""
        client = Strategy.__init_s3()
        bucket = "portfolio-management-developer"
        prefix = "strategy_tradelogs/"
        objects = client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        response = client.get_object(
            Bucket=bucket,
            Key=cls.strategy_file_name
        )
        text = response["Body"].read().decode("utf-8")
        cls.tradelog = TradeLog.model_validate_json(text)



    @classmethod
    def update_tradelog(cls): 
        """after performing trades, update the json tradelog"""

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




