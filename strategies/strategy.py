from abc import ABC, abstractmethod
import boto3
import json

from strategies.interfaces import TradeLog



class Strategy(ABC): 


    strategy_name = "" 
    strategy_file_name = strategy_name + ".json"
    tradelog = None | TradeLog #: stores the current tradelog. 

    @staticmethod
    def __get_s3_client(): 
        client = boto3.client("s3")
        return client 



    @classmethod
    def get_tradelog(cls)-> None: 
        """retrieve the json tradelog; stores a dictionary"""
        client = Strategy.__get_s3_client()
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
    def upload_tradelog(cls)-> None: 
        """after performing trades, update the json tradelog"""
        client = Strategy.__get_s3_client()
        bucket = "portfolio-management-developer"
        prefix = "strategy_tradelogs/"
        
        response = client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/{cls.filename}",
            Body=json.dumps(cls.tradelog.model_dump_json(), ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )

        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            return 
        else:
            print('upload failed')
            return 
            

    @classmethod
    @abstractmethod
    def trade(cls): 
        """

        - when called, this should fetch the data, extract features and perform trades
        - new trades should added to the tradelog 
        
        """
        pass


    @classmethod
    @abstractmethod
    def get_training_data(cls):
        """retrieve the data for training"""
        pass

    @classmethod
    @abstractmethod
    def extract_features(cls):
        """implement feature extraction for the model; should be used for inference (=trading) and training"""
        pass


    @classmethod
    @abstractmethod
    def train(cls): 
        """
        
        train the model. this method is supposed to be implemented for production training; not development

        get the data, extract features, train the model 
        
        
        """
        pass







