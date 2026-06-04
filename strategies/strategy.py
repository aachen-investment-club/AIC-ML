from abc import ABC, abstractmethod
import boto3
import json
import os
import shutil
from typing import List
from strategies.interfaces import TradeLog
from .util import get_hash_folder_name 
from typing import List, Optional
from .model_artifact_manager import ModelArtifactManager



class Strategy(ABC): 


    strategy_name = "" 
    strategy_file_name = strategy_name + ".json"
    tradelog : Optional[None | TradeLog] #: stores the current tradelog. 
    s3_bucket_name = "portfolio-management-developer" # Refactored hardcode

    explanation = "" 

    _artifact_manager: Optional[ModelArtifactManager] = None

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
        #objects = client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        response = client.get_object(
            Bucket=bucket,
            Key=prefix+cls.strategy_file_name
        )
        text = response["Body"].read().decode("utf-8")
        print(text)
        cls.tradelog = TradeLog.model_validate_json(text)



    @classmethod
    def upload_tradelog(cls)-> None: 
        """after performing trades, update the json tradelog"""
        client = Strategy.__get_s3_client()
        bucket = "portfolio-management-developer"
        prefix = "strategy_tradelogs"
        
        print(cls.tradelog)
        tradelog_json = cls.tradelog.model_dump_json()
        response = client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/{cls.strategy_file_name}",
            Body=tradelog_json.encode("utf-8"),
            ContentType="application/json",
        )

        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            return 
        else:
            print('upload failed')
            return 
            



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
    def get_manager(cls) -> ModelArtifactManager:
        """Lazy loads the artifact manager tailored to this specific strategy."""
        if cls._artifact_manager is None:
            cls._artifact_manager = ModelArtifactManager(
                strategy_name=cls.strategy_name, 
                model_extension=cls.get_model_extension()
            )
        return cls._artifact_manager

    @classmethod
    def get_model_extension(cls) -> str:
        return ""

    @classmethod
    def save_model(cls) -> None:
        """Uses manager to get paths, uses framework to save bytes."""
        manager = cls.get_manager()
        version_path, latest_path = manager.get_save_paths()
        
        cls._save_model_to_disk(version_path)
        cls._save_model_to_disk(latest_path)
        
        print(f"[{cls.strategy_name}] Model version saved to: {version_path}")
        print(f"[{cls.strategy_name}] Updated latest model at: {latest_path}")

    @classmethod
    def load_model(cls, version_folder: Optional[str] = None) -> None: 
        manager = cls.get_manager()
        load_path = manager.get_load_path(version_folder)
        
        cls._load_model_from_disk(load_path)
        print(f"[{cls.strategy_name}] Model successfully loaded from: {load_path}")
    



    @classmethod
    def train(cls) -> None: 
        """Template method: Handles artifact lifecycle automatically."""
        # 1. Setup: Start a fresh session
        cls.get_manager().reset_session()
        
        # 2. Execute ML logic (implemented by child class)
        print(f"[{cls.strategy_name}] Starting training...")
        train_results = cls._execute_train()
        
        # 3. Teardown: Save everything automatically
        cls.save_model()
        if train_results is not None:
            cls.get_manager().save_results(train_results, "train_results.json")
            
        print(f"[{cls.strategy_name}] Training complete.")


    @classmethod
    def test(cls, model_version: Optional[str] = None) -> None: 
        """Template method: tests the trained strategy"""
        # 1. Setup: Load the requested model
        cls.load_model(version_folder=model_version)
        
        # 2. Execute Testing/Trading logic (implemented by child class)
        print(f"[{cls.strategy_name}] Starting trading/testing execution...")
        test_results = cls._execute_test()
        
        # 3. Teardown: Save test metrics automatically
        if test_results is not None:
            cls.get_manager().save_results(test_results, "test_results.json", version_folder=model_version)


    @classmethod
    @abstractmethod
    def _save_model_to_disk(cls, path: str):
        pass


    @classmethod
    def trade(cls, *args, model_version: Optional[str] = None, **kwargs) -> any: 
        """
        Template method: Loads the trained model and executes a live trade.
        """
        cls.load_model(version_folder=model_version)
        
        print(f"[{cls.strategy_name}] Model loaded. Executing live trade...")
        trade_result = cls._execute_trade(*args, **kwargs)
        
            
        return trade_result



    @classmethod
    @abstractmethod
    def get_data_for_trade(cls, current_data): 
        """Get data for executing trades"""
        pass 

    @classmethod
    @abstractmethod
    def extract_features_for_trade(cls, current_data): 
        """Extract features for trading"""
        pass 

    @classmethod
    @abstractmethod
    def _load_model_from_disk(cls, path: str):
        pass




    @classmethod
    @abstractmethod
    def _execute_train(cls) -> dict:
        """Implement ML training logic. Must return a dictionary of metrics to save."""
        pass


    @classmethod
    @abstractmethod
    def _execute_test(cls) -> dict:
        """Implement ML testing/trading logic. Must return a dictionary of metrics to save."""
        pass



    @classmethod
    @abstractmethod
    def _execute_trade(cls, *args, **kwargs):
        """Implement live trading logic. Executed after the model is automatically loaded."""
        pass



