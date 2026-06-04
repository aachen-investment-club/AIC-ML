from abc import ABC, abstractmethod
import boto3
import json
import os
import shutil
from typing import List
from strategies.interfaces import TradeLog
from .util import get_hash_folder_name 
from typing import List, Optional



class Strategy(ABC): 


    strategy_name = "" 
    strategy_file_name = strategy_name + ".json"
    tradelog = None | TradeLog #: stores the current tradelog. 

    explanation = "" 

    _current_version_folder: Optional[str] = None

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
    def get_model_extension(cls) -> str:
        """Override this in subclasses to return the framework's file extension (e.g., '.keras', '.pt')."""
        return ""



    @classmethod
    def get_base_model_dir(cls) -> str:
        """Returns the root directory for this specific strategy's weights."""
        return os.path.join(".", "training_outputs", cls.strategy_name)


    @classmethod
    def reset_version_session(cls) -> None:
        """Call this at the start of a training run to generate a fresh date-hash."""
        cls._current_version_folder = None


    @classmethod
    def _get_or_create_version_folder(cls) -> str:
        if not cls._current_version_folder:
            cls._current_version_folder = get_hash_folder_name(f"{cls.strategy_name}_weights")
        return cls._current_version_folder


    @classmethod
    def get_save_paths(cls) -> tuple[str, str]:
        """Returns the specific file paths for saving the model weights."""
        version_dir, latest_dir = cls.get_save_dirs()
        filename = f"{cls.strategy_name}{cls.get_model_extension()}"
        return os.path.join(version_dir, filename), os.path.join(latest_dir, filename)




    @classmethod
    def get_save_paths(cls) -> tuple[str, str]:
        """Creates necessary directories and returns (versioned_path, latest_path)."""
        version_dir, latest_dir = cls.get_save_dirs()
        filename = f"{cls.strategy_name}{cls.get_model_extension()}"
        return os.path.join(version_dir, filename), os.path.join(latest_dir, filename)

    @classmethod
    def get_load_path(cls, version_folder: Optional[str] = None) -> str:
        """Resolves the load path. Defaults to 'latest' if no version_folder is provided."""
        

        base_dir = cls.get_base_model_dir()
        filename = f"{cls.strategy_name}{cls.get_model_extension()}"
        
        if version_folder:
            path = os.path.join(base_dir, version_folder, filename)
        else:
            path = os.path.join(base_dir, "latest", filename)
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
            
        return path

    

    @classmethod
    def load_model(cls, version_folder: Optional[str] = None) -> None: 
        """Template method: Handles finding the right path, calls framework-specific load logic."""
        load_path = cls.get_load_path(version_folder)
        cls._load_model_from_disk(load_path)
        print(f"[{cls.strategy_name}] Model successfully loaded from: {load_path}")

    @classmethod
    def save_model(cls) -> None:
        """Template method: Handles file system tracking, calls framework-specific save logic."""
        version_path, latest_path = cls.get_save_paths()
        
        # We call the concrete save method twice so the framework safely handles 
        # the I/O for both paths (works for both single files and directory formats like TF SavedModel).
        cls._save_model_to_disk(version_path)
        cls._save_model_to_disk(latest_path)
        
        print(f"[{cls.strategy_name}] Model version saved to: {version_path}")
        print(f"[{cls.strategy_name}] Updated latest model at: {latest_path}")


    @classmethod
    def save_results(cls, results_dict: dict, filename: str, version_folder: Optional[str] = None) -> None:
        """
        Saves a dictionary as a JSON file. 
        If version_folder is specified (e.g. during testing an old model), saves there.
        Otherwise, saves to both the current training session's folder and 'latest'.
        """
        base_dir = cls.get_base_model_dir()
        dirs_to_save = []
        
        if version_folder:
            dirs_to_save.append(os.path.join(base_dir, version_folder))
        else:
            version_dir, latest_dir = cls.get_save_dirs()
            dirs_to_save.extend([version_dir, latest_dir])
            
        for d in dirs_to_save:
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, filename)
            with open(path, "w") as f:
                # Convert numpy types to standard python types if necessary, then save
                json.dump(results_dict, f, indent=4)
            print(f"[{cls.strategy_name}] Saved {filename} to: {path}")

    @classmethod
    def get_save_dirs(cls) -> tuple[str, str]:
        """Returns (version_dir, latest_dir) keeping the same hash for the current run."""
        base_dir = cls.get_base_model_dir()
        folder_name = cls._get_or_create_version_folder()
        
        version_dir = os.path.join(base_dir, folder_name)
        latest_dir = os.path.join(base_dir, "latest")
        
        os.makedirs(version_dir, exist_ok=True)
        os.makedirs(latest_dir, exist_ok=True)
        
        return version_dir, latest_dir



    @classmethod
    @abstractmethod
    def _save_model_to_disk(cls, path: str):
        """Implement framework-specific saving logic (e.g., model.save(path) or torch.save(model, path))."""
        pass

    @classmethod
    @abstractmethod
    def _load_model_from_disk(cls, path: str):
        """Implement framework-specific loading logic (e.g., load_model(path))."""
        pass




    @classmethod
    @abstractmethod
    def train(cls): 
        """
        
        train the model. this method is supposed to be implemented for production training; not development

        get the data, extract features, train the model 
        
        
        """
        pass







