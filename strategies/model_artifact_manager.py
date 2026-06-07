import os
import json
from typing import Optional, Tuple
from .util import get_hash_folder_name 

class ModelArtifactManager:
    """
    this class is in charge of file management. 

    we create a folder for each strategy, and this folder contains 
    each version of the training strategy (using the date and a hash as naming conventions). 

    we also store the training results. 
    """
    def __init__(self, strategy_name: str, model_extension: str):
        self.strategy_name = strategy_name
        self.model_extension = model_extension
        self.base_dir = os.path.join(".", "training_outputs", self.strategy_name)
        self._current_version_folder: Optional[str] = None

    def reset_session(self) -> None:
        """Call this at the start of a training run to generate a fresh date-hash."""
        self._current_version_folder = None

    def _get_or_create_version_folder(self) -> str:
        if not self._current_version_folder:
            self._current_version_folder = get_hash_folder_name(f"{self.strategy_name}_weights")
        return self._current_version_folder

    def get_save_dirs(self) -> Tuple[str, str]:
        """Returns (version_dir, latest_dir) keeping the same hash for the current run."""
        folder_name = self._get_or_create_version_folder()
        
        version_dir = os.path.join(self.base_dir, folder_name)
        latest_dir = os.path.join(self.base_dir, "latest")
        
        os.makedirs(version_dir, exist_ok=True)
        os.makedirs(latest_dir, exist_ok=True)
        
        return version_dir, latest_dir

    def get_save_paths(self) -> Tuple[str, str]:
        """Returns the specific file paths for saving the model weights."""
        version_dir, latest_dir = self.get_save_dirs()
        filename = f"{self.strategy_name}{self.model_extension}"
        return os.path.join(version_dir, filename), os.path.join(latest_dir, filename)

    def get_load_path(self, version_folder: Optional[str] = None) -> str:
        """Resolves the load path. Defaults to 'latest' if no version_folder is provided."""
        filename = f"{self.strategy_name}{self.model_extension}"
        
        if version_folder:
            path = os.path.join(self.base_dir, version_folder, filename)
        else:
            path = os.path.join(self.base_dir, "latest", filename)
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
            
        return path

    def save_results(self, results_dict: dict, filename: str, version_folder: Optional[str] = None) -> None:
        """Saves a dictionary as a JSON file."""
        dirs_to_save = []
        
        if version_folder:
            dirs_to_save.append(os.path.join(self.base_dir, version_folder))
        else:
            version_dir, latest_dir = self.get_save_dirs()
            dirs_to_save.extend([version_dir, latest_dir])
            
        for d in dirs_to_save:
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, filename)
            with open(path, "w") as f:
                json.dump(results_dict, f, indent=4)
            print(f"[{self.strategy_name}] Saved {filename} to: {path}")

    def save_lines(self, lines: list, filename: str, version_folder: Optional[str] = None) -> None:
        """Saves a list of strings as individual lines in a text file."""
        dirs_to_save = []
        
        if version_folder:
            dirs_to_save.append(os.path.join(self.base_dir, version_folder))
        else:
            version_dir, latest_dir = self.get_save_dirs()
            dirs_to_save.extend([version_dir, latest_dir])
            
        for d in dirs_to_save:
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, filename)
            
            with open(path, "w", encoding="utf-8") as f:
                # Strip existing newlines to prevent double-spacing, then add a newline to each
                for line in lines:
                    f.write(f"{str(line).rstrip()}\n")
                    
            print(f"[{self.strategy_name}] Saved {filename} to: {path}")




