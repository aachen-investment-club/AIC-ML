


import uuid
from datetime import datetime


def get_hash_folder_name(file:str): 

    date_str = datetime.now().strftime("%Y-%m-%d")
    short_hash = uuid.uuid4().hex[:8]
    return  f"{file}_{date_str}_{short_hash}" 