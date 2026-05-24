from huggingface_hub import login, upload_folder
from huggingface_hub import hf_hub_download

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


#: this script is for uploading to HF.

def publish():
    login()
    upload_folder(folder_path="./regression_finance_finetune_gbert_distil", 
                repo_id="Aachen-Investment-Club/Distil-Gbert-Regression", 
              repo_type="model")




def test(): 
    repo_id = "Aachen-Investment-Club/Finance-Finetune-distil-GBert-Finetune-Regression"  

    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)


    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    model.eval()

    text = "example input text"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.squeeze().item()  # scalar regression output

    print("prediction:", pred)

def test2():
    p = hf_hub_download("Aachen-Investment-Club/Finance-Finetune-distil-GBert-Finetune-Regression", "pytorch_model.bin")
    sd = torch.load(p, map_location="cpu")
    print(list(sd.keys())[:30])



if __name__ =="__main__": 
    test2()