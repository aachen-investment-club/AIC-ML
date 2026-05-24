import torch.nn as nn
from transformers import AutoModel , AutoTokenizer
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os

from pathlib import Path
from datasets import Dataset, DatasetDict
from datasets import load_from_disk

import transformers
from transformers import (
        AutoTokenizer, AutoModelForMaskedLM,
        DataCollatorForLanguageModeling,
        TrainingArguments, Trainer
    )







def fine_tune_distil_gbert_mini():
    dataset = load_from_disk("data/fincorpus_datasetdict")

    train_size = 2000
    eval_size = 500

    train_size = min(train_size, len(dataset["train"]))
    eval_size  = min(eval_size,  len(dataset["validation"]))

    dataset = dataset.shuffle(seed=42)
    dataset["train"] = dataset["train"].select(range(train_size))
    dataset["validation"] = dataset["validation"].select(range(eval_size))





    dataset = dataset.shuffle(seed=42)
    dataset["train"] = dataset["train"].select(range(train_size))
    dataset["validation"] = dataset["validation"].select(range(eval_size))

    model_name = "distilbert/distilbert-base-german-cased" 
    tokenizer = AutoTokenizer.from_pretrained(model_name) 
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,          
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    print(dataset)
    
    args = TrainingArguments(
        output_dir="distilbert-german-finance-mlm-smoketest",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        fp16=False,
        report_to="none",
)

    print("Sample text:")
    print(dataset["train"][0]["text"][:300])

    print("Tokenized example keys:")
    print(tokenized["train"][0].keys())

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        #eval_dataset=tokenized["validation"],
        data_collator=collator,
    )
    print("started training")
    trainer.train()

    trainer.save_model("./distilbert-german-finance-mlm-smoketest")
    tokenizer.save_pretrained("./distilbert-german-finance-mlm-smoketest")




    





def fine_tune_distil_gbert(): 
    dataset = load_from_disk("data/fincorpus_datasetdict")

    model_name = "distilbert/distilbert-base-german-cased" 
    tokenizer = AutoTokenizer.from_pretrained(model_name) 
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,          
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    args = TrainingArguments(
        output_dir="distilbert-german-finance-mlm",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,   
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=200,
        eval_strategy="steps",
        eval_steps=2000,
        save_steps=2000,
        save_total_limit=2,
        fp16=True,                        # set False if CPU / no CUDA
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
    )
    print("started training")
    trainer.train()

    trainer.save_model("./distilbert-german-finance-mlm")
    tokenizer.save_pretrained("./distilbert-german-finance-mlm")





def fine_tune_distil_gbert_import_data_to_disk(): 
    #: for preprocessing the fincorpus-de-10k corpus

    ZIP_PATH = "data/corpus_safe_txt_only (1).zip"
    META_PATH = "data/metadata.csv"
    EXTRACT_DIR = "data/corpus_safe_txt_only (1)/txt"



    txt_files = list(Path(EXTRACT_DIR).rglob("*.txt"))

    txt_map = {p.name: str(p) for p in txt_files}         
    txt_stems = {p.stem for p in txt_files}                 

    meta = pd.read_csv(META_PATH)
    meta_cols = list(meta.columns)
    print(meta_cols)

    TXT_COL = meta.columns[1] 
    print(meta["language"].head())
    print("Using TXT_COL =", TXT_COL)

    def read_text(rel_path):

        EXTRACT_DIR = "data/corpus_safe_txt_only (1)"
        if pd.isna(rel_path):
            return None
        rel_path = str(rel_path).replace("/", os.sep)  
        full_path = os.path.join(EXTRACT_DIR, rel_path)

        if not os.path.exists(full_path):
            return None

        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    print("filtering:")
    print(len(meta))
    meta = meta[meta["language"]=="DE"]
    print(len(meta))
    meta["text"] = meta[TXT_COL].apply(read_text)
    meta = meta.dropna(subset=["text"]).reset_index(drop=True)

    print("Loaded docs:", len(meta))


    ds = Dataset.from_pandas(meta, preserve_index=False).shuffle(seed=42)
    splits = ds.train_test_split(test_size=0.02, seed=42)
    dataset = DatasetDict(train=splits["train"], validation=splits["test"])
    dataset.save_to_disk("data/fincorpus_datasetdict")


    print(dataset)









if __name__== "__main__": 
    fine_tune_distil_gbert()
    #fine_tune_distil_gbert_import_data_to_disk()