import torch
import torch.nn as nn
from transformers import AutoModel , AutoTokenizer
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from ML.Models.FinBERT.data_loader import DataFinbert, convert_labels
from ML.Models.FinBERT.model import  RegressionGFinBERT, load_trained_model, DistilRegressionGFinBERT, GermanRegressionFinBERTDistil, EnglishRegressionFinBERTDistil
from ML.Models.FinBERT.train_test import train, test, train_cross_validation, grid_search_cross_validation, predict, train_knowledge_distillation
from ML.Models.FinBERT.config import load_env, get_float, get_int, get_str
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import zipfile, os
import inspect

from pathlib import Path
from datasets import Dataset, DatasetDict
from datasets import load_from_disk

import transformers
from transformers import (
        AutoTokenizer, AutoModelForMaskedLM,
        DataCollatorForLanguageModeling,
        #TrainingArguments, Trainer
    )





load_env()





# some hyperparams 
MAX_LEN = get_int("MAX_LEN", 128)
EPOCHS = get_int("EPOCHS", 20)
LR = get_float("LR", 2e-5)
SAVE_DIR = get_str("SAVE_DIR", "./regression_gbert")
NUM_GPUS = torch.cuda.device_count()
BATCH_SIZE = get_int("BATCH_SIZE", 16) * max(1, NUM_GPUS)

CV_EPOCHS = get_int("CV_EPOCHS", 20)


tokenizer_GBERT = AutoTokenizer.from_pretrained("deepset/gbert-base")
tokenizer_GFINBERT = AutoTokenizer.from_pretrained("scherrmann/GermanFinBert_SC")





def test_grid_search_english(model_class):
    # used for fine tuning distilbert-base-cased on the MLM task (english)

    params = {
        "learning_rate": [1e-5],
        "dropout": [0.2],
        "unfreeze_freq": [2],
        "norm_clip": [1]
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    splits = {
        'train': 'data/train-00000-of-00001.parquet',
        'validation': 'data/validation-00000-of-00001.parquet',
        'test': 'data/test-00000-of-00001.parquet'
    }

    # English tokenizer (correct)
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert/distilbert-base-uncased"
    )

    train_df = convert_labels(
        pd.read_parquet(
            "hf://datasets/FinanceMTEB/financial_phrasebank/" + splits["train"]
        )
    )

    test_df = convert_labels(
        pd.read_parquet(
            "hf://datasets/FinanceMTEB/financial_phrasebank/" + splits["test"]
        )
    )

    grid_search_cross_validation(
        model_class=model_class,
        params=params,
        tokenizer=tokenizer,
        train_df=train_df,
        test_df=test_df,
        folds=5,
        BATCH_SIZE=BATCH_SIZE,
        device=device,
        bert_layers=6,
        bert_has_pooler=False,
        epochs=EPOCHS,
        cv_epochs=CV_EPOCHS,
    )



def test_kd(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    repo_id = "anhaltai/fincorpus-de-10k"

    zip_path = hf_hub_download(repo_id=repo_id, filename="data/corpus_safe_txt_only.zip")
    meta_path = hf_hub_download(repo_id=repo_id, filename="data/metadata.csv")

    out_dir = "./fincorpus_txt"
    os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

    meta = pd.read_csv(meta_path)
    print(meta.columns)
    print(len(meta))




"""def fine_tune_distil_gbert_mini():
    dataset = load_from_disk("data/fincorpus_datasetdict")

    # ---- SMOKE TEST SUBSET ----
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
    # --------------------------

    model_name = "distilbert/distilbert-base-german-cased" 
    tokenizer = AutoTokenizer.from_pretrained(model_name) 
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,          # better: dynamic padding
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
            padding=False,          # better: dynamic padding
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
        per_device_train_batch_size=16,   # adjust to your GPU
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



"""


def fine_tune_distil_gbert_import_data_to_disk(): 

    ZIP_PATH = "data/corpus_safe_txt_only (1).zip"
    META_PATH = "data/metadata.csv"
    EXTRACT_DIR = "data/corpus_safe_txt_only (1)/txt"



    txt_files = list(Path(EXTRACT_DIR).rglob("*.txt"))

    txt_map = {p.name: str(p) for p in txt_files}          
    txt_stems = {p.stem for p in txt_files}                

    meta = pd.read_csv(META_PATH)
    meta_cols = list(meta.columns)

    TXT_COL = meta.columns[1]  # <-- uses the 2nd column shown in your head()
    print(meta.head())
    print("Using TXT_COL =", TXT_COL)

    def read_text(rel_path):

        EXTRACT_DIR = "data/corpus_safe_txt_only (1)"
        if pd.isna(rel_path):
            return None
        rel_path = str(rel_path).replace("/", os.sep)  # Windows-friendly
        full_path = os.path.join(EXTRACT_DIR, rel_path)

        if not os.path.exists(full_path):
            return None

        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    
    meta["text"] = meta[TXT_COL].apply(read_text)
    meta = meta.dropna(subset=["text"]).reset_index(drop=True)

    print("Loaded docs:", len(meta))


    ds = Dataset.from_pandas(meta, preserve_index=False).shuffle(seed=42)
    splits = ds.train_test_split(test_size=0.02, seed=42)
    dataset = DatasetDict(train=splits["train"], validation=splits["test"])
    dataset.save_to_disk("data/fincorpus_datasetdict")


    print(dataset)


def test_kd_mini(): 
    #: just for testing the pipeline, take a small subset of the data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}



    train_df = convert_labels(
        pd.read_parquet("hf://datasets/scherrmann/financial_phrasebank_75agree_german/" + splits["train"])
    )
    train_df = train_df.head(20)

    test_df = convert_labels(
        pd.read_parquet("hf://datasets/scherrmann/financial_phrasebank_75agree_german/" + splits["test"])
    )
    test_df= test_df.head(20) #: small data subset.

    teacher, tokenizer, _ = load_trained_model("./output_models/", 
                                                      RegressionGFinBERT)
                                            

    train_dataset = DataFinbert(train_df, tokenizer)
    test_dataset = DataFinbert(test_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    student = DistilRegressionGFinBERT()

    train_results = train_knowledge_distillation(
        teacher=teacher,
        student= student, 
        train_loader = train_loader, 
        epochs = EPOCHS, 
        learning_rate= LR, 
        kd_weight = 0.5, 
        truth_weight= 0.5, 
        device= device
    )

    train_mse, train_mae = train_results["MSE"], train_results["MAE"]

    test_results = test(
        model=student,
        test_loader=test_loader,
        device=device
    )

    test_mse, test_mae = test_results["mse"], test_results["mae"]


    train_test_metrics = {
        "type": ["train", "test"], 
        "mse": [train_mse, test_mse],
        "mae": [train_mae, test_mae]
    }

    train_test_metrics = pd.DataFrame(train_test_metrics)
    train_test_metrics.to_csv(f"{SAVE_DIR}/train_test_metrics.csv")


    student.eval()
    torch.save(student.state_dict(), f"{SAVE_DIR}/pytorch_model.bin")
    tokenizer.save_pretrained(SAVE_DIR)


    train_test_metrics.to_csv(f"{SAVE_DIR}/train_test_metrics.csv")


    torch.save({
        "model_state_dict": student.state_dict(),
        "epochs": EPOCHS,
        "learning_rate": LR,
        "max_len": MAX_LEN,
        "label_mapping": {0: -1.0, 1: 0.0, 2: 1.0}
    }, f"{SAVE_DIR}/checkpoint.pt")








def test_load_model(): 
    #: for testing inference in the regression task; for REgressionGFinBERT
    model, tokenizer, checkpoint = load_trained_model("./output_models/", 
                                                      RegressionGFinBERT)
    print(model)
    print(
        predict(
        "Die NVIDIA-Aktie hat diese Woche Rekordhohen erreicht, und Anleger reagieren positiv. ",
        model, 
        tokenizer
        )
    )


def test_grid_search_distillation():
    #: for distillation.
    train_df = convert_labels(
        pd.read_parquet("hf://datasets/scherrmann/financial_phrasebank_75agree_german/" + splits["train"])
    )

    test_df = convert_labels(
        pd.read_parquet("hf://datasets/scherrmann/financial_phrasebank_75agree_german/" + splits["test"])
    )

    teacher, tokenizer, _ = load_trained_model("./output_models/", 
                                                      RegressionGFinBERT)
                                            

    train_dataset = DataFinbert(train_df, tokenizer)
    test_dataset = DataFinbert(test_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    student = DistilRegressionGFinBERT()

    train_results = train_knowledge_distillation(
        teacher=teacher,
        student= student, 
        train_loader = train_loader, 
        epochs = EPOCHS, 
        learning_rate= LR, 
        kd_weight = 0.5, 
        truth_weight= 0.5, 
        device= device
    )

    train_mse, train_mae = train_results["MSE"], train_results["MAE"]

    test_results = test(
        model=student,
        test_loader=test_loader,
        device=device
    )

    test_mse, test_mae = test_results["mse"], test_results["mae"]


    train_test_metrics = {
        "type": ["train", "test"], 
        "mse": [train_mse, test_mse],
        "mae": [train_mae, test_mae]
    }

    train_test_metrics = pd.DataFrame(train_test_metrics)
    train_test_metrics.to_csv(f"{SAVE_DIR}/train_test_metrics.csv")


    student.eval()
    torch.save(student.state_dict(), f"{SAVE_DIR}/pytorch_model.bin")
    tokenizer.save_pretrained(SAVE_DIR)


    train_test_metrics.to_csv(f"{SAVE_DIR}/train_test_metrics.csv")


    torch.save({
        "model_state_dict": student.state_dict(),
        "epochs": EPOCHS,
        "learning_rate": LR,
        "max_len": MAX_LEN,
        "label_mapping": {0: -1.0, 1: 0.0, 2: 1.0}
    }, f"{SAVE_DIR}/checkpoint.pt")





def test_grid_search(model_class):
    #: grid - CV search implementation, used in regression context.
    params = {
        "learning_rate": [1e-5], 
        "dropout": [0.2], 
        "unfreeze_freq": [1], 
        "norm_clip": [1]
    }





    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device )
    splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    tokenizer = AutoTokenizer.from_pretrained("./distilbert-german-finance-mlm")

    #tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")
    #tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-german-cased")


    train_df = convert_labels(
        pd.read_parquet("hf://datasets/scherrmann/financial_phrasebank_75agree_german/" + splits["train"])
    )

    test_df = convert_labels(
        pd.read_parquet("hf://datasets/scherrmann/financial_phrasebank_75agree_german/" + splits["test"])
    )


    grid_search_cross_validation(
        #model_class = None, 
        model_class = model_class, 
        #model = None, 
        params = params,
        tokenizer = tokenizer, 
        train_df= train_df, 
        test_df = test_df, 
        folds = 5, 
        BATCH_SIZE=BATCH_SIZE, 
        device=device, 
        bert_layers = 6, 
        bert_has_pooler = False, 
        epochs= EPOCHS, 
        cv_epochs = CV_EPOCHS,
    )


def test_grid_search_mini(model_class, tokenizer):
    #: grid search - CV  on small data subset

    print("started test run")
    params = {
        "learning_rate": [1e-5], 
        "dropout": [0.2], 
        "unfreeze_freq": [2], 
        "norm_clip": [1]
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}

    #tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-german-cased")

    train_df = convert_labels(
        pd.read_parquet("hf://datasets/scherrmann/financial_phrasebank_75agree_german/" + splits["train"])
    )

    train_df = train_df.head(20)

    test_df = convert_labels(
        pd.read_parquet("hf://datasets/scherrmann/financial_phrasebank_75agree_german/" + splits["test"])
    )

    """
    train_df = convert_labels(
        pd.read_parquet("hf://datasets/scherrmann/financial_phrasebank_75agree_german/" + splits["train"])
    )

    train_df = train_df.head(20)

    test_df = convert_labels(
        pd.read_parquet("hf://datasets/scherrmann/financial_phrasebank_75agree_german/" + splits["test"])
    )
    """
    test_df = test_df.head(20)

    grid_search_cross_validation(
        model = model_class, 
        params = params,
        tokenizer = tokenizer, 
        train_df= train_df, 
        test_df = test_df, 
        folds = 2, 
        BATCH_SIZE=BATCH_SIZE, 
        epochs= 2, 
        device=device, 
        bert_layers = 6, 
        bert_has_pooler = False, 
        cv_epochs = CV_EPOCHS,
    )

def test_cross_validation(model_class, tokenizer):
    # only CV training (regression) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}



    train_df = convert_labels(
        pd.read_parquet("hf://datasets/scherrmann/financial_phrasebank_75agree_german/" + splits["train"])
    )

    test_df = convert_labels(
        pd.read_parquet("hf://datasets/scherrmann/financial_phrasebank_75agree_german/" + splits["test"])
    )


    train_cross_validation(
        model_class, 
        tokenizer, 
        train_df, 
        5, 
        BATCH_SIZE, 
        epochs=CV_EPOCHS, 
        learning_rate=LR, 
        device = device
    ) 













def test_small_validation(model_class, tokenizer):
    #: validation trainingon small subset of the data. 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}



    train_df = convert_labels(
        pd.read_parquet("hf://datasets/scherrmann/financial_phrasebank_75agree_german/" + splits["train"])
    )
    train_df = train_df.head(500)

    test_df = convert_labels(
        pd.read_parquet("hf://datasets/scherrmann/financial_phrasebank_75agree_german/" + splits["test"])
    )
    test_df= test_df.head(20)


    train_cross_validation(
        model_class, 
        tokenizer, 
        train_df, 
        5, 
        BATCH_SIZE, 
        2, 
        LR, 
        device
    ) 








def test_small_dataset(model_class, tokenizer):

    splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}



    train_df = convert_labels(
        pd.read_parquet("hf://datasets/scherrmann/financial_phrasebank_75agree_german/" + splits["train"])
    )
    train_df = train_df.head(20)

    test_df = convert_labels(
        pd.read_parquet("hf://datasets/scherrmann/financial_phrasebank_75agree_german/" + splits["test"])
    )
    test_df= test_df.head(20)


    train_dataset = DataFinbert(train_df, tokenizer)
    test_dataset = DataFinbert(test_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class()


    train(
        model=model,
        train_loader=train_loader,
        epochs=EPOCHS,
        learning_rate=LR,
        device=device
    )

    test(
        model=model,
        test_loader=test_loader,
        device=device
    )








def train_test_full(model_class, tokenizer):
    # train-test, no CV nor gridsearch. (regression)
    splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}




    train_df = convert_labels(
        pd.read_parquet("hf://datasets/scherrmann/financial_phrasebank_75agree_german/" + splits["train"])
    )

    test_df = convert_labels(
        pd.read_parquet("hf://datasets/scherrmann/financial_phrasebank_75agree_german/" + splits["test"])
    )

    train_dataset = DataFinbert(train_df, tokenizer)
    test_dataset = DataFinbert(test_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class()


    train(
        model=model,
        train_loader=train_loader,
        epochs=EPOCHS,
        learning_rate=LR,
        device=device
    )

    test(
        model=model,
        test_loader=test_loader,
        device=device
    )


    model.eval()
    torch.save(model.state_dict(), f"{SAVE_DIR}/pytorch_model.bin")
    tokenizer.save_pretrained(SAVE_DIR)

    torch.save({
        "model_state_dict": model.state_dict(),
        "epochs": EPOCHS,
        "learning_rate": LR,
        "max_len": MAX_LEN,
        "label_mapping": {0: -1.0, 1: 0.0, 2: 1.0}
    }, f"{SAVE_DIR}/checkpoint.pt")




def test_gfinbert(model_class, tokenizer): 
    splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}




    train_df = convert_labels(
        pd.read_parquet("hf://datasets/scherrmann/financial_phrasebank_75agree_german/" + splits["train"])
    )
    train_df = train_df.head(20)

    test_df = convert_labels(
        pd.read_parquet("hf://datasets/scherrmann/financial_phrasebank_75agree_german/" + splits["test"])
    )
    test_df= test_df.head(20)


    train_dataset = DataFinbert(train_df, tokenizer)
    test_dataset = DataFinbert(test_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_class()



    train(
        model=model,
        train_loader=train_loader,
        epochs=EPOCHS,
        learning_rate=LR,
        device=device
    )

    test(
        model=model,
        test_loader=test_loader,
        device=device
    )






if __name__== "__main__": 
    #train_test_full(GermanRegressionFinBERT, tokenizer_GBERT)
    #print(f"GPUS: {torch.cuda.device_count()}")
    #test_small_validation(GermanRegressionFinBERT, tokenizer_GBERT)
    #test_cross_validation(GermanRegressionFinBERT, tokenizer_GBERT)
    #test_grid_search_mini(GermanRegressionFinBERT, tokenizer_GBERT)
    #test_grid_search(GermanRegressionFinBERT, tokenizer_GBERT)
    #test_gfinbert(RegressionGFinBERT, tokenizer_GFINBERT)
    #test_small_validation(RegressionGFinBERT, tokenizer_GFINBERT)
    #test_load_model()
    #test_kd()
    #test_kd_mini()
    #test_grid_search(GermanRegressionFinBERTDistil)
    #fine_tune_distil_gbert_mini()
    #fine_tune_distil_gbert()
    #test_grid_search(None)
    #test_grid_search(GermanRegressionFinBERTDistil)
    test_grid_search_english(EnglishRegressionFinBERTDistil)
