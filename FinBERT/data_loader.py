

import torch
from torch.utils.data import Dataset



class DataFinbert(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        if "sentence" not in df.columns:
            self.sentences = df["text"].tolist()
        else: 
            self.sentences = df["sentence"].tolist()
        self.labels = df["label"].values.astype("float32")
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sentences[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
            torch.tensor(self.labels[idx])
        )


def convert_labels(df):
    """convert labels before regession"""

   

    df = df.copy()
    df["label"] = df["label"].map({
        0: -1.0,
        1:  0.0,
        2:  1.0
    }).astype("float32")
    return df


"""
def tokenize_batch(tokenizer, sentences):
    return tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
"""