
from transformers import AutoModel , AutoTokenizer, AutoModelForMaskedLM

import torch.nn as nn
import torch
from transformers import AutoTokenizer
import copy






def load_trained_model(save_dir, model_class):
    checkpoint_path = f"{save_dir}/checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    model = model_class() 
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() 
    
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    
    print(f"Model and Tokenizer loaded successfully from {save_dir}")
    return model, tokenizer, checkpoint



class EnglishRegressionFinBERTDistil(nn.Module):
    #: distilled BERT (not finbert!), with regression head
    def __init__(self, dropout = 0.1):
        super().__init__()

        self.bert= AutoModel.from_pretrained("distilbert/distilbert-base-uncased")

        self.drop = nn.Dropout(p=dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)
        for param in self.bert.parameters():
            param.requires_grad = False #: this ensure that bert is frozen!! (pretrained modes are not frozen by default!)

    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        #: pooler_output is the CLS token (see BERT paper). 
        pooled_output = outputs.last_hidden_state[:,0]
        
        dropped= self.drop(pooled_output)
        return self.out(dropped)
    
    def unfreeze_bert_layers(self, num_layers): 
        
        total_layers = 6

        for i in range(total_layers - num_layers, total_layers):
            for param in self.bert.transformer.layer[i].parameters():
                param.requires_grad = True




class GermanRegressionFinBERTDistil(nn.Module):
    def __init__(self, dropout = 0.1):
        super().__init__()
        # Load the plain BERT engine --> this is GERMAN bert

        #: other alternatives: 
        #self.bert = AutoModel.from_pretrained("deepset/gbert-base", dtype="auto")
        #self.bert= AutoModel.from_pretrained("distilbert/distilbert-base-german-cased")
        #: here we use our own distilled bert fine tuned for MLM
        self.bert = AutoModel.from_pretrained("./distilbert-german-finance-mlm")

        self.drop = nn.Dropout(p=dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)
        for param in self.bert.parameters():
            param.requires_grad = False #: this ensure that bert is frozen!! (pretrained modes are not frozen by default!)

    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:,0]
        
        dropped= self.drop(pooled_output)
        return self.out(dropped)
    
    def unfreeze_bert_layers(self, num_layers): 
        
        total_layers = 6

        for i in range(total_layers - num_layers, total_layers):
            for param in self.bert.transformer.layer[i].parameters():
                param.requires_grad = True





class RegressionGFinBERT(nn.Module):
    def __init__(self, dropout = 0.1):
        super(RegressionGFinBERT, self).__init__()
        #: here without distillation
        self.bert = AutoModel.from_pretrained("scherrmann/GermanFinBert_SC")
        
        hidden_size = self.bert.config.hidden_size
        
        self.out= nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)  
        )
        for param in self.bert.parameters():
            param.requires_grad = False #: this ensure that bert is frozen!! (pretrained modes are not frozen by default!)



    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled_output = outputs.pooler_output
        
        out = self.out(pooled_output)

        return out 

    

    def unfreeze_bert_layers(self, num_layers): 
        
        for i in range(12-num_layers, 12): 
            for param in self.bert.encoder.layer[i].parameters(): 
                param.requires_grad = True




class DistilRegressionGFinBERT(nn.Module):
    def __init__(self, dropout = 0.1, path= "./output_models"):

        super(DistilRegressionGFinBERT, self).__init__()

        teacher, _ ,_ = load_trained_model(path, RegressionGFinBERT)

        self.mini_gfinbert = copy.deepcopy(teacher)
        self.mini_gfinbert.bert.encoder.layer = nn.ModuleList(self.mini_gfinbert.bert. encoder.layer[::2])


    def forward(self, input_ids, attention_mask):

        out = self.mini_gfinbert(input_ids=input_ids, attention_mask=attention_mask)
        return out 

    
