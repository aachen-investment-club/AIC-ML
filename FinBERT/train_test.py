
import torch
import torch.nn as nn
from data_loader import DataFinbert
from torch.utils.data import DataLoader
from config import load_env, get_float, get_int, get_str
import pandas as pd

load_env()

MAX_LEN = get_int("MAX_LEN", 128)
EPOCHS = get_int("EPOCHS", 20)
CV_EPOCHS = get_int("CV_EPOCHS", 20)
LR = get_float("LR", 2e-5)
SAVE_DIR = get_str("SAVE_DIR", "./regression_gbert")
NUM_GPUS = torch.cuda.device_count()
BATCH_SIZE = get_int("BATCH_SIZE", 16) * max(1, NUM_GPUS)


def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, kd_weight, truth_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    #: recall that  KL divergence con be simplified to cross entropy if the teacher is frozen!

    teacher.to(device)
    student.to(device)

    loss_fn = nn.MSELoss().to(device)

    mae_fn = nn.L1Loss().to(device) 


    for param in teacher.parameters():
        param.requires_grad = False # freeze teacher

    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=10)


    overall_loss= 0
    overall_mae = 0


    teacher.eval()  # Teacher in evaluation mode
    student.train() # Student in train mode

    for epoch in range(epochs):

        running_loss = 0
        running_mae = 0.0

        for step, batch in enumerate(train_loader): 
            inputs, mask, truth = batch
            inputs, mask,  truth = inputs.to(device), mask.to(device), truth.to(device) 

            optimizer.zero_grad()

            #: perform forward pass in teacher
            with torch.no_grad():
                teacher_preds = teacher(inputs, mask)

            #: same for student 
            student_preds = student(inputs, mask)

            distil_loss = loss_fn(student_preds , teacher_preds )

            #: we also consider the loss wrt. ground truth labels 
            label_loss = loss_fn(student_preds, truth)
            mae = mae_fn(student_preds, truth)

            # Weighted sum of the two losses

            loss = kd_weight * distil_loss+ truth_weight* label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_mae += mae.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, MSE: {running_loss / len(train_loader)}")
        print(f"Epoch {epoch+1}/{epochs}, MAE: {running_mae/ len(train_loader)}")

        overall_loss += running_loss / len(train_loader)
        overall_mae += running_mae/ len(train_loader)

    return {
        "MSE": overall_loss/ epochs, 
        "MAE": overall_mae/ epochs
    }




def predict(text, model, tokenizer, device="cpu"):
    model.to(device)
    model.eval()

    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        
        prediction = output.squeeze().item()


    return {
        "text": text,
        "prediction":prediction 
    }








def grid_search_cross_validation(
    model_class,
    params, 
    tokenizer, 
    train_df,  
    test_df, 
    folds, 
    BATCH_SIZE,
    epochs, 
    device, 
    cv_epochs, 
    bert_layers = 12, 
    bert_has_pooler  = True
):

    min_mae = 1e10
    best_config = {}

    metrics = {
        "learning_rate": [], 
        "dropout": [], 
        "unfreeze_freq": [], 
        "norm_clip": [], 
        "mse": [], 
        "mae": []  
    }
    for lr in params["learning_rate"]: 
        for dropout in params["dropout"]:
            for freq in params["unfreeze_freq"]:
                for clip in params["norm_clip"]: 
                    loss, mae = train_cross_validation(
                        model_class = model_class,
                        tokenizer = tokenizer, 
                        train_df = train_df, 
                        folds = folds, 
                        BATCH_SIZE=BATCH_SIZE, 
                        epochs= cv_epochs, 
                        learning_rate = lr, 
                        device =device, 
                        unfreeze_freq= freq, 
                        clip = clip, 
                        dropout= dropout, 
                        bert_layers= bert_layers, 
                        bert_has_pooler= bert_has_pooler
                    )
                    metrics["learning_rate"].append(lr)
                    metrics["dropout"].append(dropout)
                    metrics["unfreeze_freq"].append(freq)
                    metrics["norm_clip"].append(clip)
                    metrics["mse"].append(loss)
                    metrics["mae"].append(mae)
                    if mae < min_mae: 
                        min_mae = mae
                        best_config = {
                            "learning_rate": lr, 
                            "dropout": dropout, 
                            "unfreeze_freq": freq, 
                            "norm_clip": clip
                        }

    train_dataset = DataFinbert(train_df, tokenizer)
    test_dataset = DataFinbert(test_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    final_model = model_class(best_config["dropout"])
    






    metrics = pd.DataFrame(metrics)


    train_mse, train_mae =train(
        final_model, 
        train_loader, 
        epochs, 
        best_config["learning_rate"], 
        device, 
        best_config["unfreeze_freq"], 
        best_config["norm_clip"],
        bert_layers= bert_layers, 
        bert_has_pooler= bert_has_pooler
    )


    test_results = test(
        model=final_model,
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



    final_model.eval()
    torch.save(final_model.state_dict(), f"{SAVE_DIR}/pytorch_model.bin")
    tokenizer.save_pretrained(SAVE_DIR)


    metrics.to_csv(f"{SAVE_DIR}/validation_metrics.csv")
    train_test_metrics.to_csv(f"{SAVE_DIR}/train_test_metrics.csv")


    torch.save({
        "model_state_dict": final_model.state_dict(),
        "epochs": EPOCHS,
        "learning_rate": LR,
        "max_len": MAX_LEN,
        "label_mapping": {0: -1.0, 1: 0.0, 2: 1.0}
    }, f"{SAVE_DIR}/checkpoint.pt")


def train_cross_validation(
    model_class,
    tokenizer, 
    train_df,  
    folds, 
    BATCH_SIZE,
    epochs, 
    learning_rate, 
    device, 
    unfreeze_freq = 2, 
    clip = 1, 
    dropout = 0.1, 
    bert_layers= 12, 
    bert_has_pooler= True
):
    

    fold_size = len(train_df)// folds
    total_loss = 0 
    total_mae = 0 
    evaluation_subset = train_df.iloc[fold_size*(folds-1):] 
    evaluation_subset= DataFinbert(evaluation_subset, tokenizer)
    evaluation_loader = DataLoader(evaluation_subset, batch_size=BATCH_SIZE, shuffle=False)
    for i in range(folds): 
        print(f"started fold {i}")
        local_model = model_class(dropout=dropout)



        subset = train_df.sample(n = len(train_df)-fold_size, replace = True, random_state= 42+i)
        train_subset= DataFinbert(subset, tokenizer)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)


        evaluation_subset_df= train_df.loc[~train_df.index.isin(subset.index)]
        evaluation_subset= DataFinbert(evaluation_subset_df, tokenizer)
        evaluation_loader = DataLoader(evaluation_subset, batch_size=BATCH_SIZE, shuffle=False)


        train(
            local_model, 
            train_loader, 
            epochs, 
            learning_rate, 
            device, 
            unfreeze_freq, 
            clip, 
            bert_layers,
            bert_has_pooler
        )
        results = test(local_model, evaluation_loader, device)
        loss, mae = results["mse"] , results["mae"]


        total_loss+= loss
        total_mae += mae

    avg_loss = total_loss/(folds)
    avg_mae = total_mae/(folds)
    print(
        f"avg MSE (cross validation): {avg_loss}",
        f"avg MAE (cross validation): {avg_mae}"
    )

    return avg_loss, avg_mae



def train(
    model, 
    train_loader, 
    epochs, 
    learning_rate, 
    device, 
    unfreeze_freq = 2, 
    clip = 1, 
    bert_layers = 12, 
    bert_has_pooler = True
):
    print("model type:", type(model))

    print(device)
    
    model.to(device)


    loss_fn = nn.MSELoss().to(device)

    #: MAE is good for reporting error in regression
    mae_fn = nn.L1Loss().to(device) 


    for param in model.parameters():
        param.requires_grad = False # freeze everything at start. 

    #: we perform gradual unfreezing 
    for param in model.out.parameters():  
        param.requires_grad = True #: unfreeze the top layer at the start. 


    if bert_has_pooler:    
        #: impliemetations like distilbert have no pooler. 
        for param in model.bert.pooler.parameters():  
            param.requires_grad = True #: unfreeze the pooler from BERT (this is used for classification/ regression)

    """
    FIRST freeze, then init the optimizer; otherwise the updates are incorrect 
    """


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=10)

    overall_loss= 0
    overall_mae = 0

    for epoch in range(epochs): 
        running_loss = 0
        running_mae = 0.0


        num_layers = min(epoch // unfreeze_freq, bert_layers) # controls how many epochs are needed to unfreeze a layer.
        model.unfreeze_bert_layers(num_layers)

        for step, batch in enumerate(train_loader): 
            inputs, mask, labels = batch
            inputs, mask,  labels = inputs.to(device), mask.to(device), labels.to(device) 

            optimizer.zero_grad()
            

            out = model(input_ids=inputs, attention_mask=mask)
            out = out.squeeze(-1)

            loss = loss_fn(out, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            running_loss+= loss.item()
            overall_loss+= loss.item()
            #: to avoid exploding gradients. 
            
            
            optimizer.step()
            with torch.no_grad(): 
                # only for reporting
                running_mae += mae_fn(out, labels).item()
                overall_mae+= mae_fn(out, labels).item()
        
        scheduler.step()


        avg_loss = running_loss / len(train_loader)
        avg_mae = running_mae / len(train_loader)

        print (
            f"epoch: {epoch} |", 
            f"mse: {avg_loss} | ", 
            f"mae: {avg_mae} | " 
        )




    return overall_loss/epochs, overall_mae/epochs
    





def test(model, test_loader, device):

    model.to(device)
    model.eval()

    mse_fn = nn.MSELoss().to(device)
    mae_fn = nn.L1Loss().to(device)

    total_mse = 0.0
    total_mae = 0.0

    with torch.no_grad():
        for inputs, mask, labels in test_loader:
            inputs = inputs.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=inputs, attention_mask=mask)
            outputs = outputs.squeeze(-1)

            total_mse += mse_fn(outputs, labels).item()
            total_mae += mae_fn(outputs, labels).item()


    avg_mse = total_mse / len(test_loader)
    avg_mae = total_mae / len(test_loader)

    print(f"Test Results ; MSE: {avg_mse:.4f} ; MAE: {avg_mae:.4f}")

    return {
        "mse": avg_mse,
        "mae": avg_mae,
    }
