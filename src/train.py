import torch
import pandas as pd
import torch.nn as nn
import numpy as np

import config, model, dataset, engine

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss

from transformers import AdamW

def train_run():
    df = pd.read_csv(config.DATA_PATH)
    
    # 10% of 53,951 for test set
    df_train, df_valid = train_test_split(df, test_size=0.1, random_state=42)
    
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    
    # train dataset dataloader
    train_dataset = dataset.BERTDataset(
        text=df_train.text.values, 
        targets=df_train.drop(columns={'text'}).values,
    )
    
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE
    )
    
    # test dataset dataloader
    valid_dataset = dataset.BERTDataset(
        text=df_valid.text.values, 
        targets=df_valid.drop(columns={'text'}).values,
    )
    
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE
    )
    
    device = torch.device(config.DEVICE)
    bert_model = model.BERTMultiLabel()
    bert_model.to(device)
    
    optimizer = AdamW(list(bert_model.parameters()), lr= config.LEARNING_RATE)
    
    for epoch in range(config.EPOCHS):
        # training
        engine.train_fn(train_data_loader, bert_model, optimizer, device)
        # evaluation
        outputs, targets = engine.eval_fn(valid_data_loader, bert_model, device)
        # threshold fixed as 0.5 while training
        outputs = (np.array(outputs) >= 0.5).astype(int)
        
        # metrics
        precision_macro_avg = precision_score(targets, outputs, average='macro')
        recall_macro_avg = recall_score(targets, outputs, average='macro')
        f1_macro_avg = f1_score(targets, outputs, average='macro')
        hamming_loss_ = hamming_loss(targets, outputs)

        print(f'Epoch: {epoch+1} | Precision: {precision_macro_avg} | Recall: {recall_macro_avg} | F1-score = {f1_macro_avg} | Hamming Loss: {hamming_loss_}')
    
    # saving the model
    torch.save(bert_model.state_dict(), config.MODEL_PATH)

    return all_metrics

if __name__ == "__main__":
    all_metrics = train_run()