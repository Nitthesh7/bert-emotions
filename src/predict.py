import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import config, model

def predict_run(sentence, threshold=0.5):
    
    # load the model (cpu here)
    device = torch.device(config.DEVICE)
    bert_model = model.BERTMultiLabel()
    bert_model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    
    # evaluation mode
    bert_model.eval()
    
    tokenized_sentence = config.TOKENIZER(
        sentence, 
        truncation=True,
        padding=True,
        return_tensors='pt',
        max_length=config.MAX_LEN
    )
    
    tokenized_sentence.to(device)
    
    with torch.no_grad():
        outputs = bert_model(
            ids = tokenized_sentence['input_ids'],
            mask = tokenized_sentence['attention_mask'],
            token_type_ids = tokenized_sentence['token_type_ids'],
        )
    
    # logits to probs
    outputs = torch.sigmoid(outputs)
    # custom threshold (Default=0.5)
    binary_predictions = (outputs>=threshold).int().tolist()
        
    return outputs.numpy()[0], binary_predictions[0]

if __name__ == '__main__':
    ### for terminal run
    sentence = input('Type a sentence: ')
    prob, binary = predict_run(sentence)

    emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval',
       'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
       'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
       'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride',
       'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
    
    predicted_emotions = [emotion for i, emotion in enumerate(emotions) if binary[i] == 1]
    predicted_emotions

    print(predicted_emotions)

    prob_emotions = []
    for i in range(len(emotions)):
        prob_emotions.append([emotions[i], prob[i]])
    
    print(sorted(prob_emotions, key=lambda x: x[1], reverse=True))

    