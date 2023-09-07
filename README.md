# Bert Emotions - Multi-label classification

[GoEmotions](https://paperswithcode.com/dataset/goemotions) (full dataset) was used for this project. After some preprocessing & criterias, only 53,951 of the original 58,011 instances are used to reduce noise. 

The GPU T4 x2 was used to train the BERT model, which took about 3.5 hours. To fine-tune the model for our objective, a dropout layer and an output layer are added to the original BERT design.

After 3 epochs with the majority of the hyperparameters left at default   
(Check the [config](https://github.com/Nitthesh7/bert-emotions/blob/main/src/config.py) file for altered hyperparameters):

Macro-Precision: 0.07142857142857142   
Macro-Recall: 0.05952380952380952  
Macro-F1-score = 0.0642857142857143  
Hamming Loss = 0.026785714285714284

## To run the app locally 

```bash
git clone https://github.com/Nitthesh7/bert-emotions.git

pip install -r requirements.txt

cd src

streamlit run app.py

```

## Some examples
Sentence: After facing numerous challenges, I received unexpected support from friends and colleagues, which filled me with a deep sense of appreciation and hope for the future.

![example-1](https://github.com/Nitthesh7/bert-emotions/blob/main/img/example1.png?raw=true)

Sentence: After waiting for hours at the airport, my flight was canceled without any explanation, leaving me feeling frustrated and furious.

![example-2](https://github.com/Nitthesh7/bert-emotions/blob/main/img/example2.png?raw=true)

Sentence: Upon receiving the news of his promotion, he couldn't help but smile from ear to ear, yet deep down, he felt a pang of nostalgia for the simpler times of his previous job.

![example-3](https://github.com/Nitthesh7/bert-emotions/blob/main/img/example3.png?raw=true)