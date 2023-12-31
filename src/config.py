import transformers

DEVICE = "cpu"  # CUDA while training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 12
VALID_BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-5
MODEL_PATH = "../models/bert-multi-label.bin"
DATA_PATH = "../data/cleaned_data.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    'bert-base-uncased', 
    do_lower_case=True)