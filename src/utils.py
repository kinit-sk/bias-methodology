import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def model_init(model_name):
    model, tokenizer = AutoModelForMaskedLM.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.to('cuda:0')
    return model, tokenizer

