from transformers import AutoModelForMaskedLM, AutoTokenizer


def model_init(model_name):
    return AutoModelForMaskedLM.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)

