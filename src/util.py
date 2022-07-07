from tokenization import tokenize_with_mask

from transformers import AutoModelForMaskedLM, AutoTokenizer


def model_init(model_name):
    return AutoModelForMaskedLM.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)


def fill_mask(sen, model, tokenizer):
    batch_encoding = tokenize_with_mask(sen, tokenizer)
    
    assert sum(id_ == tokenizer.mask_token_id for id_ in batch_encoding['input_ids']) == 1
    
    probs = model(**batch_encoding).logits.softmax(dim=-1)[0]  # [0] because we only have one sample
    print(probs)
    probs_mask = torch.index_select(probs, 0, indices)
    
    # for mask in batch_encoding:
    #     model(predict)
    #     fill first mask
    # print(sen)
    # print(batch_encoding.encode())
    
    
