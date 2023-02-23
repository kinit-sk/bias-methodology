from difflib import SequenceMatcher
import re


def tokenize(sen, tokenizer, only_ids=False, **kwargs):
    """
    Use `tokenizer` to parse <bracketed> sentence `sen`.
    
    `only_ids` - Return only token ids if True, `BatchEncoding` otherwise.
    `kwargs` - Are sent to tokenizer.
    """
    sen = sen.replace('<', '').replace('>', '')
    batch_encoding = tokenizer(sen, return_tensors="pt", **kwargs)
    if only_ids:
        return batch_encoding['input_ids'][0].tolist()
    else:
        return batch_encoding


def tokenize_with_mask(sen, tokenizer, only_ids=False):
    '''
    Use `tokenizer` to parse sentence `sen`. Replace keyword with appropriate
    number of `mask_token` tokens.
  
    We need to use `SequenceMatcher` because simply adding <mask> tokens is not
    realiable enough and weird empty tokens are being injected if the mask touches
    interpunction. E.g. XLM-R will tokenize `<mask>,` as: `['<mask>', '', ',']`
    Note the unexpected `''` token in the middle. Instead, we detect the tokens
    that stay the same in the original sentence using `Sequencematcher` and mask
    all the other tokens.
    
    `only_ids` - Return only token ids if True, `BatchEncoding` otherwise.
    '''
    batch_encoding = tokenize(sen, tokenizer)
    matcher = SequenceMatcher(
        None,
        batch_encoding['input_ids'][0].tolist(),
        tokenizer(re.sub('<.*>', tokenizer.mask_token, sen), return_tensors="pt")['input_ids'][0].tolist()
    )
    for (op, s1_start, s1_end, _, _) in matcher.get_opcodes():
        for token_id in range(s1_start, s1_end):
            if op != 'equal':
                batch_encoding['input_ids'][0][token_id] = tokenizer.mask_token_id
    if only_ids:
        return batch_encoding['input_ids'][0].tolist()
    else:
        return batch_encoding


def kw(sen):
    '''
    Return the keyword from the <brackets>.
    '''
    return re.search('<(.*)>', sen).groups()[0]


def kw_len(sen, tokenizer):
    '''
    Number of keyword tokens.
    '''
    tokens = tokenize_with_mask(sen, tokenizer, only_ids=True)
    return tokens.count(tokenizer.mask_token_id)
