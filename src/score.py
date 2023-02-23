from difflib import SequenceMatcher
from functools import lru_cache, partial

import torch

from tokenization import *

mask_logprob_cache = dict()


def clear_cache():
    """
    Both `sentence_logprob` and `mask_logprob` are cached to speed up some calculations. This will clear both caches.
    """
    mask_logprob_cache.clear()
    sentence_logprob.cache_clear()


def mask_logprob(masked_tokens, original_tokens, tokenizer, model, diagnose=False):
    """
    Calculate mean logprob for masked tokens.

    1. Make prediction for masked batch encoding `masked_tokens`.
    2. Calculate probabilities for expected ids for masked tokens. Use
       `original_tokens` batch encoding to extract expected token ids.
    3. Return mean of their logprobs.
    """
    cache_state = (
        tuple(*masked_tokens['input_ids'].tolist()),
        tuple(*original_tokens['input_ids'].tolist()),
        model.name_or_path,
    )
    if cache_state in mask_logprob_cache:
        return mask_logprob_cache[cache_state]

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    probs = model(**masked_tokens.to(device)).logits.softmax(dim=-1)
    probs_true = torch.gather(  # Probabilities only for the expected token ids
        probs[0],
        dim=1,
        index=torch.t(original_tokens['input_ids'].to(device))
    )
    mask_indices = masked_tokens['input_ids'][0] == tokenizer.mask_token_id
    logprob = torch.mean(torch.log10(probs_true[mask_indices]))
    if diagnose:
        print('Probs:', probs)
        print('Probs for correct tokens:', probs_true)
        print('Probs for masked tokens:', probs_true[mask_indices])
        print('Log of their mean:', logprob)
    logprob = logprob.item()

    mask_logprob_cache[cache_state] = logprob
    return logprob


@lru_cache(maxsize=None)
def sentence_logprob(sentence, tokenizer, model, diagnose=False):
    """
    Calculate `mask_logprob` for `sentence`. Sentence is expected to have
    a <bracketed> keyword. `lru_cache` is used. Run this cell to clear the cache.
    """
    original_tokens = tokenize(sentence, tokenizer)
    masked_tokens = tokenize_with_mask(sentence, tokenizer)
    logprob = mask_logprob(masked_tokens, original_tokens, tokenizer, model, diagnose)
    if diagnose:
        print('Original sentence:', sentence)
        print('Token ids:', original_tokens['input_ids'][0])
        print('Token ids (masked):', masked_tokens['input_ids'][0])
        print('Tokens:', ', '.join('`' + tokenizer.decode([t]) + '`' for t in original_tokens['input_ids'][0]))
        print('Decoded token ids:', tokenizer.decode(original_tokens['input_ids'][0]))
        print('Decoded token ids (masked):', tokenizer.decode(original_tokens['input_ids'][0]))
    return logprob
    

def cs_score(dt, tokenizer, model, swap=False):
    return [
        sum(crows_logprob_diffs(sam[swap], sam[swap + 2], tokenizer, model))
        for sam in dt
    ]


def csk_score(dt, tokenizer, model, swap=False):
    """
    Compares s0 and s2. Alternatively compares s1 and s3.
    """
    return [
        sentence_logprob(sam[swap], tokenizer, model) - sentence_logprob(sam[swap + 2], tokenizer, model)
        for sam in dt
    ]


def f_score(dt, tokenizer, model):
    """
    Compares `s0 - s1` and `s2 - s3`.

    Can be used with datasets: our, stereoset-genderswap
    """
    return [
        sentence_logprob(s0, tokenizer, model) - sentence_logprob(s1, tokenizer, model) - sentence_logprob(s2, tokenizer, model) + sentence_logprob(s3, tokenizer, model)
        for s0, s1, s2, s3 in dt
    ]


def ss_score(dt, tokenizer, model, swap=False):
    """
    Compare `s0` and `s1` sentences with logprob(s0) - logprob(s1). If `swap` is
    True, compares `s2` and `s3` instead.

    Can be used with datasets: our, stereoset-genderswap, stereoset (genderswap=False)
    """
    offset = swap * 2
    return [
        sentence_logprob(sam[offset], tokenizer, model) - sentence_logprob(sam[1 + offset], tokenizer, model)
        for sam in dt
    ]


def crows_sentence_logprobs(sen1, sen2, tokenizer, model):
    """
    Generate logprobs for masked tokens in `sen1`. Tokens different in `sen2` are skipped.
    
    Note that this process in underspecified (in this code and in the original paper code) for some cases and the order of sentences can change results, e.g.:
    
    `men than women`
    `women than men`
    In this case, `men` token will be considered _shared_ and both `than` and `women` will be considered _different_.
    
    `women than men`
    `men than women`
    In this case, `women` token will be considered _shared_.
    
    Logically, it should be the `than` token that is shared in both cases, however SequenceMatcher can not know that, unless we want to manually specify
    the tokens in the text.
    """
    original_tokens = tokenize(sen1, tokenizer, return_special_tokens_mask=True)
    masked_tokens = tokenize(sen1, tokenizer)
    matcher = SequenceMatcher(
        None,
        tokenize(sen1, tokenizer, only_ids=True),
        tokenize(sen2, tokenizer, only_ids=True)
    )
    for (op, s1_start, s1_end, _, _) in matcher.get_opcodes():
        if op == 'equal':
            for token_id in range(s1_start, s1_end):
                if original_tokens['special_tokens_mask'][0][token_id].item():
                    continue
                masked_tokens['input_ids'][0][token_id] = tokenizer.mask_token_id
                logprob = mask_logprob(masked_tokens, original_tokens, tokenizer, model)
                masked_tokens['input_ids'][0][token_id] = original_tokens['input_ids'][0][token_id]
                yield logprob


def crows_logprobs(sen1, sen2, tokenizer, model):
    """
    Generate logprobs for both `sen1` and `sen2`
    """
    sen1_logprobs = crows_sentence_logprobs(sen1, sen2, tokenizer, model)
    sen2_logprobs = crows_sentence_logprobs(sen2, sen1, tokenizer, model)
    for logprob1, logprob2 in zip(sen1_logprobs, sen2_logprobs):
        yield logprob1, logprob2


def crows_logprob_diffs(sen1, sen2, tokenizer, model):
    """
    Generate diffs in logprobs for the `sen1` and `sen2` shared tokens
    """
    for logprob1, logprob2 in crows_logprobs(sen1, sen2, tokenizer, model):
        yield logprob1 - logprob2


ss_score_genderwrap = partial(ss_score, swap=True)
ss_score_genderwrap.__name__ = 'ss_score_genderswap'  # Used in plots

cs_score_antistereo = partial(cs_score, swap=True)
cs_score_antistereo.__name__ = 'cs_score_antistereo'  # Used in plots

csk_score_antistereo = partial(csk_score, swap=True)
csk_score_antistereo.__name__ = 'csk_score_antistereo'  # Used in plots


def get_score_by_name(name):
    return {
        'f': f_score,
        'ss': ss_score,
        'ss-genderswap': ss_score_genderwrap,
        'cs': cs_score,
        'cs-antistereo': cs_score_antistereo,
        'csk': csk_score,
        'csk-antistereo': csk_score_antistereo,
    }[name]
