from collections import Counter
import csv
from functools import partial
from itertools import count
import json
import os
import re
from typing import List, Tuple

from tokenization import *


def print_false(func):
    """
    Decorator that prints samples that fail the filter function `func`
    """

    def wrapper(sam):
        result = func(sam)
        if not result:
            print(func.__name__)
            for sen in sam:
                print(sen)
            print()
        return result

    return wrapper


@print_false
def filter_length(sam):
    """
    Do all sentences have the same length?
    """
    sen_len = lambda sen: len(sen.split())
    return all(
        sen_len(sam[0]) == sen_len(sen)
        for sen in sam
    )


@print_false
def filter_one_token(sam):
    """
    Do stereotype-antistereotype pairs differ in exactly one token? Check both
    normal and gender-swapped version if present.
    """
    dif = lambda a, b: sum(w1 != w2 for w1, w2 in zip(a.split(), b.split()))
    if len(sam) == 2:
        return dif(sam[0], sam[1]) == 1
    elif len(sam) == 4:
        return dif(sam[0], sam[1]) == dif(sam[2], sam[3]) == 1


@print_false
def filter_brackets(sam):
    """
    Do all sentences have exactly one <masked> word with brackets?
    """
    return all(
        sen.count('>') == sen.count('<') == 1 and re.search(r'<\S+>', sen)
        for sen in sam
    )


@print_false
def filter_no_brackets(sam):
    """
    Are there any <brackets>? (For StereoSet dataset)
    """
    return all(
        sen.count('>') + sen.count('<') == 0
        for sen in sam
    )


@print_false
def filter_interpunction(sam):
    """
    Do all sentences end with an interpunction sigsn?
    """
    return all(sen[-1] in {'.', '!', '?'} for sen in sam)


def filter_tokenization(sam, tokenizer):
    '''
    Filters out the samples where the length of masked words do not match in the samples
    assert len(keyword1) == len(keyword3) and len(keyword2) == len(keyword4)
    '''
    kwl = lambda i: kw_len(sam[i], tokenizer)
    result = kwl(0) - kwl(1) == kwl(2) - kwl(3)
    if not result:
        print('filter_tokenization')
        for sen in sam:
            for token in tokenize(sen, tokenizer, only_ids=True):
                print(tokenizer.decode([token]), end=', ')
            print()
        print()
    return result


def create_dataset(func):
    '''
    Decorator for other `create_dataset` functions. Remove duplicates and count sampples.
    '''

    def wrapper(*args, **kwargs):
        dt = func(*args, **kwargs)
        dt = list(dt)
        for sam, count in Counter(dt).items():
            if count > 1:
                print(f'Duplicate {count}x:', sam)
        print('# Samples:', len(dt), ', # Unique:', len(set(dt)))
        return dt

    return wrapper


@create_dataset
def create_our_dataset(tokenizer) -> List[Tuple[str, str, str, str]]:
    with open(os.path.join('..', 'data', 'dataset.v2.csv')) as csvfile:
        reader = csv.DictReader(csvfile)
        dt = [
            tuple(row[col] for col in ('veta1', 'veta2', 'veta3', 'veta4'))
            for row in reader
            if not row['competence?'] and not row['misogynistic?']
        ]
    dt = filter(filter_length, dt)
    dt = filter(filter_one_token, dt)
    dt = filter(filter_brackets, dt)
    dt = filter(filter_interpunction, dt)
    dt = filter(partial(filter_tokenization, tokenizer=tokenizer), dt)
    return dt


@create_dataset
def create_crows_dataset(tokenizer, bias_type='gender') -> List[Tuple[str, str]]:
    with open(os.path.join('..', 'data', 'crows.csv')) as csvfile:
        reader = csv.DictReader(csvfile)
        dt = [
            (row['sent_more'], None, row['sent_less'], None)
            for row in reader
            if row['bias_type'] == bias_type
        ]
    return dt


@create_dataset
def create_crows_revised_dataset(tokenizer, bias_type='gender') -> List[Tuple[str, str]]:
    with open(os.path.join('..', 'data', 'crows_pairs_our_revised.csv')) as csvfile:
        reader = csv.DictReader(csvfile)
        dt = [
            (row['sent_more'], None, row['sent_less'], None)
            for row in reader
            if row['bias_type'] == bias_type
        ]
    return dt


@create_dataset
def create_stereoset_dataset(tokenizer, bias_type='gender') -> List[Tuple[str, str]]:
    def add_brackets(sample):
        """
        Add <brackets> to the diff word between s0 and s1
        """
        s0, s1 = sample[0].split(), sample[1].split()
        df = next(i for i in count() if s0[i] != s1[i])
        for s in (s0, s1):
            s[df] = '<' + s[df] + '>'
            for punct in {'.', '!', '?', ',', '\"'}:
                s[df] = s[df].replace(punct + '>', '>' + punct)
        return ' '.join(s0), ' '.join(s1)

    path = os.path.join('..', 'data', 'stereoset.json')
    dt = json.load(open(path))['data']['intrasentence']
    dt = (
        sam['sentences']
        for sam in dt
        if sam['bias_type'] == bias_type
    )
    dt = (
        (
            next(sen['sentence'] for sen in sam if sen['gold_label'] == 'stereotype'),
            next(sen['sentence'] for sen in sam if sen['gold_label'] == 'anti-stereotype'),
        )
        for sam in dt
    )
    dt = filter(filter_no_brackets, dt)
    dt = filter(filter_length, dt)
    dt = filter(filter_one_token, dt)
    dt = map(add_brackets, dt)
    return dt


@create_dataset
def create_stereoset_genderswap_dataset(tokenizer) -> List[Tuple[str, str, str, str]]:
    path = os.path.join('..', 'data', 'stereoset_genderswap.txt')
    lines = list(filter(None, open(path).read().splitlines()))  # Remove empty lines
    dt = zip(*(lines[i::4] for i in [0, 2, 1, 3]))  # Correct order
    dt = filter(partial(filter_tokenization, tokenizer=tokenizer), dt)
    return dt


def get_dataset_by_name(dataset_name, tokenizer) -> List[Tuple]:
    return {
        'our': create_our_dataset,
        'stereoset': create_stereoset_dataset,
        'stereoset-genderswap': create_stereoset_genderswap_dataset,
        'crows': create_crows_dataset,
        'crows-revised': create_crows_revised_dataset,
    }[dataset_name](tokenizer=tokenizer)
