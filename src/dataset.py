from collections import Counter
import csv
from functools import partial
from itertools import count
import json
import os
import random
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


@create_dataset
def create_stereoset_genderswap_filtered_dataset(tokenizer) -> List[Tuple[str, str, str, str]]:
    path = os.path.join('..', 'data', 'stereoset_genderswap_filtered.txt')
    lines = list(filter(None, open(path).read().splitlines()))  # Remove empty lines
    dt = zip(*(lines[i::4] for i in [0, 2, 1, 3]))  # Correct order
    dt = filter(partial(filter_tokenization, tokenizer=tokenizer), dt)
    return dt


def create_stereoset_race_control(tokenizer) -> List[Tuple[str, str, str, str]]:
    countries = ['Ethiopia', 'Europe', 'Ukraine', 'Sudan', 'Afghanistan', 'Iraq', 'Yemen', 'Russia', 'Italy', 'Iran',
             'Venezuela', 'Oman', 'Somalia', 'Syria', 'China', 'India', 'Britain', 'France', 'Greece', 'Scotland',
             'Brazil', 'Africa', 'Jordan', 'Indonesia', 'Vietnam', 'Pakistan', 'Romania', 'Mexico', 'Puerto Rico',
             'Kenya', 'Liberia', 'Sierra Leon', 'Saudi Arabia', 'Lebanon', 'South Africa', 'Korea', 'Singapore',
             'Germany', 'Ireland', 'Ecuador', 'Morocco', 'Qatar', 'Turkey', 'Laos', 'Bangladesh', 'Guatemala', 'Ghana',
             'Cameroon', 'Nepal', 'Albania', 'Spain', 'Paraguay', 'Peru', 'Poland', 'Eriteria', 'Egypt', 'Finland',
             'Australia', 'Taiwan', 'Argentina', 'Chile', 'Netherlands', 'Sweden', 'Crimea', 'Japan', 'Norway',
             'Cape Verde', 'Portugal', 'Austria', 'Columbia', 'Bolivia']
    adjectives = ['Nigerian', 'European', 'Russian', 'Ukrainian', 'Somali', 'Afghan', 'Indian', 'Italian', 'Australian',
                  'Spanish', 'Guatemalan', 'Hispanic', 'Saudi Arabian', 'Finnish', 'Swedish', 'Venezuelan', 'Puerto Rican',
                  'Ghanaian', 'Moroccan', 'Sudanese', 'Chinese', 'Pakistani', 'German', 'Mexican', 'Paraguayan', 'African',
                  'Eritrean', 'Sierra Leonean', 'Irish', 'Brazilian', 'Ecuadorian', 'Kenyan', 'Liberian', 'Cameroonian',
                  'Qatari', 'Syrian', 'Arab', 'Indonesian', 'French', 'Norweigan', 'Jordanian', 'Romanian', 'Crimean',
                  'Native American', 'Omani', 'Iranian', 'Iraqi', 'British', 'Polish', 'Greek', 'Scottish', 'Bolivian',
                  'Vietnamese', 'Nepali', 'Japanese', 'Taiwanese', 'Bengali', 'Albanian', 'Columbian', 'Peruvian',
                  'Argentian', 'Ethiopian', 'Egyptian', 'Portuguese', 'Cape Verdean', 'Turkish', 'Yemeni', 'Austrian',
                  'South African', 'Korean', 'Chilean', 'Laotian', 'Lebanese', 'Singaporean', 'Persian']
    
    dt = create_stereoset_dataset(tokenizer, 'race')

    # {sample -> word from the list above that describes the group}
    subjects = {
        sam: max(  # max because there might be multiple groups, e.g. african and south african - we want the longest possible
            [
                term
                for term in countries + adjectives
                if term.lower() in sam[0].lower() and term.lower() in sam[1].lower()
            ],
            key=len,
            default=None,
        )
        for sam in dt
    }

    # sample -> positions of the subject word in both sentences
    positions = {
        sam: (
            sam[0].lower().index(subject.lower()),
            sam[1].lower().index(subject.lower()),
        )
        for sam, subject in subjects.items()
        if subject
    }


    def alter(sen, old_subject, new_subject, pos):
        return sen[:pos] + new_subject + sen[pos + len(old_subject):]
    
    new_dt = []
    
    for sam in dt:
        if subjects[sam] in countries:
            lst = countries
        elif subjects[sam] in adjectives:
            lst = adjectives
        else:
            raise ValueError('Could not detect keyword')
        
        random.seed(sam[0])
        for new_subject in [random.choice(lst) for _ in range(10)]:
            new_dt.append((
                sam[0],
                sam[1],
                alter(sam[0], subjects[sam], new_subject, positions[sam][0]),
                alter(sam[1], subjects[sam], new_subject, positions[sam][1]),
            ))
            
    return new_dt


def create_stereoset_profession_control(tokenizer) -> List[Tuple[str, str, str, str]]:
    professions = ['barber', 'coach', 'businessperson', 'football player', 'construction worker', 'manager', 'CEO',
               'accountant', 'commander', 'firefighter', 'mover', 'software developer', 'guard', 'baker', 'doctor',
               'athlete', 'artist', 'dancer', 'mathematician', 'janitor', 'carpenter', 'mechanic', 'actor', 'handyman',
               'musician', 'detective', 'politician', 'entrepreneur', 'model', 'opera singer', 'chief', 'lawyer',
               'farmer', 'writer', 'librarian', 'army', 'real-estate developer', 'broker', 'scientist', 'butcher',
               'electrician', 'prosecutor', 'banker', 'cook', 'hairdresser', 'prisoner', 'plumber', 'attourney',
               'boxer', 'chess player', 'priest', 'swimmer', 'tennis player', 'supervisor', 'attendant', 'housekeeper',
               'maid', 'producer', 'researcher', 'midwife', 'judge', 'umpire', 'bartender', 'economist', 'physicist',
               'psychologist', 'theologian', 'salesperson', 'physician', 'sheriff', 'cashier', 'assistant',
               'receptionist', 'editor', 'engineer', 'comedian', 'painter', 'civil servant', 'diplomat', 'guitarist',
               'linguist', 'poet', 'laborer', 'teacher', 'delivery man', 'realtor', 'pilot', 'professor ', 'chemist',
               'historian', 'pensioner', 'performing artist', 'singer', 'secretary', 'auditor ', 'counselor',
               'designer', 'soldier', 'journalist', 'dentist', 'analyst', 'nurse', 'tailor', 'waiter', 'author',
               'architect', 'academic', 'director', 'illustrator', 'clerk', 'policeman', 'chef', 'photographer',
               'drawer', 'cleaner', 'pharmacist', 'pianist', 'composer', 'handball player', 'sociologist']
    
    dt = create_stereoset_dataset(tokenizer, 'profession')

    # {sample -> word from the list above that describes the group}
    subjects = {
        sam: max(  # max because there might be multiple groups, e.g. african and south african - we want the longest possible
            [
                term
                for term in professions
                if term.lower() in sam[0].lower() and term.lower() in sam[1].lower()
            ],
            key=len,
            default=None,
        )
        for sam in dt
    }

    # sample -> positions of the subject word in both sentences
    positions = {
        sam: (
            sam[0].lower().index(subject.lower()),
            sam[1].lower().index(subject.lower()),
        )
        for sam, subject in subjects.items()
        if subject
    }


    def alter(sen, old_subject, new_subject, pos):
        return sen[:pos] + new_subject + sen[pos + len(old_subject):]
    
    new_dt = []
    
    for sam in dt:
        random.seed(sam[0])
        for new_subject in [random.choice(professions) for _ in range(10)]:
            new_dt.append((
                sam[0],
                sam[1],
                alter(sam[0], subjects[sam], new_subject, positions[sam][0]),
                alter(sam[1], subjects[sam], new_subject, positions[sam][1]),
            ))
            
    return new_dt


@create_dataset
def create_crows_negation_dataset(tokenizer):
    with open(os.path.join('..', 'data', 'crows-neg.csv')) as csvfile:
        cs_samples = list(zip(*csv.reader(csvfile)))

    for cs_sample in cs_samples:
        for i in range(2, len(cs_sample), 2):
            if cs_sample[i]:
                yield (
                    cs_sample[0],
                    cs_sample[1],
                    cs_sample[i],
                    cs_sample[i+1],
                )    

@create_dataset    
def create_crows_antistereotype_dataset(tokenizer):
    with open(os.path.join('..', 'data', 'crows-anti.csv')) as csvfile:
        cs_samples = list(zip(*csv.reader(csvfile)))

    for cs_sample in cs_samples:
        for i in range(2, len(cs_sample), 2):
            if cs_sample[i]:
                yield (
                    cs_sample[0],
                    cs_sample[1],
                    cs_sample[i],
                    cs_sample[i+1],
                )    

    
    
def get_dataset_by_name(dataset_name, tokenizer) -> List[Tuple]:
    return {
        'our': create_our_dataset,
        'stereoset': create_stereoset_dataset,
        'stereoset-genderswap': create_stereoset_genderswap_dataset,
        'stereoset-genderswap-filtered': create_stereoset_genderswap_filtered_dataset,
        'stereoset-race-control': create_stereoset_race_control,
        'stereoset-profession-control': create_stereoset_profession_control,
        'crows': create_crows_dataset,
        'crows-revised': create_crows_revised_dataset,
        'crows-negation': create_crows_negation_dataset,
        'crows-antistereotypes': create_crows_antistereotype_dataset,
    }[dataset_name](tokenizer=tokenizer)
