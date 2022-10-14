
import os
import pickle
import re
import csv
import json
import numpy as np
import pandas as pd
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import torch
import nltk
import argparse
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

SPECIAL_CHARS = [".", ",", "!", "?", ":", ";", ")", "(", 
        ">", "<", "[", "]", '"', "'"]


def find_spans(
        params: argparse.Namespace,
        dataset: Dataset,
        tokenizer: BertTokenizerFast,
        is_train: bool) -> dict:
    """
    Create a dictionary of span masks to represent each observaitons e-SNLI span
    """

    span_all = {}

    if is_train:
        expl_col_list = ['Sentence2_marked_1']
    else:
        expl_col_list = [
                'Sentence2_marked_1', 
                'Sentence2_marked_2', 
                'Sentence2_marked_3']

    for i in range(len(dataset)):

        obs_dict = {}

        for expl_col in expl_col_list: 
            start_locations_h, end_locations_h, token_len_h, obs_tokens \
                    = _find_individual_span(
                            params, 
                            expl_col, 
                            dataset[i], 
                            tokenizer, i)

            span_list=[0]*token_len_h
            span_list = [0 if idx < start_locations_h or idx > end_locations_h \
                    else 1 for idx in range(len(span_list))]
            obs_dict[expl_col] = span_list

        span_all.update({str(obs_tokens): obs_dict})
    return span_all


def _find_individual_span(
        params: argparse.Namespace,
        expl_col: str,
        data_obs: dict,
        tokenizer: BertTokenizerFast,
        i: int) -> (int, int, int, list):
    """
    Find the start and end points (no. tokens) for each eSNLI span
    """

    sentence_to_tokenize_hyp =  data_obs['hypothesis']

    exp_with_stars_hyp =  data_obs[expl_col]

    if exp_with_stars_hyp == None:
        exp_with_stars_hyp = ''

    start_locations_h, end_locations_h, token_len_h = _find_start_end(
            params,
            tokenizer,
            sentence_to_tokenize_hyp, 
            exp_with_stars_hyp)
    
    tokenized_data_both = tokenizer([(data_obs['premise'], data_obs['hypothesis'])],
            add_special_tokens=True)


    return start_locations_h, end_locations_h, token_len_h, tokenized_data_both['input_ids'][0]


def _find_start_end(
        params: argparse.Namespace,
        tokenizer: BertTokenizerFast, 
        sentence_to_tokenize: str, 
        explanation_with_stars: str) -> (int, int, int):
    """
    We calculate the token start and end for the highlighted words
    """

    supervise_example = True

    tokens = tokenizer.tokenize(sentence_to_tokenize)

    star_locations, explanation_reduced = _construct_from_annotated(
            tokenizer,
            sentence_to_tokenize, 
            explanation_with_stars)

    word_indices = _find_word_indices_annotated(
            explanation_reduced, 
            star_locations)

    char_len = 0
    token_lens = [0]

    weights = torch.zeros(len(tokens))

    for i, j in enumerate(tokens):
        if params.model_type[0:4] == 'bert':
            char_len = char_len + len(j.replace("##", ""))
        if params.model_type[0:9] == 'microsoft':
            char_len = char_len + len(j.replace("Ġ", ""))
        token_lens.append(char_len)

    token_lens = token_lens[:-1]

    assert char_len  == len(sentence_to_tokenize.replace(" ", "")),\
            "Explanation length should not change"

    # Knowing the star positions
    if supervise_example == True:
        for i, j in enumerate(word_indices):
            start, end = j
            for index, cum_length in enumerate(token_lens):
                if cum_length >= start and cum_length <= end:
                    assert index < len(weights)
                    weights[index] = 1

    weights = [1 if x >= 1 else 0 for x in weights]

    # We only want explanations where we have a single span
    if 1 in weights:
        first_occurance = weights.index(1)
        last_occurance = len(weights) - weights[::-1].index(1) - 1
    else:
        supervise_example = False
        first_occurance = -1
        last_occurance = -1

    for i, j in enumerate(weights):
        if i >= first_occurance and i <= last_occurance:
            if j == 0:
                if tokens[i] not in SPECIAL_CHARS:
                    supervise_example = False

    if supervise_example == False:
        first_occurance = -1
        last_occurance = -1

    return first_occurance, last_occurance, len(weights)


def _construct_from_annotated(
        tokenizer: BertTokenizerFast, 
        sentence: str, 
        explanation_with_stars: str) -> (list, str):
    """
    Finds the position of stars in the annotations, and returns the
        ... rationales without these
    """

    original_encoded = tokenizer.encode(sentence, add_special_tokens=False)

    explanation_with_stars = explanation_with_stars + " "
    # We want to take punctuation outside of the * *'s
    for char in SPECIAL_CHARS:
        explanation_with_stars \
                = explanation_with_stars.replace(" *" + char, " " + char + "*")
        explanation_with_stars \
                = explanation_with_stars.replace(char + "* ", "*" + char + " ")
        explanation_with_stars \
                = explanation_with_stars.replace(char + "*<", "*" + char + "<")
        explanation_with_stars \
                = explanation_with_stars.replace(">*" + char, ">" + char + "*")
        explanation_with_stars \
                = explanation_with_stars.replace(char + "*[", "*" + char + "[")
        explanation_with_stars \
                = explanation_with_stars.replace("]*" + char, "]" + char + "*")

    # As BERT tokens do not include a symbol for spaces, we removes spaces
    explanation_reduced = explanation_with_stars.replace(" ", "")

    # Find where the star characters are:
    star_locations = [m.start() for m in re.finditer('\*', explanation_reduced)]

    return star_locations, explanation_reduced


def _find_word_indices_annotated(explanation: str, star_locations: list) -> list:

    """
    We return the character number for words that are highlighted
    """

    # We now loop through all words mentioned:
    word_indexes = []
    for i in range(int(len(star_locations)/2)):

        a = 2*i  #Starting star
        b = 2*i + 1  #Ending star

        start = star_locations[a]
        end = star_locations[b]

        word_length = end - start - 1
        explanation_reduced = explanation[:end+1]
        explanation_reduced = explanation_reduced.replace("*", "")

        # We find the word start and the word end character numbers
        word_start = len(explanation_reduced) - word_length
        word_end = len(explanation_reduced) - 1

        word_indexes.append((word_start, word_end))

    return word_indexes


