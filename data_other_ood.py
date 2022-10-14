
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

from create_spans import create_span_masks


def add_sick(
        description: str, 
        s1_filename: str, 
        s2_filename: str, 
        labels_filename: str,
        params: argparse.Namespace, 
        tokenizer: BertTokenizerFast,
        reduce_data=False,
        reduce_number=1000) -> dict:
    """
    We load and tokenize the SICK dataset
    """
 
    batch_size = params.batch_size
        
    output_data_dict = {}

    with open(s1_filename) as f:
        s1_data = f.readlines()

    with open(s2_filename) as f:
        s2_data = f.readlines()

    with open(labels_filename) as f:
        labels_data = f.readlines()

    
    sick_dataframe = pd.DataFrame(
            {'premise': s1_data, 'hypothesis': s2_data, 'labels': labels_data})

    sick_dataframe['label'] = -1

    sick_dataframe['label'].loc[
            sick_dataframe['labels'] == 'entailment\n'] = 0

    sick_dataframe['label'].loc[
            sick_dataframe['labels'] == 'neutral\n'] = 1

    sick_dataframe['label'].loc[
            sick_dataframe['labels'] == 'contradiction\n'] = 2

    if reduce_data:
        dataset_len = min(len(sick_dataframe), reduce_number)
        sick_dataframe = sick_dataframe[0:dataset_len]

    sick_dataset = Dataset.from_pandas(sick_dataframe)

    sick_dataset = sick_dataset.filter(
            lambda example: example['label'] in [0, 1, 2])

    if params.span_groups != 'none':
        spans_dictionary = create_span_masks(
                params, 
                sick_dataset, 
                tokenizer, 
                params.span_groups)
    else:
        spans_dictionary = None

    sick_dataset = sick_dataset.map(
            lambda x: tokenizer(
                x['premise'], 
                x['hypothesis'],
                truncation=False,
                padding=True), 
            batched=True, batch_size=batch_size)

    sick_dataset.set_format(
            type='torch', columns=[
                'input_ids', 'token_type_ids', 'attention_mask', 'label'])

    dataloader_sick = torch.utils.data.DataLoader(
            sick_dataset, batch_size=batch_size)

    output_data_dict.update({description: (
        dataloader_sick, 
        spans_dictionary, 
        None)})

    return output_data_dict


def add_snli_hard(
    description: str, 
    filename: str, 
    params: argparse.Namespace, 
    tokenizer: BertTokenizerFast) -> dict:

    """
    We load and tokenize the SNLI-Hard dataset
    """

    output_data_dict = {}

    batch_size = params.batch_size

    snli_hard_df = pd.DataFrame(columns=['premise', 'hypothesis', 'labels'])
    
    with open(filename) as f:
        data = list(f)

    for json_str in data:
        result = json.loads(json_str)
        snli_hard_df = snli_hard_df.append(
                {'premise': result['sentence1'], 
                    'hypothesis': result['sentence2'], 
                    'labels': result['gold_label']}, 
                ignore_index=True)

    snli_hard_df['label'] = -1
    snli_hard_df['label'].loc[snli_hard_df['labels'] == 'entailment'] = 0
    snli_hard_df['label'].loc[snli_hard_df['labels'] == 'neutral'] = 1
    snli_hard_df['label'].loc[snli_hard_df['labels'] == 'contradiction'] = 2

    snli_hard = Dataset.from_pandas(snli_hard_df)

    snli_hard = snli_hard.filter(lambda example: example['label'] in [0, 1, 2])

    if params.span_groups != 'none':
        spans_dictionary = create_span_masks(
                params, 
                snli_hard, 
                tokenizer, 
                params.span_groups)
    else:
        spans_dictionary = None

    snli_hard = snli_hard.map(
            lambda x: tokenizer(
                x['premise'], 
                x['hypothesis'],
                truncation=False,
                padding=True), 
            batched=True, batch_size=batch_size)

    snli_hard.set_format(
            type='torch', 
            columns=['input_ids', 'token_type_ids', 
                'attention_mask', 'label'])

    dataloader_snli_hard = torch.utils.data.DataLoader(
            snli_hard, 
            batch_size=batch_size)

    output_data_dict.update(
            {description: (dataloader_snli_hard, spans_dictionary, None)})

    return output_data_dict


def add_hans(
        description: str, 
        filename: str, 
        params: argparse.Namespace, 
        tokenizer: BertTokenizerFast,
        heuristic_list: list) -> dict:
    """
    We load and tokenize the SNLI-Hard dataset
    """

    output_data_dict = {}

    batch_size = params.batch_size
    
    fi = open(filename, "r")

    correct_dict = {}
    first = True

    for line in fi:
        if first:
            labels = line.strip().split("\t")
            idIndex = labels.index("pairID")
            first = False
            continue
        else:
            parts = line.strip().split("\t")
            this_line_dict = {}
            for index, label in enumerate(labels):
                if label == "pairID":
                    continue
                else:
                    this_line_dict[label] = parts[index]
            correct_dict[parts[idIndex]] = this_line_dict

    hans_df = pd.DataFrame.from_dict(correct_dict, orient='index')

    hans_df = hans_df[hans_df['heuristic'].isin(heuristic_list)]

    hans_df = hans_df[['sentence1', 'sentence2', 'gold_label']]
    
    hans_df = hans_df.rename(
            columns={
                'sentence1': 'premise', 
                'sentence2': 'hypothesis', 
                'gold_label': 'label'})


    hans_df['label'].loc[hans_df['label'] == 'entailment'] = 0
    hans_df['label'].loc[hans_df['label'] == 'non-entailment'] = 1

    hans = Dataset.from_pandas(hans_df)

    hans = hans.filter(lambda example: example['label'] in [0, 1])

    if params.span_groups != 'none':
        spans_dictionary = create_span_masks(
                params, 
                hans, 
                tokenizer, 
                params.span_groups)
    else:
        spans_dictionary = None

    hans = hans.map(
            lambda x: tokenizer(
                x['premise'], 
                x['hypothesis'],
                truncation=False,
                padding=True), 
            batched=True, 
            batch_size=batch_size)

    hans.set_format(
            type='torch', columns=['input_ids', 'token_type_ids', 
                'attention_mask', 'label'])

    dataloader_hans = torch.utils.data.DataLoader(hans, batch_size=batch_size)

    output_data_dict.update({description: (
        dataloader_hans, 
        spans_dictionary, 
        None)})

    return output_data_dict



