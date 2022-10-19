
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
import argparse
from create_spans import create_span_masks
from get_esnli_spans import find_spans
from data_other_ood import add_sick
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast


def create_dictionary_of_hf_ood_datasets(
        params: argparse.Namespace, 
        snli=True, 
        mnli=True) -> list:
    """
    We create dictionary with the description of each validation set we want
    """

    eval_data_list = []

    if snli:
        eval_data_list = _append_snli_dict(eval_data_list)

    if mnli:
        eval_data_list = _append_mnli_dict(eval_data_list)

    return eval_data_list


def _append_snli_dict(eval_data_list: list) -> list:
    """
    Appending SNLI dictionary
    """

    for split in ['test', 'validation']:
        eval_data_list.append({'description': 'snli', 'split_name': split})

    return eval_data_list


def _append_mnli_dict(eval_data_list: list) -> list:
    """
    Appending MNLI dictionary
    """
    for split in ['validation_mismatched', 'validation_matched']:
        eval_data_list.append(
                {'description': 'multi_nli', 'split_name': split})

    return eval_data_list


def prepare_all_dataloaders(
        eval_data_list: list,
        params: argparse.Namespace,
        tokenizer: BertTokenizerFast) -> (dict, dict):
    """
    Creates our dataset_config dictionary, with parameters for our datasets
    
    We then create and return the train, dev, test and evaluation dataloaders
    """

    # Set the columns required for our tokenized data
    dataset_config = {}

    dataset_config['data_cols'] = [
            'input_ids', 
            'token_type_ids', 
            'attention_mask', 
            'label']

    dataset_config['esnli_train_file1'] = "dataset_esnli/esnli_train_1.csv"
    dataset_config['esnli_train_file2'] = "dataset_esnli/esnli_train_2.csv"
    dataset_config['dev_file'] = "dataset_esnli/esnli_dev.csv"
    dataset_config['test_file'] = "dataset_esnli/esnli_test.csv"

    esnli_train, esnli_dev, esnli_test = _get_esnli(dataset_config)

    # Creating train, dev and test dataloaders from esnli csvs
    if params.train_data == 'snli':

        train_dataloader_dict = _prepare_esnli_data(
                params,
                tokenizer,
                esnli_train, 
                'esnli_train', 
                is_train=True)

        dev_dataloader_dict = _prepare_esnli_data(
                params,
                tokenizer,
                esnli_dev, 
                'esnli_dev')

        test_dataloader_dict = _prepare_esnli_data(
                params,
                tokenizer,
                esnli_test, 
                'esnli_test')

        esnli_dataloaders = {}
        esnli_dataloaders.update(train_dataloader_dict)
        esnli_dataloaders.update(dev_dataloader_dict)
        esnli_dataloaders.update(test_dataloader_dict)

    elif params.train_data == 'sick':
        sick_folder = '/SICK/uncorrected_SICK/'

        sick_dict = add_sick(
        'sick_train',
        os.getcwd() + "/data" + sick_folder + "s1.train",
        os.getcwd() + "/data" + sick_folder + "s2.train",
        os.getcwd() + "/data" + sick_folder + "labels.train",
        params,
        tokenizer,
        params.reduce,
        params.reduce_number)

        esnli_dataloaders = sick_dict

    eval_dataloaders = {}

    # Creating evaluation dataloaders (from Huggingface)
    for dataset_dict in eval_data_list:
        eval_dataloaders.update(
            {dataset_dict['description'] + "_" + dataset_dict['split_name']: \
                _prepare_huggingface_data(
                    params,
                    tokenizer,
                    dataset_dict)})

    return esnli_dataloaders, eval_dataloaders


def _get_esnli(dataset_config: dict) -> (pd.DataFrame, pd.DataFrame):

    # Loading eSNLI data
    esnli_train = pd.DataFrame()
    dataset1 = pd.read_csv(dataset_config['esnli_train_file1'])
    dataset2 = pd.read_csv(dataset_config['esnli_train_file2'])
    esnli_train = pd.concat([dataset1, dataset2])
    
    esnli_dev = pd.read_csv(dataset_config['dev_file'])
    esnli_test = pd.read_csv(dataset_config['test_file'])
    
    return esnli_train, esnli_dev, esnli_test


def _tokenize_data(
        loaded_data: Dataset, 
        batch_size: int, 
        tokenizer: BertTokenizerFast) -> Dataset:
    """
    We tokenize the data
    """

    loaded_data = loaded_data.map(lambda x: tokenizer(
            x['premise'],
            x['hypothesis'], 
            truncation=False, 
            padding=True),
        batched=True, 
        batch_size=batch_size)

    return loaded_data


def _load_hf_dataset(dataset_dict: dict) -> Dataset:

    # Load dataset
    loaded_data = load_dataset(
            dataset_dict['description'], 
            split=dataset_dict['split_name'])
 
    return loaded_data


def _format_esnli(esnli_df: pd.DataFrame) -> pd.DataFrame:

    # Updating column names
    esnli_df['premise'] = esnli_df['Sentence1']
    esnli_df['hypothesis'] = esnli_df['Sentence2']
    esnli_df['label'] = esnli_df['gold_label']

    # Label needs to be 0, 1 or 2
    esnli_df['label'].loc[esnli_df['label'] == 'entailment'] = 0
    esnli_df['label'].loc[esnli_df['label'] == 'neutral'] = 1
    esnli_df['label'].loc[esnli_df['label'] == 'contradiction'] = 2

    # Both sentences need to be strings
    esnli_df['premise'] = esnli_df['premise'].apply(lambda x: str(x))
    esnli_df['hypothesis'] = esnli_df['hypothesis'].apply(lambda x: str(x))

    return esnli_df


def _reduce_size(
        esnli_df: pd.DataFrame, 
        number_observations: int) -> pd.DataFrame:

    # Loading eSNLI data

    dataset_len = min(len(esnli_df), number_observations)

    esnli_df = esnli_df[0:dataset_len]

    return esnli_df


def _filter_labels(dataset: Dataset) -> Dataset:
    # Remove examples with no gold label
    
    dataset = dataset.filter(
            lambda example: example['label'] in [0, 1, 2])

    return dataset


def _create_dataloader(esnli_dataset: Dataset, batch_size: int) \
        -> torch.utils.data.dataloader.DataLoader:
    
    esnli_dataset.set_format(
            type='torch', 
            columns=[
            'input_ids',
            'token_type_ids',
            'attention_mask',
            'label'])

    dataloader = torch.utils.data.DataLoader(
            esnli_dataset, 
            batch_size=batch_size)

    return dataloader


def _prepare_esnli_data(
        params: argparse.Namespace,
        tokenizer: BertTokenizerFast,
        esnli_df: pd.DataFrame, 
        data_description: str,
        is_train=False) -> dict:
    
    if params.reduce:
        esnli_df = _reduce_size(esnli_df, params.reduce_number)
    
    esnli_df = _format_esnli(esnli_df)
    esnli_dataset = Dataset.from_pandas(esnli_df)
    esnli_dataset = _filter_labels(esnli_dataset)

    if params.span_groups != 'none':
        span_dictionary = create_span_masks(
                params, 
                esnli_dataset, 
                tokenizer, 
                params.span_groups)
    else:
        span_dictionary = None

    esnli_span_list = find_spans(params, esnli_dataset, tokenizer, is_train)
    
    if is_train:
        esnli_dataset = esnli_dataset.shuffle(seed=params.random_seed)
    
    esnli_dataset = _tokenize_data(esnli_dataset, params.batch_size, tokenizer)
    dataloader = _create_dataloader(esnli_dataset, params.batch_size)
    
    return {data_description: (dataloader, span_dictionary, esnli_span_list)}


def _prepare_huggingface_data(
        params: argparse.Namespace,
        tokenizer: BertTokenizerFast,
        dataset_dict: dict) -> (torch.utils.data.dataloader.DataLoader, dict):
    """
    From dictionary for dataset, returns dataloader for huggingface dataset
    """
    huggingface_dataset = _load_hf_dataset(dataset_dict)
    
    if params.span_groups != 'none':
        spans_dictionary = create_span_masks(
                params, 
                huggingface_dataset, 
                tokenizer, 
                params.span_groups)
    else:
        spans_dictionary = None

    huggingface_dataset = _filter_labels(huggingface_dataset)

    huggingface_dataset = _tokenize_data(
            huggingface_dataset, 
            params.batch_size, 
            tokenizer)
    dataloader = _create_dataloader(huggingface_dataset, params.batch_size)

    return dataloader, spans_dictionary, None


def _load_data(name):
    data = []
    a_file = open(name, "r")
    data = json.load(a_file)
    return data


