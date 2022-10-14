
import os
import numpy as np
import transformers
from transformers import AutoTokenizer
import torch
import spacy
import argparse
from datasets import Dataset
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast


def create_span_masks(
        params: argparse.Namespace,
        dataset: Dataset, 
        tokenizer: BertTokenizerFast, 
        consecutive_segments_list: list) -> dict:
    """
    We create span masks based on the presence of spacy noun chunks

    Args: 
        params: Training parameters
        dataset: The dataset we are finding the span segments masks for
        tokenizer: The tokenizer being used
        consecutive_segments_list: A list of how many consecutive segments
            ... we can make spans from (e.g. 1 only, 2 for consecutive).

    Returns: 
        span_masks_dict_with_token_keys: Contains a list of span masks for 
            ... each input_id key

    """

    noun_chunks_word_level = _get_noun_chunks_word_level(
            dataset)

    chunks_token_level = _find_all_pos_tags_token_level(
            params,
            noun_chunks_word_level,
            dataset,
            tokenizer)

    # We create masks based on the segments
    span_masks_dict = _create_span_masks_from_segments(
            chunks_token_level,
            consecutive_segments_list)

    # Changing dictionary keys to the input ids
    span_masks_dict_token_keys = _change_keys_to_input_ids(
            span_masks_dict,
            dataset,
            tokenizer)

    return span_masks_dict_token_keys


def _change_keys_to_input_ids(
        span_masks_dict: dict, 
        dataset: Dataset, 
        tokenizer: BertTokenizerFast) -> dict:
    """
    We change the span dictionary keys to input_ids from observation numbers

    Args:
        span_masks_dict: dictionary with list of masks for each observation
        dataset: dataset that we are finding the span masks for
        tokenizer: tokenizer used to find input_ids

    Returns:
        span_masks_dict_token_keys: dictionary with masks for each input_id

    """

    span_masks_dict_token_keys = {}

    for obs_no in span_masks_dict.keys():
        obs_tokens = tokenizer(dataset[obs_no]['premise'],
            dataset[obs_no]['hypothesis'])['input_ids']
        span_masks_dict_token_keys[str(obs_tokens)] = span_masks_dict[obs_no]
        
    return span_masks_dict_token_keys


def _create_span_masks_from_segments(
        pos_token_level: dict,
        consecutive_segments_list: list) -> dict:

    """
    We convert single segments into a list with combinations of
    consecutive segments.

    We include each individual segment, consecutive segments,
    and also three segments in a row

    Args:
    pos_token_level: A dictionary where each numbered observation key contains:
        1) 'segments': containing a list allocating tokens to segments
        2) 'token_level_pos_words': spacy words mapped to each token pos tag
    consecutive_segments_list: A list of which combinations of segments make
        ... the spans

    Returns:
        span_mask_dict: A dictionary with a list of masks for each obs number

    """

    span_mask_dict = {}
    for obs_no in pos_token_level.keys():
        
        span_mask_dict[obs_no] = []
        segment_values = list(
                set(pos_token_level[obs_no]['segments']))
        
        for segment in segment_values:
            # First: create masks based on individual spans
            mask = [1 if pos_token_level[obs_no]['segments'][idx] \
                    == segment else 0 for idx in \
                    range(len(pos_token_level[obs_no]['segments']))]
            if 1 in consecutive_segments_list:
                span_mask_dict[obs_no].append((mask,0))

            # Now we create masks based on consecutive spans
            mask = [1 if pos_token_level[obs_no]['segments'][idx] \
                    == segment or \
                    pos_token_level[obs_no]['segments'][idx] \
                    == segment + 1 \
                    else 0 for idx in \
                    range(len(pos_token_level[obs_no]['segments']))]
            
            if 2 in consecutive_segments_list:
                span_mask_dict[obs_no].append((mask,1))

            # Now we create masks based on longer consecutive spans
            mask = [1 if pos_token_level[obs_no]['segments'][idx] \
                    == segment or \
                    pos_token_level[obs_no]['segments'][idx] \
                    == segment + 1 or \
                    pos_token_level[obs_no]['segments'][idx] \
                    == segment + 2 \
                    else 0 for idx in \
                    range(len(pos_token_level[obs_no]['segments']))]

            if 3 in consecutive_segments_list:
                span_mask_dict[obs_no].append((mask,2))

        # Make sure the span lists are unique
        unique_span_list = []
        list_with_overlap_status = []
        for span in span_mask_dict[obs_no]:
            if span[0] not in unique_span_list:
                unique_span_list.append(span[0])
                list_with_overlap_status.append(span)

        span_mask_dict[obs_no] = list_with_overlap_status

    return span_mask_dict


def _get_noun_chunks_word_level(
        dataset: Dataset) -> dict:

    """
    Allocates words to segments based on where noun chunks are in the sentence

    Args:
        dataset: dataset that we are finding spans for

    Returns:
        pos_tags_dict_word_level
    """

    nlp = spacy.load('en_core_web_sm')
    pos_tags_dict_word_level = {}

    for idx, obs in enumerate(dataset):
        hypothesis = obs['hypothesis']
        document = nlp(hypothesis)
        pos_tags_dict_word_level[idx] = {}

        noun_chunk_end_points = []

        for chunk in document.noun_chunks:

            end = chunk.end -1

            if len(document) > chunk.end:
                if str(document[chunk.end]) in ['.', ',', '!', '?', "'", '"']:
                    end = end + 1

            noun_chunk_end_points.append(end)

        pos_tags_dict_word_level[idx]['segments'] = [0]*len(document)

        for chunk_end in noun_chunk_end_points:
            for word_idx in range(
                len(document)):
                if word_idx > chunk_end:
                    pos_tags_dict_word_level[idx]['segments'][word_idx] += 1

        pos_tags_dict_word_level[idx]['words'] = []

        for word in document:
            pos_tags_dict_word_level[idx]['words'].append(str(word))

        pos_tags_dict_word_level[idx]['pos_tags'] \
                = pos_tags_dict_word_level[idx]['segments']

    return pos_tags_dict_word_level


def _find_all_pos_tags_token_level(
        params: argparse.Namespace,
        pos_tags_dict_word_level: dict, 
        dataset: Dataset, 
        tokenizer: BertTokenizerFast) -> dict:

    """
    We find the token level PoS tags based on word level PoS tags

    Args:
        pos_tags_dict_word_level: dictionary with pos_tags and words at a
            word-level for each observation number

    Returns:
        pos_token_level, the dictionary containing:                         
            1) 'segments': pos tags mapped to a token level
            2) 'token_level_pos_words': spacy words mapped to each token pos tag
    """

    pos_token_level = {}
    for idx, obs in enumerate(dataset):

        pos_token_level[idx] = {}

        # We find the bert tokens for the hypothesis
        hypothesis = obs['hypothesis']
        transformer_tokens = tokenizer.tokenize(hypothesis)
        if params.model_type[:4] == 'bert':
            
            transformer_tokens \
                    = [x.replace("##","") for x in transformer_tokens]
        
        elif params.model_type[:9] == 'microsoft': 
            
            transformer_tokens \
                    = [x.replace("Ġ", "") for x in transformer_tokens]
            
            transformer_tokens \
                    = [x.replace("Â£", "£") for x in transformer_tokens]
            
            transformer_tokens \
                    = [x.replace("Ã´", "ô") for x in transformer_tokens]
            
            transformer_tokens \
                    = [x.replace("Ã©", "é") for x in transformer_tokens]
            
            transformer_tokens \
                    = [x if x != "Ċ" else "" for x in transformer_tokens]

        # Initializing lists for our pos tags at token level
        pos_token_level[idx]['segments'] = []
        pos_token_level[idx]['token_level_pos_words'] = []
     
        words_correspond_pos_tags = pos_tags_dict_word_level[idx]['words']
        
        # Line breaks do not appear as bert tokens
        if words_correspond_pos_tags[-1] == '\n':
            words_correspond_pos_tags = words_correspond_pos_tags[:-1]    
        
        # Dict to keep track of bert/spacy characters checked so far
        bert_spacy_counts = {}
        bert_spacy_counts['bert_chars_so_far'] = 0
        bert_spacy_counts['spacy_chars_so_far'] = 0
        bert_spacy_counts['transformer_tokens_so_far'] = 0
        bert_spacy_counts['spacy_words_so_far'] = 0

        for word_idx in range(len(words_correspond_pos_tags)):
            
            # We may have skipped ahead some words, in the case
            # where multiple bert tokens align to multiple spacy words,
            # in such cases we do not want to add a PoS tag again
            if word_idx < bert_spacy_counts['spacy_words_so_far']:    
                continue
            
            bert_spacy_counts['spacy_words_so_far'] += 1
            
            pos_token_level = _add_word_pos_tags(
                    pos_tags_dict_word_level,
                    pos_token_level,
                    bert_spacy_counts,
                    words_correspond_pos_tags,
                    word_idx,
                    idx,
                    transformer_tokens)       

        # There could still be empty transformer tokens (e.g. spaces if using DeBERTa) 
        if len(pos_token_level[idx]['segments']) \
                < len(transformer_tokens):
            
            additional_tokens = len(transformer_tokens) \
                    - len(pos_token_level[idx]['segments'])
            
            last_pos_token_value = pos_token_level[idx]['segments'][-1]
            len_pos_tokens = len(pos_token_level[idx]['segments'])
            for i in range(additional_tokens):
                
                if transformer_tokens[len_pos_tokens+i] == '':
                    pos_token_level[idx]['segments'].append(
                            last_pos_token_value)

        assert len(pos_token_level[idx]['segments']) == \
                    len(transformer_tokens), \
                    "Pos tags should match the number of tokens" \
                    + str(pos_token_level[idx])\
                    + str(pos_token_level[idx]['segments']) \
                    + str(transformer_tokens)

    return pos_token_level


def _add_word_pos_tags(
        pos_tags_dict_word_level: dict,
        pos_token_level: dict, 
        bert_spacy_counts: dict, 
        words_correspond_pos_tags: dict,
        word_idx: int,
        idx: int,
        transformer_tokens) -> dict:

    """
    Add pos tags for each token associated with an individual word

    This may be for multiple words if we do not have a exactly mapping
    from one word to multple tokens


    Args:
        pos_tags_dict_word_level: dictionary with pos_tags and words at a       
            word-level for each observation number
        pos_token_level: A dictionary with:
            1) 'segments': PoS tags for the spacy words so far
            2) 'token_level_pos_words': The  words associated with each PoS tag

        bert_spacy_counts: A dictionary keeping track how far through both 
            ... the Bert tokens and Spacy words we are
        words_correpond_pos_tags: A dictionary with a list of all spacy words
            ... for each observation number
        word_idx: The word number within the observation we're considering
        int: The observation number
        transformer_tokens: token input ids

    Returns:
        pos_token_level: A dictionary with:
            1) 'segments': PoS tags for the spacy words so far
            2) 'token_level_pos_words': The words associated with each PoS tag


    """
    # We compare the chars of the word against its corresponding tokens
    word = words_correspond_pos_tags[word_idx]
    word = word.replace(" ","")

    if len(word) == 0:
        return pos_token_level

    bert_spacy_counts['spacy_chars_so_far'] += len(word)

    assert len(word) > 0, \
            "All spacy words must be at least one character"
    
    # We count how many bert tokens map to spacy words before they are aligned
    # e.g. until the number of chars match again
    bert_spacy_counts['transformer_tokens_in_word'] = 0
    bert_spacy_counts['spacy_tokens_in_word'] = len(word)

    # How many characters since PoS tags previously asssigned to tokens
    # e.g. since we last had alignment between spacy words and bert tokens:
    chars_until_align = {'bert':0, 'spacy': 1}

    bert_spacy_counts, chars_until_align = \
            _align_spacy_words_and_transformer_tokens(
            transformer_tokens,
            words_correspond_pos_tags,
            bert_spacy_counts,
            chars_until_align)

    assert bert_spacy_counts['spacy_chars_so_far'] == \
            bert_spacy_counts['bert_chars_so_far'], \
            "Words and tokens need to be aligned"
   
    pos_token_level = _pos_tag_for_transformer_tokens(
            pos_token_level,
            bert_spacy_counts,
            chars_until_align,
            pos_tags_dict_word_level,
            words_correspond_pos_tags,
            word_idx,
            idx)
    
    return pos_token_level


def _pos_tag_for_transformer_tokens(
        pos_token_level: dict,
        bert_spacy_counts: dict,
        chars_until_align: dict,
        pos_tags_dict_word_level: dict,
        words_correspond_pos_tags: list,
        word_idx: int,
        idx: int) -> dict:
    """
    We add the new pos tags, along with their corresponding spacy words,
    these tags are added to the token level dictionary (per associated token)

    Args:
        pos_token_level: A dictionary with:
            1) 'segments': PoS tags for the spacy words so far
            2) 'token_level_pos_words': The  words associated with each PoS tag

        bert_spacy_counts: A dictionary keeping track how far through both
            ... the Bert tokens and Spacy words we are
        chars_until_align: Counting how many spacy words and bert tokens since
            we last appended the corresponding PoS tags (since last alignment)
        
        pos_tags_dict_word_level: dictionary with pos_tags and words at a       
            word-level for each observation number 

        words_correpond_pos_tags: A dictionary with a list of all spacy words   
            ... for each observation number     

        word_idx: word number in the observation being considered
        idx: observation number for the observation in the dataset

    Returns:
        pos_token_level: A dictionary with:                                     
            1) 'segments': PoS tags for the spacy words so far                  
            2) 'token_level_pos_words': The words associated with each PoS tag 

    """

    word_to_use_as_pos = bert_spacy_counts['spacy_words_so_far'] \
            - chars_until_align['spacy']

    pos_tag_added = \
            pos_tags_dict_word_level[idx]['pos_tags'][word_to_use_as_pos]

    pos_token_level[idx]['segments'].extend(
            [pos_tag_added]*chars_until_align['bert'])
    
    word = ''

    for spacy_word_idx in range(
            bert_spacy_counts['spacy_words_so_far']-chars_until_align['spacy'],
            bert_spacy_counts['spacy_words_so_far']):
        word = word + words_correspond_pos_tags[spacy_word_idx]

    pos_token_level[idx]['token_level_pos_words'].extend(
            [word]*chars_until_align['bert'])

    return pos_token_level


def _align_spacy_words_and_transformer_tokens(
        transformer_tokens: list,
        words_correspond_pos_tags: list,
        bert_spacy_counts: dict,
        chars_until_align: dict) -> (dict, dict):


    if bert_spacy_counts['bert_chars_so_far'] > \
            bert_spacy_counts['spacy_chars_so_far']:
        
        bert_spacy_counts, chars_until_align = \
                        _add_spacy_words_until_at_least_bert_token_chars(
                                words_correspond_pos_tags,
                                bert_spacy_counts,
                                chars_until_align)

    if bert_spacy_counts['bert_chars_so_far'] < \
            bert_spacy_counts['spacy_chars_so_far']:

        bert_spacy_counts, chars_until_align = \
                        _add_transformer_tokens_until_at_least_spacy_token_chars(
                                words_correspond_pos_tags,
                                bert_spacy_counts,
                                chars_until_align,
                                transformer_tokens)
        
    if bert_spacy_counts['bert_chars_so_far'] != \
            bert_spacy_counts['spacy_chars_so_far']:

        return _align_spacy_words_and_transformer_tokens(
                transformer_tokens,
                words_correspond_pos_tags,
                bert_spacy_counts,
                chars_until_align)
    
    elif bert_spacy_counts['bert_chars_so_far'] \
            == bert_spacy_counts['spacy_chars_so_far']:


        return bert_spacy_counts, chars_until_align 


def _add_transformer_tokens_until_at_least_spacy_token_chars(
        words_correspond_pos_tags: list,
        bert_spacy_counts: dict,
        chars_until_align: dict,
        transformer_tokens: list) -> (dict, dict):

    start_of_loop = bert_spacy_counts['transformer_tokens_so_far']
    for token_idx in range(
            start_of_loop,
            len(transformer_tokens)):
     
        if bert_spacy_counts['bert_chars_so_far'] >= \
            bert_spacy_counts['spacy_chars_so_far']:

            return bert_spacy_counts, chars_until_align
        
        # Need Bert tokens to see if we have [UNK] or not
        token = transformer_tokens[token_idx]

        chars_until_align['bert'] += 1
        bert_spacy_counts['transformer_tokens_so_far'] += 1
        if token != '[UNK]':
            bert_spacy_counts['bert_chars_so_far'] \
                += len(transformer_tokens[token_idx])
        else:

            bert_spacy_counts['bert_chars_so_far'] += 1

    return bert_spacy_counts, chars_until_align 


def _add_spacy_words_until_at_least_bert_token_chars(
        words_correspond_pos_tags: list,
        bert_spacy_counts: dict,
        chars_until_align: dict) -> (dict, dict):
    
    start_of_loop = bert_spacy_counts['spacy_words_so_far']
    for word_idx in range(
        start_of_loop, 
        len(words_correspond_pos_tags)):
        if bert_spacy_counts['bert_chars_so_far'] <= \
            bert_spacy_counts['spacy_chars_so_far']:
            break

        chars_until_align['spacy'] += 1
        
        bert_spacy_counts['spacy_words_so_far'] += 1
        bert_spacy_counts['spacy_chars_so_far'] += \
                len(words_correspond_pos_tags[word_idx])
    
    return bert_spacy_counts, chars_until_align 

