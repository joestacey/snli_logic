
import os
from torch import nn
import numpy as np
import transformers
import torch
import random
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
from transformers import AdamW
from datasets import load_dataset
import sys
import argparse
from typing import Tuple
from torch.utils.data.dataloader import DataLoader

from utils_log import Log
from utils_ood_data import load_ood_datasets
from utils_metrics import calculate_f1_and_print, update_confusion_matrix
from data_prep import create_dictionary_of_hf_ood_datasets, \
        prepare_all_dataloaders
from models import LogicModel
import warnings
warnings.filterwarnings('ignore')

def get_args():

  parser = argparse.ArgumentParser(description="Training model parameters")

  # Arguments for modelling different scenarios
  parser.add_argument("--model_type", type=str, default="bert-base-uncased", 
          help="Model to be used")
  parser.add_argument("--random_seed", type=int, default=42, 
          help="Choose the random seed")

  # Arguments for model training
  parser.add_argument("--epochs", type=int, default=2, 
          help="Number of epochs for training")
  parser.add_argument("--batch_size", type=int, default=8, 
          help="batch_size")
  parser.add_argument("--learning_rate", type=float, default=5e-6, 
          help="Choose learning rate")
  parser.add_argument("--reduce", type=int, default=0, 
          help="Reduce dataset or not")
  parser.add_argument("--reduce_number", type=int, default=1000,
          help="Number of observations to reduce to")

  # Evaluate-only options
  parser.add_argument("--eval_only", type=int, default=0,
          help="Only evaluate without training")
  parser.add_argument("--model_file", type=str, default='saved_model.pt',
          help="File to load model from if eval_only mode")

  # Learning schedule arguments
  parser.add_argument("--linear_schedule", type=int, default=1, 
          help="To use linear schedule with warm up or not")
  parser.add_argument("--warmup_epochs", type=int, default=1, 
          help="Warm up period")
  parser.add_argument("--warmdown_epochs", type=int, default=1, 
          help="Warm down period")

  # Span arguments
  parser.add_argument("--name_id", type=str, default="default", 
          help="Name used for saved model and log file")
  parser.add_argument("--span_groups", type=str, default='default',
          help="Groups of spans (1 for single segments, 2 for consecutive \
                  segments, etc). Provide as 1,2,3")

  # Loss multipliers
  parser.add_argument("--esnli_loss_multiplier", type=float, default=0.1,
                    help="Multiplier for esnli loss")
  parser.add_argument("--span_supervision", type=float, default=1,
          help="Supervise eSNLI spans or not")

  # Dropout arguments
  parser.add_argument("--dropout_type", type=str, default='overlap',
          help="'standard' or 'overlap'. Only 'overlap' dropout was applied")
  parser.add_argument("--span_dropout", type=float, default=0.0,
          help="Dropout acting on spans in additional attention layer")

  # Train data arguments
  parser.add_argument("--train_data", type=str, default='snli',
          help="'snli' or 'sick'")

  params, _ = parser.parse_known_args()

  return params


def set_seed(seed_value: int) -> None:
    """
    Set seed for reproducibility.

    Args:
        seed_value: chosen random seed
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def find_logic_predictions(
        att_unnorm_cont: torch.tensor, 
        att_unnorm_neutral: torch.tensor, 
        dataset_name: str) -> int:

    """
    Find logics sentence predictions based on any spans in the sentence

    Args:
        att_unnorm_cont: output from contradiction detection attention layer
        att_unnorm_neutral: output from neutral detection attention layer
        dataset_name: dataset name

    Returns:
        pred: logic prediction for NLI sentence pair
    """
    # First check if we have contradiction or not
    if torch.max(att_unnorm_cont) > 0.5:
        pred = 2
    elif torch.max(att_unnorm_neutral) > 0.5:
        pred = 1
    else:
        pred = 0
    
    if pred == 2 and dataset_name in two_class_datasets:
        pred = 1

    return pred


def find_logic_preds_span(
        outputs: dict, 
        span_no: int) -> int:
    """
    Predict class of an individual span

    Args:
        outputs: modeling outputs including unnormalized att. weights
        span_no: span number we are finding the class of

    Returns:
        pred: span predicted label
    """

    # First check if we have contradiction or not                               
    if outputs['cont']['att_unnorm'][span_no] > 0.5:                                        
        pred = 2                                                                
    elif outputs['neutral']['att_unnorm'][span_no] > 0.5:                                   
        pred = 1                                                                
    else:                                                                       
        pred = 0                                                                
                                                                                
    return pred


def evaluate_esnli_spans(
        confusion_dict: dict, 
        outputs: dict) -> (int, int, dict):
    """
    We count the number of eSNLI spans matching, and how many were correct

    Args:
        sentence_label: class of the NLI sentence pair
        outputs: outputs from logic model, including unnormalized att. weights

    Returns:
        obs_correct_esnli_span: correct eSNLI spans from observation
        obs_total_esnli_span: total eSNLI spans matched so far from observation
        confusion_dict: updated confusion dictionary
    """

    # We count eSNLI spans for the individual observation
    obs_total_esnli_span = 0
    obs_correct_esnli_span = 0
    results = []
    # We count up how many eSNLI spans there are
    for span_no in range(len(outputs['supervise_span_or_not'])):
        if outputs['supervise_span_or_not'][span_no] != False:
            expl_col_list = list(
                    outputs['supervise_span_or_not'][span_no].keys())
            for expl_col in expl_col_list:
                if outputs['supervise_span_or_not'][span_no][expl_col]:
                    obs_total_esnli_span += 1
     
                    pred = find_logic_preds_span(
                            outputs,
                            span_no)
                    confusion_dict = update_confusion_matrix(
                            outputs['true_label'], 
                            pred, 
                            confusion_dict)

                    # We check if the esnli span was predicted correctly
                    if outputs['true_label'] == pred:
                        obs_correct_esnli_span += 1

    return obs_correct_esnli_span, obs_total_esnli_span, confusion_dict

@torch.no_grad()
def evaluate(dataset_name: str, 
        dataloader_eval: Tuple[DataLoader, dict, dict, dict]) -> None:
    """
    Evaluates NLI logic model

    Args:
        dataset_name: description of the evaluation dataset
        dataloader_eval: dataloader and lookup dictionaries..
            for recalling spans, eSNLI spans
    """

    logic_model.encoder.eval()
    logic_model.attention_neutral.eval_()
    logic_model.attention_cont.eval_()

    correct_logic_att, total =  0, 0

    # Stats for evaluating span performance
    correct_esnli_span, total_esnli_span = 0, 0
    
    confusion_dict = {'e_tp': 0, 'e_fp': 0, 'e_tn': 0, 'e_fn': 0,
            'n_tp': 0, 'n_fp': 0, 'n_tn': 0, 'n_fn': 0,
            'c_tp': 0, 'c_fp': 0, 'c_tn': 0, 'c_fn': 0}

    all_spans = 0

    dataloader, span_mask_list_all, esnli_spans \
            = dataloader_eval
            
        
    final_res = []
    import pandas as pd
    counter = 0
    for i, batch in enumerate(dataloader):
        
        batch = {k: v.to(device) for k, v in batch.items()}

        for obs_no in range(batch['input_ids'].shape[0]):

            outputs = logic_model(
                    batch, 
                    obs_no, 
                    span_mask_list_all, 
                    esnli_spans)
            # Evaluating eSNLI spans
            obs_correct_esnli_span, obs_total_esnli_span, confusion_dict  = \
                evaluate_esnli_spans(confusion_dict, outputs)
            correct_esnli_span += obs_correct_esnli_span
            total_esnli_span += obs_total_esnli_span

            all_spans += len(outputs['supervise_span_or_not'])
        
            pred_logic_att = find_logic_predictions(
                    outputs['cont']['att_unnorm'], 
                    outputs['neutral']['att_unnorm'],
                    dataset_name)
            
            file1 = open(f"{dataset_name}_predictions.txt", "a")  # append mode
            file1.write(f"observation: {dataloader.dataset['pairID'][counter]}, prediction output: {pred_logic_att}, contradiction probs: {torch.max(outputs['cont']['att_unnorm'])}, neutral probs: {torch.max(outputs['neutral']['att_unnorm'])} \n")
            counter+=1
            
            total += 1
            if pred_logic_att == batch['label'][obs_no].item():
                correct_logic_att = correct_logic_att + 1

    model_log.msg(["Total accuracy (logic att):" + \
            str(round(correct_logic_att/total, 4))])

    model_log.msg(["Total correct & total:" + \
            str(round(correct_logic_att, 4)) + " & " + str(round(total, 4))])
    
    
    
    
    


def get_att_loss(
        outputs: dict, 
        class_str: str, 
        sent_loss: torch.tensor, 
        loss_esnli: torch.tensor, 
        additional_loss_term: torch.tensor) \
                -> (torch.tensor, torch.tensor, torch.tensor):
    """
    Finds the the sentence loss, esnli span loss, and additional loss term
    
    Args:
        outputs: dictionary of model outputs
        class_str: 'neutral' or 'cont' for different attention layers
        sent_loss: sentence loss for observation so far
        loss_esnli: esnli loss for observation so far
        additional_loss_term: additional loss term for observation so far

    Returns:
        sent_loss: updated sentence loss for observation
        loss_esnli: updated esnli loss for obervation
        additional_loss_term: updated additional loss for observation
    """

    # Sent loss   SENT_OUTPUT PROBABILITY DÖNÜYOR - LABEL **2 NASIL Bİ LOSS HESABI ? 
    sent_loss += (outputs[class_str]['sent_output'] \
                            - outputs[class_str]['label'])**2

    # Span loss (eSNLI supervision)
    for span_no in range(len(outputs['supervise_span_or_not'])):
        # We check if we know the label for the span
        # Note: in training only one expl is provided (Sentence2_marked_1)
        if outputs['supervise_span_or_not'][span_no] != False:
            if outputs['supervise_span_or_not'][span_no]['Sentence2_marked_1']:
                # We check no dropoput was applied to the span
                if outputs[class_str]['dropout'][span_no].item() == 1:
                    loss_esnli += torch.square(
                        outputs[class_str]['att_unnorm'][span_no] \
                                - outputs[class_str]['label'])

    # Additional loss term
    additional_loss_term += torch.square(
            torch.max(outputs[class_str]['att_unnorm']) \
                    - outputs[class_str]['label'])

    return sent_loss, loss_esnli, additional_loss_term


def train() -> None:

    # Our train dataloader
    if params.train_data == 'snli':
        dataloader_train, span_mask_list_all, esnli_spans \
            = esnli_dataloaders['esnli_train']
    elif params.train_data == 'sick':
        dataloader_train, span_mask_list_all, esnli_spans \
            = esnli_dataloaders['sick_train']
    
    # Training loop
    for epoch in range(params.epochs):

        # Set model in training mode
        logic_model.encoder.train()
        logic_model.attention_neutral.train()
        logic_model.attention_cont.train()

        for i, batch in enumerate(dataloader_train):
            print("Minibatch:", i+1)

            batch = {k: v.to(device) for k, v in batch.items()}
            for obs_no in range(batch['input_ids'].shape[0]):
        
                outputs = logic_model(
                        batch, 
                        obs_no, 
                        span_mask_list_all, 
                        esnli_spans)

                # We calculate losses
                sent_loss = torch.tensor([0.0]).to(device)
                additional_loss_term = torch.tensor([0.0]).to(device)
                loss_esnli = torch.tensor([0.0]).to(device)

                # Update the loss from the neutral and cont attention layers
                # We do not update the neutral loss if we have a contradiction
                if outputs['true_label'] == 1 or outputs['true_label'] == 0:
                    sent_loss, loss_esnli, additional_loss_term = get_att_loss(
                            outputs,
                            'neutral',
                            sent_loss,
                            loss_esnli,
                            additional_loss_term)
                
                sent_loss, loss_esnli, additional_loss_term = get_att_loss(
                            outputs,
                            'cont',
                            sent_loss,
                            loss_esnli,
                            additional_loss_term)

                loss = sent_loss + additional_loss_term

                if params.span_supervision:
                    loss += params.esnli_loss_multiplier*loss_esnli

                optimizer_encoder.zero_grad()
                optimizer_neutral.zero_grad()
                optimizer_contradiction.zero_grad()
                loss.backward(retain_graph=True)
                
                if outputs['true_label'] == 1 or outputs['true_label'] == 0:
                    optimizer_neutral.step()

                optimizer_contradiction.step()
                optimizer_encoder.step()

            if params.linear_schedule:
                if not params.reduce:
                    schedule_encoder.step()
                    schedule_neutral.step()
                    schedule_cont.step()
 
        evaluate_each_epoch(epoch)


def evaluate_each_epoch(epoch: int) -> None:
    """
    Evaluates model on each dataset after each epoch

    Args:
        epoch: epoch number (starting at 0)
    """
    model_log.msg(["Epoch:" + str(epoch+1)])

    # We evaluate on the eSNLI dev data
    for dataset_name, dataset in esnli_dataloaders.items():
        print(dataset_name)
        if dataset_name != 'esnli_train' or params.reduce:
            model_log.msg(["Dataset: " + dataset_name])
            evaluate(dataset_name, dataset)
    print('finish')
    # # We evaluate on other huggingface evaluation datasets
    # for dataset_name, dataset in eval_dataloaders.items():
    #     model_log.msg(["Dataset: " + dataset_name])
    #     evaluate(dataset_name, dataset)


def create_lr_schedules():

    if params.linear_schedule:
        if not params.reduce:
            schedule_encoder = LambdaLR(optimizer_encoder, lr_lambda)
            schedule_neutral = LambdaLR(optimizer_neutral, lr_lambda)
            schedule_cont = LambdaLR(optimizer_contradiction, lr_lambda)

            return schedule_encoder, schedule_neutral, schedule_cont

    return None, None, None


def lr_lambda(current_step: int) -> float:

    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    return max(
            0.0, float(num_training_steps - current_step) / float(
                max(1, num_training_steps - num_warmup_steps)))


if __name__ == '__main__':

    params = get_args()

    params.reduce = bool(params.reduce)
    params.linear_schedule = bool(params.linear_schedule)
    params.eval_only = bool(params.eval_only)
    params.span_supervision = bool(params.span_supervision)
    

    if params.span_groups == 'default':
        params.span_groups = [1, 2, 3]
    else:
        params.span_groups = params.span_groups.split(",")
        params.span_groups = [int(x) for x in params.span_groups]

    print(params)

    two_class_datasets = ['hans'] 

    if params.name_id == 'default':
        name_id =  str(os.getpid())
    else:
        name_id = params.name_id
    
    # Logging file
    log_file_name = 'log_logic_model_' + name_id + '.txt'
    model_log = Log(log_file_name, params)

    # Set CUDAS to GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(params.random_seed)

    # Create folder for saving models
    if not os.path.exists('savedmodel'):
        os.makedirs('savedmodel')

    # We create a dictionary of the Huggingface datasets to be OOD datasets
    eval_data_list = create_dictionary_of_hf_ood_datasets(
                params)

    tokenizer = AutoTokenizer.from_pretrained(
            params.model_type,
            truncation=False)

    # We create dataloaders for eSNLI and HuggingFace datasets
    esnli_dataloaders, eval_dataloaders = prepare_all_dataloaders(
                    eval_data_list,
                    params,
                    tokenizer)

    # # We add additional OOD datasets
    # eval_dataloaders = load_ood_datasets(
    #         eval_dataloaders,
    #         params,
    #         tokenizer)

    if params.model_type == 'microsoft/deberta-large' or \
            params.model_type == 'microsoft/deberta-xlarge':
        dim = 1024
    else:
        dim = 768

    logic_model = LogicModel(
            dim, 
            params.model_type, 
            params.span_dropout, 
            params.dropout_type)

    logic_model.to(device)

    # We create our optimizers
    optimizer_encoder = AdamW(
            list(logic_model.encoder.parameters()),
            lr=params.learning_rate)

    optimizer_neutral = AdamW(
            list(logic_model.attention_neutral.parameters()),
            lr=params.learning_rate)

    optimizer_contradiction = AdamW(
            list(logic_model.attention_cont.parameters()),
            lr=params.learning_rate)

    # We create our learning schedules
    if params.train_data == 'snli':
        dataloader_name = 'esnli_train'
    elif params.train_data == 'sick':
        dataloader_name = 'sick_train'

    num_warmup_steps = len(
            esnli_dataloaders[dataloader_name][0])*params.warmup_epochs
    warm_down_steps = len(
            esnli_dataloaders[dataloader_name][0])*params.warmdown_epochs
    num_training_steps = num_warmup_steps + warm_down_steps

    schedule_encoder, schedule_neutral, schedule_cont = create_lr_schedules()

    if params.eval_only:

        logic_model.load_state_dict(torch.load(
        os.getcwd() + "/savedmodel/" + params.model_file + '.pt'))
        evaluate_each_epoch(0)

    else:

        train()
        print("All done")
        torch.save(logic_model.state_dict(),
                            os.getcwd() + "/savedmodel/saved_model_" \
                                    + name_id + '.pt')


#python run.py --eval_only 1 --model_file saved_model_reduced_1000_seed_42

#python run.py --linear_schedule 0 --epochs 10 --reduce 1 --reduce_number 1000 --span_supervision 1 --name_id reduced_1000_seed_42 --learning_rate 1e-05 --random_seed 42 

