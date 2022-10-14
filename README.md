# Logical Reasoning with Span-Level Predictions for Interpretable and Robust NLI Models - Project Code

This repository contains the code supporting our EMNLP-2022 paper Logical Reasoning with Span-Level Predictions for Interpretable and Robust NLI Models:

Contact details: j.stacey20@imperial.ac.uk

## 1) Downloading the data

To download the datasets required, run setup_data.sh

## 2) Running experiments

We provide test and dev figures for a specific seed below (42), as the figures in the paper average over 5 random seeds.

### 2.1) Running SLR-NLI (without additional e-SNLI supervision)

python run.py --span_supervision 0 --name_id lr_0.75seed_42 --learning_rate 7.5e-06 --random_seed 42

SNLI-test performance: 90.30%
SNLI-dev performance:  90.89%

### 2.2) Running SLR-NLI-esnli (with additional e-SNLI supervision)

python run.py --span_supervision 1 --name_id esnli_lr_0.5seed_42 --learning_rate 5e-06 --random_seed 42

SNLI-test performance: 90.47%
SNLI-dev performance:  90.89%

### 2.3) Running SLR-NLI with dropout on consecutive spans (without additional e-SNLI supervision)

python run.py --span_supervision 0 --name_id dropout10_lr_0.75seed_42 --learning_rate 7.5e-06 --random_seed 42 --span_dropout 0.1

SNLI-test performance: 90.40%
SNLI-dev performance:  90.58%

### 2.4) Running SLR-NLI-esnli on a reduced training set

python run.py --linear_schedule 0 --epochs 10 --reduce 1 --reduce_number 1000 --span_supervision 1 --name_id reduced_1000_lr_1.0seed_42 --learning_rate 1e-05 --random_seed 42

Results from epoch 5 (best dev performance):

SNLI-test performance: 75.25%
SNLI-dev performance:  74.96%




