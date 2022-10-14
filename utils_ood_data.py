import os
from data_other_ood import add_snli_hard, \
        add_hans, add_sick


def load_ood_datasets(eval_dataloaders, params_training, tokenizer):

    # Load SICK
    sick_dict = add_sick(
            'sick_test',
            os.getcwd() + "/data/SICK/uncorrected_SICK/s1.test",
            os.getcwd() + "/data/SICK/uncorrected_SICK/s2.test",
            os.getcwd() + "/data/SICK/uncorrected_SICK/labels.test",
            params_training,
            tokenizer)

    eval_dataloaders.update(sick_dict)


    # Load SICK
    sick_dict_dev = add_sick(
            'sick_dev',
            os.getcwd() + "/data/SICK/uncorrected_SICK/s1.dev",
            os.getcwd() + "/data/SICK/uncorrected_SICK/s2.dev",
            os.getcwd() + "/data/SICK/uncorrected_SICK/labels.dev",
            params_training,
            tokenizer)

    eval_dataloaders.update(sick_dict_dev)

    # Load SICK (corrected)
    sick_dict_corrected = add_sick(
            'sick_test_corrected',
            os.getcwd() + "/data/SICK/corrected_SICK/s1.test",
            os.getcwd() + "/data/SICK/corrected_SICK/s2.test",
            os.getcwd() + "/data/SICK/corrected_SICK/labels.test",
            params_training,
            tokenizer)

    eval_dataloaders.update(sick_dict_corrected)

     # Load SICK (corrected)
    sick_dict_dev_corrected = add_sick(
            'sick_dev_corrected',
            os.getcwd() + "/data/SICK/corrected_SICK/s1.dev",
            os.getcwd() + "/data/SICK/corrected_SICK/s2.dev",
            os.getcwd() + "/data/SICK/corrected_SICK/labels.dev",
            params_training,
            tokenizer)

    eval_dataloaders.update(sick_dict_dev_corrected)

    # Load SNLI-hard
    snli_hard_dict = add_snli_hard('snli_hard',
            os.getcwd() + "/data/snli_hard.jsonl",
            params_training,
            tokenizer)

    eval_dataloaders.update(snli_hard_dict)

    # Load HANS
    heuristic_list = ['lexical_overlap', 'subsequence', 'constituent']

    hans_dict = add_hans('hans',
            os.getcwd() + "/data/heuristics_evaluation_set.txt",
            params_training,
            tokenizer,
            heuristic_list)

    eval_dataloaders.update(hans_dict)

    return eval_dataloaders

