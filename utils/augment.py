# Use Textfooler and a model to generate adversarial examples
# for adversarial training.

import argparse
import pandas as pd
import textattack
import torch
import os

from tqdm import tqdm
from textattack import Attacker
from textattack import AttackArgs
from textattack.attack_recipes import TextFoolerJin2019
from textattack.attack_results import (
    FailedAttackResult,
    MaximizedAttackResult,
    SkippedAttackResult,
    SuccessfulAttackResult,
)
from textattack.models.wrappers import PyTorchModelWrapper
from torch.utils.data import DataLoader

from model_factory import construct_model_from_config, ModelWithSigmoid
from yelp_review_dataset import YelpReviewDataset
from tokenizer import MyTokenizer


def _generate_attacked_texts(model_wrapper, train_dataset):
    """
    Adapted from https://github.com/Falanke21/TextAttack/blob/master/textattack/trainer.py
    Generate adversarial examples using attacker.
    params:
        args: command line arguments from train.py
        model_wrapper: PyTorchModelWrapper from TextAttack
        train_dataset: training dataset wrapped by textattack.datasets.Dataset
        epoch: current epoch of training
    """

    attack = TextFoolerJin2019.build(model_wrapper)
    num_train_adv_examples = len(train_dataset)
    # generate example for all of training data.
    attack_args = AttackArgs(
        num_examples=num_train_adv_examples,
        num_examples_offset=0,
        query_budget=100,
        shuffle=False,
        parallel=False,
        num_workers_per_device=1,
        disable_stdout=True,
        silent=True,
    )
    attack_args.attack_recipe = "textfooler"
    attack_args.model_cache_size = 0
    attack_args.constraint_cache_size = 0

    attacker = Attacker(attack, train_dataset, attack_args=attack_args)
    results = attacker.attack_dataset()

    # atacked_texts is a list which might create reference which leads to memory leak
    attacked_texts = []
    # attacked_texts will be a list of attacked text
    for r in results:
        # a successful attack
        if isinstance(r, (SuccessfulAttackResult, MaximizedAttackResult)):
            attacked_texts.append(
                tuple(r.perturbed_result.attacked_text._text_input.values())[0])
        # a failed attack
        elif isinstance(r, FailedAttackResult):
            attacked_texts.append(
                tuple(r.original_result.attacked_text._text_input.values())[0])
        # a skipped attack
        elif isinstance(r, SkippedAttackResult):
            attacked_texts.append(
                tuple(r.original_result.attacked_text._text_input.values())[0])
        else:
            raise ValueError(f"Unknown attack result type {type(r)}")

    attack.clear_cache()
    # Delete TextAttack related objects to free up memory
    del attacker, results, attack, model_wrapper, train_dataset, attack_args
    return attacked_texts


def create_ta_dataset(text_lst, labels_lst, max_char_length=1500):
    """
    Create a textattack dataset from a list of text and labels.
    Filter text that is too long. This prevents memory error.
    A review with over 1500 characters is overkill.
    """
    for i in range(len(text_lst)):
        if len(text_lst[i]) > max_char_length:
            text_lst[i] = text_lst[i][:max_char_length]
    # Transform data into a list of tuples [(text, label), ...]
    train_dataset = list(zip(text_lst, labels_lst))  # list of tuples
    # Wrap batch data into textattack dataset
    train_dataset = textattack.datasets.Dataset(train_dataset)
    return train_dataset


def attack_and_save(train_dataset, output_csv_path):
    # Generate adversarial examples
    model_tokenizer = MyTokenizer(
        vocab, Config.MAX_SEQ_LENGTH, remove_stopwords=False)
    model_wrapper = PyTorchModelWrapper(
        ModelWithSigmoid(model), model_tokenizer)

    print(f"Saving adversarial examples to {output_csv_path}...")
    # write header to csv file if it doesn't exist
    if not os.path.exists(output_csv_path):
        df = pd.DataFrame({'text': [], 'label': []})
        df.to_csv(output_csv_path, mode='a', header=True, index=False)
    # text_to_adv_data
    for i, (_, labels, text) in enumerate(tqdm(train_loader)):
        # text is a tuple of size (batch_size), each element is a review
        text_lst = list(text)
        # labels is a batched tensor of size (batch_size)
        labels_lst = labels.tolist()
        train_dataset = create_ta_dataset(text_lst, labels_lst, 1500)

        # Generate adversarial examples
        with torch.no_grad():
            attacked_texts = _generate_attacked_texts(
                model_wrapper, train_dataset)

        # Save and append to a csv file in the same folder as the trained model,
        # with the same format as the original csv file
        # Save to a csv file
        df = pd.DataFrame({'text': attacked_texts, 'label': labels_lst})
        df.to_csv(output_csv_path, mode='a', header=False, index=False)
        del attacked_texts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-folder', type=str, required=True)
    parser.add_argument('--load-trained', type=str, required=True,
                        help='Load trained model from a .pt file')
    parser.add_argument('--attack-batch-size', type=int, default=200,
                        help='Batch size for generating and saving adversarial \
                              examples, Note: attack generation is still \
                              sequential')
    parser.add_argument('--data-proportion', type=float, default=0.5,
                        help='Proportion of training data to use for \
                              adversarial training')
    parser.add_argument('--concat-with-original', action='store_true',
                        help='Concatenate original training data with \
                              adversarial examples in output csv file')
    args = parser.parse_args()

    # default config file to output_dir/config.py
    output_dir = args.load_trained[:args.load_trained.rfind("/")]
    config_path = f'{output_dir}/config.py'

    # Constructing model...
    model, Config, vocab, device = construct_model_from_config(config_path)

    model.load_state_dict(torch.load(args.load_trained))
    print(f"Loaded trained model from {args.load_trained}")
    model.to(device)
    model.eval()

    # Load data
    train_data = pd.read_csv(f'{args.csv_folder}/train.csv')
    # Use only a proportion of training data for adversarial training
    train_data = train_data.sample(frac=args.data_proportion)
    # reset index
    train_data = train_data.reset_index(drop=True)
    print(
        f"Using {args.data_proportion} of training data for adversarial training")

    train_dataset = YelpReviewDataset(
        train_data, vocab, Config.MAX_SEQ_LENGTH)
    # get dataloader from dataset
    train_loader = DataLoader(
        train_dataset, batch_size=args.attack_batch_size, shuffle=False)

    new_data_dir = f'{output_dir}/augment_csv_concat' if args.concat_with_original \
        else f'{output_dir}/augment_csv'
    os.makedirs(new_data_dir, exist_ok=True)
    # copy original test.csv and val.csv to new_data_dir
    print(f"Copying original test.csv and val.csv to {output_dir}...")
    os.system(f"cp {args.csv_folder}/test.csv {new_data_dir}")
    os.system(f"cp {args.csv_folder}/val.csv {new_data_dir}")
    output_csv_path = f'{new_data_dir}/train.csv'
    attack_and_save(train_dataset, output_csv_path)

    if args.concat_with_original:
        # Concatenate original training data with adversarial examples
        # in output csv file
        # Load original training data
        original_train_data = pd.read_csv(f'{args.csv_folder}/train.csv')
        # Load adversarial examples
        adv_train_data = pd.read_csv(output_csv_path)
        # Concatenate
        train_data = pd.concat([original_train_data, adv_train_data])
        # Save to csv file
        train_data.to_csv(output_csv_path, index=False)
        print(
            f"Concatenated original training data with adversarial examples in {output_csv_path}")
