# Note: Only do 1 epoch of adversarial training.

import os
import torch
import textattack
import torch.nn as nn
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

from utils.model_factory import ModelWithSigmoid
from project.utils import tokenizer


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


def text_to_adv_data(model, model_tokenizer, text, labels):
    """
    Prepare and generate adversarial examples for adversarial training.
    """
    # Update wrap model because we attack the newest model every batch
    model_wrapper = PyTorchModelWrapper(
        ModelWithSigmoid(model), model_tokenizer)

    # text is a tuple of size (batch_size), each element is a review
    text_lst = list(text)
    # labels is a batched tensor of size (batch_size)
    labels_lst = labels.tolist()
    train_dataset = create_ta_dataset(text_lst, labels_lst, 1500)

    # Generate adversarial examples
    with torch.no_grad():
        attacked_texts = _generate_attacked_texts(
            model_wrapper, train_dataset)

    # need to convert attacked_texts to a tensor of size (batch_size, max_seq_length)
    # Convert text to ids
    data = torch.tensor(model_tokenizer(attacked_texts), dtype=torch.long)
    del attacked_texts
    return data


def get_criterion():
    criterion = nn.BCEWithLogitsLoss()
    return criterion


def get_optimizer(model, Config):
    if hasattr(Config, 'USE_ADAMW') and Config.USE_ADAMW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE,
                                      betas=Config.BETAS, eps=Config.ADAM_EPSILON,
                                      weight_decay=Config.WEIGHT_DECAY)
        print("Using AdamW optimizer")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE,
                                     betas=Config.BETAS, eps=Config.ADAM_EPSILON,
                                     weight_decay=Config.WEIGHT_DECAY)
    return optimizer


def adversarial_training(model, Config, device, args, train_loader, val_loader, vocab):
    """
    Adversarial training.
    Note: Only do 1 epoch of adversarial training.
    """
    print("Adversarial Training...")
    # Construct model wrapper for TextAttack
    model_tokenizer = tokenizer.MyTokenizer(
        vocab, Config.MAX_SEQ_LENGTH, remove_stopwords=False)

    # define binary cross entropy loss function and optimizer
    criterion = get_criterion()
    optimizer = get_optimizer(model, Config)
    val_losses, val_accuracy = [], []
    for i, (_, labels, text) in enumerate(tqdm(train_loader)):
        model.eval()
        # Generate adversarial examples
        data = text_to_adv_data(
            model, model_tokenizer, text, labels)
        # Now do the real training
        data = data.to(device)
        labels = labels.unsqueeze(1).float()  # (batch_size, 1)
        labels = labels.to(device)

        # Apply label smoothing by changing labels from 0, 1 to 0.1, 0.9
        if Config.LABEL_SMOOTHING:
            labels = (1 - Config.LABEL_SMOOTHING_EPSILON) * labels + \
                Config.LABEL_SMOOTHING_EPSILON * (1 - labels)

        model.train()
        # forward
        # A temporary fix for device mismatch when running on parallel
        if model.embedding.weight.device != data.device:
            model.to(data.device)
        outputs = model(data)
        loss = criterion(outputs, labels)
        # backward
        optimizer.zero_grad()
        loss.backward()
        if Config.GRADIENT_CLIP:
            # clip gradient norm
            nn.utils.clip_grad_norm_(model.parameters(),
                                        max_norm=Config.GRADIENT_CLIP_VALUE)
        optimizer.step()
        del data, labels, outputs, _
        torch.cuda.empty_cache()
            
    # save model to at_model.pt
    # Note: in adv training we save the model at every 1/10 of each training
    # see example-train-adv.sh for more details 
    print(f"Saving model to {args.output_dir}/at_model.pt")
    torch.save(model.state_dict(), f'{args.output_dir}/at_model.pt')

    # evaluate on validation set every 100 batches
    model.eval()
    with torch.no_grad():
        total_loss = total = TP = TN = 0
        print(f"Validation...")
        for data, labels, _ in tqdm(val_loader):
            data = data.to(device)
            labels = labels.unsqueeze(1).float().to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)

            TP += ((predicted == 1) & (labels == 1)).sum().item()
            TN += ((predicted == 0) & (labels == 0)).sum().item()
        del data, labels, outputs, _
        print(f"Validation Accuracy: {(TP + TN) / total:.4f}")
        print(f"Validation Loss: {total_loss / len(val_loader):.4f}")
        val_losses.append(total_loss / len(val_loader))
        val_accuracy.append((TP + TN) / total)

    # plot loss and accuracy values to file
    if args.loss_values:
        with open(f'{args.output_dir}/{os.environ["MODEL_CHOICE"]}_val_losses.txt', 'a') as f:
            f.write(f'{val_losses[-1]}\n')
        with open(f'{args.output_dir}/{os.environ["MODEL_CHOICE"]}_val_accuracy.txt', 'a') as f:
            f.write(f'{val_accuracy[-1]}\n')
