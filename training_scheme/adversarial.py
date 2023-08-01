import collections
import os
import torch
import textattack
import torch.nn as nn
from tqdm import tqdm

from textattack import Attack
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


def _generate_attacked_texts(args, model_wrapper, train_dataset, epoch):
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

    base_file_name = f"attack-train-{epoch}"
    log_file_name = os.path.join(args.output_dir, base_file_name)
    print("Attacking model to generate new adversarial training set...")

    num_train_adv_examples = len(train_dataset)
    # generate example for all of training data.
    attack_args = AttackArgs(
        num_examples=num_train_adv_examples,
        num_examples_offset=0,
        query_budget=None,
        shuffle=False,
        parallel=False,
        num_workers_per_device=1,
        disable_stdout=True,
        silent=True,
        log_to_txt=log_file_name + ".txt",
        log_to_csv=log_file_name + ".csv",
    )
    attack_args.attack_recipe = "textfooler"

    attacker = Attacker(attack, train_dataset, attack_args=attack_args)
    results = attacker.attack_dataset()

    attack_types = collections.Counter(r.__class__.__name__ for r in results)
    total_attacks = (
        attack_types["SuccessfulAttackResult"] +
        attack_types["FailedAttackResult"]
    )
    success_rate = attack_types["SuccessfulAttackResult"] / total_attacks * 100
    print(f"Total number of attack results: {len(results)}")
    print(
        f"Attack success rate: {success_rate:.2f}% [{attack_types['SuccessfulAttackResult']} / {total_attacks}]"
    )
    # TODO: This will produce a bug if we need to manipulate ground truth output.

    # To Fix Issue #498 , We need to add the Non Output columns in one tuple to represent input columns
    # Since adversarial_example won't be an input to the model , we will have to remove it from the input
    # dictionary in collate_fn
    attacked_texts = []
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

    return attacked_texts


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
    print("Adversarial Training...")
    # Construct model wrapper for TextAttack
    model_tokenizer = tokenizer.MyTokenizer(
        vocab, Config.MAX_SEQ_LENGTH, remove_stopwords=False)

    # define binary cross entropy loss function and optimizer
    criterion = get_criterion()
    optimizer = get_optimizer(model, Config)

    train_losses, val_losses, val_accuracy = [], [], []
    for epoch in range(Config.NUM_EPOCHS):
        total_loss = 0
        for i, (_, labels, text) in enumerate(tqdm(train_loader)):
            # Update wrap model because we attack the newest model every batch
            model_wrapper = PyTorchModelWrapper(
                ModelWithSigmoid(model), model_tokenizer)

            # text is a tuple
            text_lst = list(text)
            # labels is a batched tensor of size (batch_size)
            labels_lst = labels.tolist()
            # Transform data into a list of tuples [(text, label), ...]
            train_dataset = list(zip(text_lst, labels_lst))  # list of tuples
            # Wrap batch data into textattack dataset
            train_dataset = textattack.datasets.Dataset(train_dataset)

            # Generate adversarial examples
            attacked_texts = _generate_attacked_texts(
                args, model_wrapper, train_dataset, epoch)
            for i in range(len(attacked_texts)):
                if attacked_texts[i] != text_lst[i]:
                    print(f"found different text: {attacked_texts[i]} \n vs \n {text_lst[i]}")

            assert False
            # need to convert attacked_texts to a tensor of size (batch_size, max_seq_length)

            # Convert text to ids
            data = torch.tensor(model_tokenizer(attacked_texts), dtype=torch.long)

            data = data.to(device)
            labels = labels.unsqueeze(1).float()  # (batch_size, 1)
            labels = labels.to(device)

            # Apply label smoothing by changing labels from 0, 1 to 0.1, 0.9
            if Config.LABEL_SMOOTHING:
                labels = (1 - Config.LABEL_SMOOTHING_EPSILON) * labels + \
                    Config.LABEL_SMOOTHING_EPSILON * (1 - labels)

            model.train()
            # forward
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # backward
            optimizer.zero_grad()
            loss.backward()
            if Config.GRADIENT_CLIP:
                # clip gradient norm
                nn.utils.clip_grad_norm_(model.parameters(),
                                         max_norm=Config.GRADIENT_CLIP_VALUE)
            optimizer.step()

            # update tqdm with loss value every a few batches
            NUM_PRINT_PER_EPOCH = 3
            if (i+1) % (len(train_loader) // NUM_PRINT_PER_EPOCH) == 0:
                # if (i+1) % (Config.BATCH_SIZE * 3) == 0:
                tqdm.write(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}, \
                            Batch {i+1}/{len(train_loader)}, \
                            Batch Loss: {loss.item():.4f}, \
                            Average Loss: {total_loss / (i+1):.4f}")
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}, \
              Average Loss: {total_loss / len(train_loader):.4f}")
        # save loss for plot
        train_losses.append(total_loss / len(train_loader))
        # save checkpoint
        if args.checkpoints:
            try:
                checkpoint_path = f'{args.output_dir}/checkpoints/{os.environ["MODEL_CHOICE"]}_model_epoch{epoch+1}.pt'
                torch.save(model.state_dict(), checkpoint_path)
            except OSError as e:
                print(
                    f"Could not save checkpoint at epoch {epoch+1}, error: {e}")

        # evaluate on validation set if necessary
        model.eval()
        with torch.no_grad():
            total_loss = total = TP = TN = 0
            print(f"Validation at epoch {epoch + 1}...")
            for data, labels, text in tqdm(val_loader):
                data = data.to(device)
                labels = labels.unsqueeze(1).float().to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                predicted = torch.round(torch.sigmoid(outputs))
                total += labels.size(0)

                TP += ((predicted == 1) & (labels == 1)).sum().item()
                TN += ((predicted == 0) & (labels == 0)).sum().item()
            print(f"Validation Accuracy: {(TP + TN) / total:.4f}")
            print(f"Validation Loss: {total_loss / len(val_loader):.4f}")
            val_losses.append(total_loss / len(val_loader))
            val_accuracy.append((TP + TN) / total)

        # plot loss and accuracy values to file
        if args.loss_values:
            with open(f'{args.output_dir}/{os.environ["MODEL_CHOICE"]}_train_losses.txt', 'a') as f:
                f.write(f'{train_losses[-1]}\n')
            with open(f'{args.output_dir}/{os.environ["MODEL_CHOICE"]}_val_losses.txt', 'a') as f:
                f.write(f'{val_losses[-1]}\n')
            with open(f'{args.output_dir}/{os.environ["MODEL_CHOICE"]}_val_accuracy.txt', 'a') as f:
                f.write(f'{val_accuracy[-1]}\n')
