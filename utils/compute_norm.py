import argparse
import csv
import importlib
import pickle
import os
import torch

from model_factory import construct_model_from_config


def compute_norm(my_model: torch.nn.Module):
    # Compute the norm
    sum_of_norm = 0
    sum_of_norm_no_embedding = 0
    for name, param in my_model.named_parameters():
        if param.requires_grad:
            norm = torch.linalg.norm(param).item()
            # print(f"{name}: {norm:.2f}")
            sum_of_norm += norm
            if name != "embedding.weight":
                # we separately compute the norm of embedding
                sum_of_norm_no_embedding += norm

    print(f"Sum of norm: {sum_of_norm:.2f}")
    print(f"Sum of norm without embedding: {sum_of_norm_no_embedding:.2f}")
    return sum_of_norm, sum_of_norm_no_embedding


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-folder', type=str, required=True)
    parser.add_argument('--load-trained', type=str, required=True,
                        help='Load trained model from a .pt file')
    # parser.add_argument('--output-dir', type=str, default='tmp')
    args = parser.parse_args()

    # We will make output_dir to be the parent directory of load_trained
    args.output_dir = os.path.dirname(args.load_trained)

    # default config file to output_dir/config.py
    config_path = f'{args.output_dir}/config.py'

    # Constructing model...
    model, Config, vocab, device = construct_model_from_config(config_path)

    if args.load_trained:
        model.load_state_dict(torch.load(args.load_trained))
        print(f"Loaded trained model from {args.load_trained}!")
    # print num of parameters
    print(
        f'Number of trainable parameters: \
        {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    compute_norm(model)
