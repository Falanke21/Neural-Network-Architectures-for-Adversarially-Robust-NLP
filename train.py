import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.model_factory import construct_model_from_config
from utils.yelp_review_dataset import YelpReviewDataset

from training_scheme.adversarial import adversarial_training
from training_scheme.standard import standard_training


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-folder', type=str, required=True)
    parser.add_argument('--load-trained', type=str,
                        help='Load trained model from a .pt file')
    parser.add_argument('--output-dir', type=str, default='tmp')
    parser.add_argument('--checkpoints', action='store_true', default=False)
    parser.add_argument('--loss-values', action='store_true',
                        default=False, help='Output txt files of loss values')
    parser.add_argument('--adversarial-training', action='store_true', default=False,
                        help='Use adversarial training rather than standard training')
    args = parser.parse_args()

    # default config file to output_dir/config.py
    config_path = f'{args.output_dir}/config.py'

    # create output directory if necessary
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.checkpoints:
        if not os.path.exists(f'{args.output_dir}/checkpoints'):
            os.makedirs(f'{args.output_dir}/checkpoints')
    print(f'Will write outputs to "{args.output_dir}"')

    # Constructing model...
    model, Config, vocab, device = construct_model_from_config(config_path)

    if args.load_trained:
        model.load_state_dict(torch.load(args.load_trained))
        print(f"Loaded trained model from {args.load_trained}!")
    # print num of parameters
    print(
        f'Number of trainable parameters: \
        {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    model.train()
    model.to(device)

    # load data
    print(f"Loading data from {args.csv_folder}")
    train_data = pd.read_csv(f'{args.csv_folder}/train.csv')
    val_data = pd.read_csv(f'{args.csv_folder}/val.csv')
    if Config.UPSAMPLE_NEGATIVE:
        # Upsample negative reviews according to Config.UPSAMPLE_RATIO
        train_data_pos = train_data[train_data['label'] == 1]
        train_data_neg = train_data[train_data['label'] == 0]
        train_data_neg_upsampled = train_data_neg.sample(
            n=int(len(train_data_neg) * Config.UPSAMPLE_RATIO), replace=True)
        train_data = pd.concat([train_data_pos, train_data_neg_upsampled])
        print(f"Upsampled negative reviews by {Config.UPSAMPLE_RATIO}x")
    
    # Reset dataframe index so that we can use df.loc[idx, 'text']
    train_data = train_data.reset_index(drop=True)
    print(
        f"Num positive reviews in training set: {len(train_data[train_data['label'] == 1])}")
    print(
        f"Num negative reviews in training set: {len(train_data[train_data['label'] == 0])}")

    train_dataset = YelpReviewDataset(
        train_data, vocab, Config.MAX_SEQ_LENGTH)
    # get dataloader from dataset
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_data = val_data.reset_index(drop=True)
    val_dataset = YelpReviewDataset(
        val_data, vocab, Config.MAX_SEQ_LENGTH)
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # train model
    if args.adversarial_training:
        adversarial_training(model, Config, device, args,
                             train_loader, val_loader, vocab)
    else:
        standard_training(model, Config, device, args,
                          train_loader, val_loader)
    print(f"Training complete with output directory {args.output_dir}")
