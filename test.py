import argparse
import importlib
import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from train import YelpReviewDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-folder', type=str, required=True)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model-choice', type=str,
                        required=True, choices=['lstm', 'transformer'])
    parser.add_argument('--output-dir', type=str, default='models')
    args = parser.parse_args()

    # default config file to output_dir/config.py
    config_file = f'{args.output_dir}/config.py'
    # check config file exists
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found")
    # import configs
    spec = importlib.util.spec_from_file_location("Config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    if args.model_choice == 'lstm':
        Config = config_module.LSTMConfig
        from lstm.my_lstm import MyLSTM
    elif args.model_choice == 'transformer':
        Config = config_module.TransformerConfig
        from transformer.my_transformer import MyTransformer
    print(f"Using config {Config.__name__} from {config_file}")

    device = torch.device(
        'cuda' if Config.USE_GPU and torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device.type == 'cuda':
        print(f'Device count: {torch.cuda.device_count()}')
        print(f'Current device index: {torch.cuda.current_device()}')
        print(f'Device name: {torch.cuda.get_device_name(0)}')
        print(
            f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3} GB')

    # load custom vocab or GloVe
    if Config.USE_GLOVE:
        import torchtext
        glove = torchtext.vocab.GloVe(
            name='6B', dim=Config.GLOVE_EMBEDDING_SIZE,
            cache=Config.GLOVE_CACHE_DIR)
        vocab = glove
    else:
        with open(args.vocab, 'rb') as f:
            vocab = pickle.load(f)

    test_data = pd.read_csv(f'{args.csv_folder}/test.csv')

    # Reset dataframe index so that we can use df.loc[idx, 'text']
    test_data = test_data.reset_index(drop=True)
    test_dataset = YelpReviewDataset(test_data, vocab, Config.MAX_SEQ_LENGTH)

    # get dataloader from dataset
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # load lstm model
    if args.model_choice == 'lstm':
        model = MyLSTM(Config=Config, vocab_size=len(
            vocab), num_classes=1, device=device)
    elif args.model_choice == 'transformer':
        model = MyTransformer(Config=Config, vocab_size=len(
            vocab), output_dim=1, device=device)
    model.load_state_dict(torch.load(args.model))
    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    # test
    with torch.no_grad():
        correct = 0
        total = 0
        total_loss = 0
        TP, FP, TN, FN = 0, 0, 0, 0
        print("Testing...")
        for data, labels in tqdm(test_loader):
            data = data.to(device)
            labels = labels.unsqueeze(1).float().to(device)
            outputs = model(data)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)

            TP += ((predicted == 1) & (labels == 1)).sum().item()
            FP += ((predicted == 1) & (labels == 0)).sum().item()
            TN += ((predicted == 0) & (labels == 0)).sum().item()
            FN += ((predicted == 0) & (labels == 1)).sum().item()
        print(f"Accuracy: {(TP + TN) / total:.4f}")
        print(f"Test Loss: {total_loss / len(test_loader):.4f}")

    # print confusion matrix
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
