import argparse
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.yelp_review_dataset import YelpReviewDataset
from utils.model_factory import construct_model_from_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-folder', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    args = parser.parse_args()

    output_dir = args.model_path[:args.model_path.rfind("/")]
    config_path = f"{output_dir}/config.py"
    print(f"Loading model from {args.model_path}")

    model, Config, vocab, device = construct_model_from_config(config_path)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    test_data = pd.read_csv(f'{args.csv_folder}/test.csv')

    # Reset dataframe index so that we can use df.loc[idx, 'text']
    test_data = test_data.reset_index(drop=True)
    test_dataset = YelpReviewDataset(test_data, vocab, Config.MAX_SEQ_LENGTH)

    # get dataloader from dataset
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    # test
    with torch.no_grad():
        correct = 0
        total = 0
        total_loss = 0
        TP, FP, TN, FN = 0, 0, 0, 0
        print("Testing...")
        for data, labels, _ in tqdm(test_loader):
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
