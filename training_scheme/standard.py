import os
import torch
import torch.nn as nn
from tqdm import tqdm


def standard_training(model, Config, device, args, train_loader, val_loader):
    print("Standard Training...")
    # define binary cross entropy loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    if hasattr(Config, 'USE_ADAMW') and Config.USE_ADAMW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE,
                                      betas=Config.BETAS, eps=Config.ADAM_EPSILON,
                                      weight_decay=Config.WEIGHT_DECAY)
        print("Using AdamW optimizer")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE,
                                     betas=Config.BETAS, eps=Config.ADAM_EPSILON,
                                     weight_decay=Config.WEIGHT_DECAY)

    # start training
    train_losses, val_losses, val_accuracy = [], [], []
    for epoch in range(Config.NUM_EPOCHS):
        total_loss = 0
        model.train()
        for i, (data, labels) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            labels = labels.unsqueeze(1).float()  # (batch_size, 1)
            labels = labels.to(device)

            # Apply label smoothing by changing labels from 0, 1 to 0.1, 0.9
            if Config.LABEL_SMOOTHING:
                labels = (1 - Config.LABEL_SMOOTHING_EPSILON) * labels + \
                    Config.LABEL_SMOOTHING_EPSILON * (1 - labels)

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
            for data, labels in tqdm(val_loader):
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
