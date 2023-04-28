import torch
import torch.nn as nn

from transformer import my_transformer

if __name__ == '__main__':
    USE_GPU = False
    device = torch.device(
        'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')

    # 1. load data
    # TODO

    # 2. define model
    d_model = 512
    ffn_hidden = 2048
    n_head = 8
    drop_prob = 0.1
    output_dim = 2

    model = my_transformer.MyTransformer(vocab_size, d_model, ffn_hidden,
                                         output_dim, n_head, drop_prob, device).to(device)
    # print number of parameters
    print('Number of parameters:', sum(p.numel()
          for p in model.parameters() if p.requires_grad))

    # 3. define loss/optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

    # 4. train
    num_epochs = 1
    for epoch in range(num_epochs):
        for idx, batch in enumerate(train_loader):
            x = batch.Text
            y = batch.Label

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'step =', '%04d' %
                      (idx + 1), 'loss =', '{:.6f}'.format(loss))

    # 6. test
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in
