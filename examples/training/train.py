from pathlib import Path
import time
import click
import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.nn import L1Loss, SmoothL1Loss
from torch.optim import Adam
from dataset import BoardDataset


@click.command("Train a resnet18 to predict w/h ratio given heatmaps with board corners")
@click.option("--train_dataset", required=True, help="train dataset")
@click.option("--val_dataset", required=True, help="validation dataset")
@click.option("--epochs", required=True, type=int, help="number of epochs")
@click.option("--lr", required=True, type=float, help="learning rate")
@click.option("--save_path", required=True, help="folder to save final checkpoint")
def train(train_dataset, val_dataset, epochs, lr, save_path):
    train_dataset = BoardDataset(train_dataset)
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=8, batch_size=32)
    val_dataset = BoardDataset(val_dataset)
    val_dataloader = DataLoader(val_dataset, shuffle=True, num_workers=8, batch_size=32)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 1)
    model.to('cuda')
    # l1_loss = L1Loss()
    l1_loss = SmoothL1Loss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        start = time.time()
        total_loss = 0
        model.train()
        for i, sample in enumerate(train_dataloader):
            x = sample['image']
            r = sample['ratio']
            x = x.to('cuda')
            r = r.to('cuda')
            optimizer.zero_grad()
            y = model(x)
            loss = l1_loss(y, r)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        end = time.time()
        print(f'epoch {epoch} in {end - start} seconds: train loss {total_loss / len(train_dataloader)}')

        model.eval()
        with torch.no_grad():
            start = time.time()
            total_loss = 0
            model.train()
            for i, sample in enumerate(val_dataloader):
                x = sample['image']
                r = sample['ratio']
                x = x.to('cuda')
                r = r.to('cuda')
                y = model(x)
                loss = l1_loss(y, r)
                total_loss += loss.item()

            end = time.time()
            print(f'epoch {epoch} in {end - start} seconds: val loss {total_loss / len(val_dataloader)}')

    torch.save(model.state_dict(), Path(save_path) / 'checkpoint.ckpt')


if __name__ == "__main__":
    train()
