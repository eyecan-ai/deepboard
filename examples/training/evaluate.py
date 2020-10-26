from pathlib import Path
import click
import cv2
import numpy as np
import torch
from torch import nn
import torchvision.models as models


@click.command("Evaluate a resnet18 model to predict w/h ratio")
@click.option("--checkpoint", required=True, help="checkpoint file")
@click.option("--folder", required=True, help="folder with heatmaps")
def evaluate(checkpoint, folder):
    images = sorted(Path(folder).glob('*.exr'))
    images = [cv2.imread(str(x), -1) for x in images]
    images = [np.repeat(x[:, :, np.newaxis], 3, axis=2) for x in images]

    ckpt = torch.load(checkpoint)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 1)
    model.load_state_dict(ckpt)
    model.to('cuda')

    model.eval()
    with torch.no_grad():
        for img in images:
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            img = img.to('cuda')
            y = model(img)
            print(y)


if __name__ == "__main__":
    evaluate()
