import yaml
import torch
from mpmath.identification import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn, optim

from yolov1.dataset import FruitDataset
from yolov1.loss import YoloV1Loss
from yolov1.model import Yolov1

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
path = './config/hyperparameter.yaml'

def load_hyperparameter(path):
    with open(path, 'r') as f:
        h = yaml.safe_load(f)

    return h

hyperparameter = load_hyperparameter(path)


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        loss = loss_fn(pred, y)
        mean_loss.append([loss.item()])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=3).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr= hyperparameter["LEARNING_RATE"], weight_decay=hyperparameter['WEIGHT_DECAY']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=3, mode='max', verbose=True)
    loss_fn = YoloV1Loss()

    train_dataset = FruitDataset(
        transform=transform,
        files_dir=files_dir
    )

    test_dataset = FruitDataset(
        transform=transform,
        files_dir=files_dir
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=hyperparameter['BATCH_SIZE'],
        shuffle=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=hyperparameter['BATCH_SIZE'],
        shuffle=True,
        drop_last=False,
    )

    for epoch in range(hyperparameter['EPOCHS']):
        train_fn(train_loader, model, optimizer, loss_fn)




