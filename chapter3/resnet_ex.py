import torchvision
import torch 
import sklearn.model_selection
import numpy as np
from torchvision import transforms as transforms
import os
from tqdm import tqdm
from submit import predict_subsec5
from transform import setup_crop_flip_transform, setup_center_crop_transform


def setup_train_val_split(labels, dryrun=True, seed=0):
    x = np.arange(len(labels))
    y = np.array(labels)

    splitter = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits = 1, train_size=0.8, random_state=seed
    )
    train_indices, val_indices = next(splitter.split(x, y))

    if dryrun:
        train_indices = np.random.choice(train_indices, 100, replace=False)
        val_indices = np.random.choice(val_indices, 100, replace=True)
    
    return train_indices, val_indices

def set_transform(dataset, transform):
    if isinstance(dataset, torch.utils.data.Subset):
        set_transform(dataset.dataset, transform)
    else:
        dataset.transform = transform

def get_labels(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        return get_labels(dataset.dataset)[dataset.indices]
    else:
        return np.array([img[1] for img in dataset.imgs])

def setup_train_val_datasets(data_dir, dryrun=False):
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=setup_center_crop_transform(),
    )
    labels = get_labels(dataset)
    train_indices, val_indices = setup_train_val_split(labels, dryrun=False)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    return train_dataset, val_dataset
    

def setup_train_val_loaders(data_dir, batch_size, dryrun = False):
    train_dataset, val_dataset = setup_train_val_datasets(
        data_dir, dryrun=dryrun
    )

    train_transform = setup_crop_flip_transform()
    set_transform(train_dataset, train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        drop_last=True,
        num_workers = 8
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size = batch_size, num_workers = 8
    )

    return train_loader, val_loader

def train_1epoch(model, train_loader, lossfun, optimizer, lr_scheduler, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out =  model(x)
        loss = lossfun(out, y)
        _, pred = torch.max(out.detach(), 1)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item() * x.size(0)
        total_acc += torch.sum(pred == y)
    
    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = total_acc / len(train_loader.dataset)
    return avg_acc, avg_loss

def validate_1epoch(model, val_loader, lossfun, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x = x.to(device)
            y = y.to(device)

            out =  model(x)
            loss = lossfun(out, y)
            _, pred = torch.max(out.detach(), 1)

            total_loss += loss.item() * x.size(0)
            total_acc += torch.sum(pred == y)
    
        avg_loss = total_loss / len(val_loader.dataset)
        avg_acc = total_acc / len(val_loader.dataset)
    return avg_acc, avg_loss

def train_1epoch_mixup(model, train_loader, lossfun, optimizer, lr_scheduler, mixup_alpha, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    
    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)

        lmd = np.random.beta(mixup_alpha, mixup_alpha)
        perm = torch.randperm(x.shape[0]).to(device)
        x2 = x[perm, :]
        y2 = y[perm]

        optimizer.zero_grad()
        out = model(lmd * x + (1.0 - lmd) * x2)
        loss = lmd * lossfun(out, y) + (1.0 - lmd) * lossfun(out, y2)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        _, pred = torch.max(out.detach(), 1)
        total_loss += loss.item() * x.size(0)
        total_acc += lmd * torch.sum(pred == y) + (1 - lmd) * torch.sum(pred == y2)

    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc  = total_acc / len(train_loader.dataset)
    return avg_acc, avg_loss


def train(model, optimizer, lr_scheduler, train_loader, val_loader, n_epochs, device):
    lossfun = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(n_epochs)):
        train_acc, train_loss = train_1epoch(
            model, train_loader, lossfun, optimizer, lr_scheduler, device
        )
        
        val_acc, val_loss = validate_1epoch(model, val_loader, lossfun, device)

        print(
            f"epoch={epoch}, train loss = {train_loss}, train_accuracy = {train_acc}",
            f"val loss={val_loss}, val accuracy = {val_acc}"
        )

if __name__ == "__main__":
    data_dir = "."
    batch_size = 32
    dryrun = False
    device="cuda:0"
    output= True
    n_epochs = 10

    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay=0.0001)
    
    train_loader, val_loader = setup_train_val_loaders(
        data_dir, batch_size , dryrun
    )
    n_iterations =len(train_loader) * n_epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iterations)

    train(
        model, optimizer, lr_scheduler, train_loader, val_loader, n_epochs=n_epochs, device=device
    )

    if output:
        predict_subsec5(data_dir, ".", model, batch_size=1)


    