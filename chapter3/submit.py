from transform import setup_center_crop_transform
import os
import torchvision
import torch.utils
import numpy as np
from tqdm import tqdm

def setup_test_loader(data_dir, batch_size, dryrun):
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, "test"), transform=setup_center_crop_transform()
    )
    image_ids = [
        os.path.splitext(os.path.basename(path))[0] for path, _ in dataset.imgs
    ]

    if dryrun:
        dataset = torch.utils.data.Subset(dataset, range(0, 100))
        image_ids = image_ids[:100]
        
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=8
    )
    return loader, image_ids
    
def predict(model, loader, device):
    pred_fun = torch.nn.Softmax(dim=1)
    preds = []
    
    for x, _ in tqdm(loader):
        with torch.set_grad_enabled(False):
            x = x.to(device)
            y = pred_fun(model(x))
        
        y = y.cpu().numpy()
        y = y[:, 1]
        preds.append(y)
    preds = np.concatenate(preds)
    return preds

def write_prediction(image_ids, prediction, out_path):
    with open(out_path, "w") as f:
        f.write("id,label\n")
        for i, p in zip(image_ids, prediction):
            f.write("{},{}\n".format(i, p))
    
def write_prediction_with_clip(image_ids, prediction, clip_threshold, out_path):
    with open(out_path, "w") as f:
        f.write("id,label\n")
        for i, p in zip(image_ids, prediction):
            p = np.clip(p, clip_threshold, 1.0 - clip_threshold)
            f.write("{},{}\n".format(i, p))

def predict_subsec5(
    data_dir, out_dir, model, batch_size, dryrun=False, device="cuda:0"
):
    test_loader, image_ids = setup_test_loader(
        data_dir, batch_size, dryrun=dryrun
    )

    preds = predict(model, test_loader, device)
    write_prediction_with_clip(image_ids, preds, 0.0025, "out.csv")

