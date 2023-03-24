import torch
import numpy as np
import inspect
import torchvision.transforms as trans
from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
import torch.nn.functional as F



class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=0, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
        return focal_loss

def train(args):
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    global_step = 0
    num_epoch = 20
    FL = FocalLoss()
    learning_rate = 1e-3 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Detector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss = torch.nn.BCEWithLogitsLoss().to(device)
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_detection_data('dense_data/train', num_workers=4, transform=transform)
    for epoch in range(num_epoch):
        loss_array =[]
        model.train()
        for img, label,_ in train_data:
            img, label = img.to(device), label.to(device)
            logit = model(img)
            loss_val = FL.forward(logit, label)
            loss_array.append(loss_val.cpu().detach().numpy())
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        print("------------------------------------------------------------")
        print("Epoch: " + str(epoch+1))
        print("Loss: " + "{0:.4f}".format(np.mean(loss_array)))
    save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor(), ToHeatmap()])')
    args = parser.parse_args()
    train(args)

