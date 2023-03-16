import torch
import numpy as np
import torch
import inspect
from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    num_epochs = 20
    learning_rate = 0.001
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device = ', device)
    model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_dense_data('dense_data/train', num_workers=4)
    train_data_transformed = load_dense_data('dense_data/train', num_workers=4, transform=transform)
    valid_data = load_dense_data('dense_data/valid', num_workers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss = torch.nn.BCEWithLogitsLoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    global_step = 0

    for epoch in range(num_epochs):
        train_accuracy = []
        train_accuracy_value = []
        model.train()
        confusion_matrix = ConfusionMatrix()
        for image, label in train_data:
            image = image.to(device)
            label = torch.tensor(label, dtype=torch.long, device=device)
            pred = model(image)
            loss_val = loss(pred, label)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            train_accuracy = accuracy(o,labels)
            train_accuracy_value.append(train_accuracy.cpu().detach().numpy())
            global_step += 1
        for image, label in train_data_transformed:
            image = image.to(device)
            label = torch.tensor(label, dtype=torch.long, device=device)
            pred = model(image)
            loss_val = loss(pred, label)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            train_accuracy = accuracy(o,labels)
            train_accuracy_value.append(train_accuracy.cpu().detach().numpy())
            global_step += 1
        model.eval()
        total_step = 0
        accuracy = 0
        for i, (image, label) in enumerate(valid_data):
            image, label = image.to(device), label.to(device)
            pred = model(image)
            accuracy = accuracy + (pred.argmax(1) == label).float().mean().item()
            total_step += 1
            print("------------------------------------------------------------")
            print("Epoch: " + str(epoch+1))
            print("Accuracy: " + "{0:.3f}".format(accuracy/total_step))
        scheduler.step(loss_val)
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

    args = parser.parse_args()
    train(args)
