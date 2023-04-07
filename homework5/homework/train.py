from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms
import inspect
from os import path
 
def train(args):
    global_step = 0
num_epoch = 50
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Planner().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
det_loss = torch.nn.MSELoss()
transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
train_data = load_data('drive_data', num_workers=4, transform=transform)

for epoch in range(num_epoch):
    epoch_loss = 0
    model.train()
    for img, gt_det in train_data:
        img, gt_det = img.to(device), gt_det.to(device)
        det = model(img)
        p_det = torch.sigmoid(det * (1-2*gt_det))
        det_loss_val = (det_loss(det, gt_det)*p_det).mean() / p_det.mean()
        loss_val = det_loss_val
        epoch_loss += loss_val.item()
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        global_step += 1
    scheduler.step()
    print("------------------------------------------------------------")
    print("Epoch: " + str(epoch+1))
    print(f"Epoch {epoch+1}/{num_epoch} - Loss: {epoch_loss/len(train_data):.4f}")

save_model(model)


def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor()])')
    args = parser.parse_args()
    train(args)
